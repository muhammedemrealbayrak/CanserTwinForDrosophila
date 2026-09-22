"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: web_dashboard/app_server.py
======================================================
Yazar: Web Entegrasyon & Canlı Telemetri Ekibi
Açıklama:
    Çok sekmeli (Tabbed) Laboratuvar Konsolu ve REST API Sunucusu (Port: 8060).
    3D Hücresel Simülasyon, 22+ Biyoaktif Bileşik Veri Bankası,
    De Novo Liderlik Tablosu ve ML Analitik verilerini sunar.
"""

import sys
import os
import json
import sqlite3
import webbrowser
import threading
import io
import csv
import datetime
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional
import numpy as np

# Proje kök dizinini ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from platform_engine import DrosophilaInSilicoPlatform
from data_manager import DrosophilaDataManager
from pipeline.online_data_fetcher import (
    fetch_from_pubchem,
    save_compound_to_db,
    generate_multi_neuron_circuit,
    MULTI_NEURON_JSON_PATH
)
from pipeline.benchmark_engine import InSilicoBenchmarkEngine
from denovo_ai.molecule_generator import DeNovoMoleculeGenerator, compute_lipinski_rules
from denovo_ai.dose_optimizer import AutonomousDoseOptimizer
from denovo_ai.cocktail_generator import cocktail_synthesizer
from pipeline.docking_engine import docking_engine
from pipeline.clinical_trial_engine import clinical_trial_engine
from pipeline.translational_engine import translational_engine
from pipeline.pdf_report_generator import generate_pdf_report

HOST = "127.0.0.1"
PORT = 8060

WHOLE_BRAIN_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flywire_whole_brain_140k.json")

SIM_LOCK = threading.Lock()
platform_instance: DrosophilaInSilicoPlatform = None
data_manager = DrosophilaDataManager()
benchmark_engine = InSilicoBenchmarkEngine()
denovo_generator = DeNovoMoleculeGenerator()
dose_optimizer = AutonomousDoseOptimizer()

SVG_CACHE: Dict[str, str] = {}


def render_molecule_svg(smiles: str) -> str:
    if not smiles:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200"><text x="50%" y="50%" fill="#94a3b8" text-anchor="middle" font-family="sans-serif">SMILES Belirtilmedi</text></svg>'
    if smiles in SVG_CACHE:
        return SVG_CACHE[smiles]
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200"><text x="50%" y="50%" fill="#ff2a6d" text-anchor="middle" font-family="sans-serif">Geçersiz SMILES Yapısı</text></svg>'
        d = rdMolDraw2D.MolDraw2DSVG(320, 220)
        opts = d.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2.2
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        svg = svg.replace("#000000", "#f8fafc")
        SVG_CACHE[smiles] = svg
        return svg
    except Exception as ex:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200"><text x="50%" y="50%" fill="#ff2a6d" text-anchor="middle" font-family="sans-serif">Çizim Hatası: {ex}</text></svg>'


def init_platform(molecule_name: str = "CC1=NC=C(C=C1)CCN(C)C(=O)CF"):
    global platform_instance
    platform_instance = DrosophilaInSilicoPlatform(
        active_compound_smiles_or_name=molecule_name,
        initial_tumor_burden=150,
        domain_size_um=500.0,
        grid_resolution=16
    )


init_platform()


def serialize_cells(platform):
    """3D WebGL renderı için aktif kanser hücrelerini ve devriye hemositlerini serileştirir."""
    cancer_list = [
        {
            "id": c.id,
            "pos": [round(float(x), 1) for x in c.position],
            "health": round(c.health, 1),
            "clone_type": getattr(c, "clone_type", "sensitive"),
            "resistance": round(getattr(c, "resistance_score", 0.0), 2),
            "is_resistant": (getattr(c, "clone_type", "sensitive") != "sensitive" or getattr(c, "resistance_score", 0.0) > 0.4),
            "eiger_level": round(float(getattr(c, "eiger_level", 0.0)), 2),
            "jnk_stress": round(float(getattr(c, "jnk_stress", 0.0)), 2),
            "isc_stemness": round(float(getattr(c, "isc_stemness", 1.0)), 2),
            "ras_hijack": bool(getattr(c, "ras_hijack_active", True)),
            "caspase_lysis": bool(getattr(c, "caspase_lysis_active", False))
        }
        for c in platform.cancer_cells
        if c.state.value not in ["apoptotic", "lysed"]
    ]
    is_shielded = bool(
        getattr(platform, "active_cocktail", None) and
        getattr(platform, "active_cocktail", {}).get("id") == "neuro_immune_quad_shield"
    )
    hemocyte_list = [
        {
            "id": h.id,
            "subtype": h.subtype.value,
            "pos": [round(float(x), 1) for x in h.position],
            "cholinergic": round(float(getattr(h, "cholinergic_activation", 0.0)), 2),
            "exhaustion": round(float(getattr(h, "exhaustion_index", 0.0)), 2),
            "kills": int(getattr(h, "kills_count", 0)),
            "shielded": is_shielded
        }
        for h in platform.hemocyte_agents
    ]
    return cancer_list, hemocyte_list


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Gelişmiş Laboratuvar Konsolu HTTP ve REST API Yöneticisi."""

    def _send_json(self, data: any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
            try:
                with open(html_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"HTML read error: {e}")

        elif self.path == "/api/compounds" or self.path.startswith("/api/compounds?"):
            # Veritabanından tüm bileşikleri ve uygulanan filtreleri çek
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            filters = {}
            if "category" in qs and qs["category"][0]:
                filters["category"] = qs["category"][0]
            if "potency" in qs and qs["potency"][0]:
                filters["potency"] = qs["potency"][0]
            if "safety" in qs and qs["safety"][0]:
                filters["safety"] = qs["safety"][0]
            if "stage" in qs and qs["stage"][0]:
                filters["stage"] = qs["stage"][0]
            if "search" in qs and qs["search"][0]:
                filters["search"] = qs["search"][0]
            compounds = data_manager.get_all_compounds(filters=filters if filters else None)
            self._send_json(compounds)

        elif self.path == "/api/ml_analytics":
            # ML Model metrikleri, öznitelik önemleri ve çapraz doğrulama
            summary = data_manager.get_ml_analytics()
            self._send_json(summary)

        elif self.path == "/api/benchmarks":
            # Liderlik tablosunu en yuksek fitness skoruna gore sirali cek
            rows = benchmark_engine.get_leaderboard(50)
            self._send_json(rows)

        elif self.path == "/api/connectome/whole_brain":
            # FlyWire FAFB v783 139,248 Tam Beyin Noron Koordinatlari & Transmitter Haritasi
            if os.path.exists(WHOLE_BRAIN_JSON_PATH):
                with open(WHOLE_BRAIN_JSON_PATH, "r", encoding="utf-8") as f:
                    wb_data = json.load(f)
                self._send_json(wb_data)
            else:
                self.send_error(404, "Whole brain connectome data not found")

        elif self.path == "/api/connectome/stats":
            # Tum beyin istatistikleri ve noropil dagilimi
            self._send_json({
                "dataset": "FlyWire FAFB v783 Whole Brain Connectome (Nature 2024)",
                "total_neurons": 139248,
                "total_synapses": 54500000,
                "superclasses": {
                    "Optik Loblar (Gorme)": 77541,
                    "Merkezi Beyin": 32383,
                    "Duyusal / Kemosensor": 16907,
                    "Gorsel Projeksiyon": 8038,
                    "Asendan (Yukselen)": 1750,
                    "Desendan (Inen Motor & Immun)": 1303,
                    "Duyu-Asendan": 612,
                    "Gorsel Santrifuj": 524,
                    "Motor": 110,
                    "Endokrin / Norosekresyon": 80
                },
                "neurotransmitters": {
                    "ACh (Asetilkolin - Eksitator)": 86193,
                    "Glutamat": 24875,
                    "GABA (Inhibitor)": 19171,
                    "Dopamin (Modulator)": 5909,
                    "Serotonin (5-HT)": 2282,
                    "Oktopamin (OA)": 216
                }
            })

        elif self.path == "/api/flywire_neuron":
            # FlyWire FAFB v783 KCg-m 720575940608530955 3D iskelet verisi
            neuron_path = os.path.join(os.path.dirname(__file__), "..", "data", "flywire_kcg_720575940608530955.json")
            if os.path.exists(neuron_path):
                with open(neuron_path, "r", encoding="utf-8") as f:
                    neuron_data = json.load(f)
                self._send_json(neuron_data)
            else:
                self.send_error(404, "FlyWire neuron model not found")

        elif self.path == "/api/cocktails":
            # Sinerjik Kombinasyon Terapisi Kokteylleri (Temel + AI Sentezlenmiş)
            try:
                ai_cocktails = cocktail_synthesizer.load_saved_cocktails()
                for ac in ai_cocktails:
                    cid = ac.get("id")
                    if cid and cid not in platform_instance.COCKTAIL_REGIMENS:
                        platform_instance.COCKTAIL_REGIMENS[cid] = ac
            except Exception as e:
                print(f"[WARN] AI kokteylleri yuklenirken hata: {e}")
            cocktails = list(platform_instance.COCKTAIL_REGIMENS.values())
            self._send_json(cocktails)

        elif self.path == "/api/treatment_modalities":
            # Onkolojik Tedavi Protokolleri ve Kurtarma Rejimleri
            modalities = list(platform_instance.TREATMENT_MODALITIES.values())
            self._send_json(modalities)

        elif self.path == "/api/connectome/multi_neuron":
            # FlyWire FAFB v783 4'lü Sinaptik Devre (KCg-m + DAN-PPL1 + MBON-gamma1 + PN-AL)
            if os.path.exists(MULTI_NEURON_JSON_PATH):
                with open(MULTI_NEURON_JSON_PATH, "r", encoding="utf-8") as f:
                    circuit_data = json.load(f)
            else:
                circuit_data = generate_multi_neuron_circuit()
            self._send_json(circuit_data)

        elif self.path == "/api/docking/receptors":
            # 3D Moleküler Kenetlenme Reseptör Bağlanma Havuzları
            receptors = docking_engine.get_available_receptors()
            self._send_json(receptors)

        elif self.path == "/api/clinical_trial/protocols":
            # Sanal Klinik Deney Protokolleri ve Tedavi Kolları
            protocols = clinical_trial_engine.get_trial_protocols()
            self._send_json(protocols)

        elif self.path == "/api/translational/homologs":
            # Drosophila <-> İnsan Onkogen Homoloji Veritabanı
            homologs = translational_engine.get_homologs()
            self._send_json({"homolog_pairs": homologs, "status": "success"})

        elif self.path == "/api/translational/tcga_cohorts":
            # TCGA İnsan Klinik Kanser Kohortları
            cohorts = translational_engine.get_tcga_cohorts()
            self._send_json({"cohorts": cohorts, "status": "success"})

        elif self.path == "/api/paper/metadata":
            # Akademik Makale Başlık ve Özet Bilgisi
            bib_path = os.path.join(os.path.dirname(__file__), "..", "paper", "references.bib")
            bib_text = ""
            if os.path.exists(bib_path):
                with open(bib_path, "r", encoding="utf-8") as f:
                    bib_text = f.read()

            self._send_json({
                "title": "In Silico Whole-Brain Digital Twin of Drosophila melanogaster Discovers Synergistic Neuro-Immune Anti-Cancer Therapies Translated to Human Onco-Genomics",
                "authors": [
                    {"name": "Muhammed Emre Albayrak", "affiliation": "In Silico Oncology & Computational Neurobiology Laboratory, Istanbul, Turkey", "email": "contact@memrealbayrak.com"}
                ],
                "target_journals": ["bioRxiv", "Nature Digital Medicine", "Cell Systems"],
                "status": "Preprint Ready (Peer-Review Draft)",
                "date": "September 2026",
                "bibtex": bib_text
            })

        elif self.path == "/api/paper/content":
            # Akademik Makale Tam Metni (Markdown)
            paper_md_path = os.path.join(os.path.dirname(__file__), "..", "paper", "drosophila_digital_twin_preprint.md")
            content = ""
            if os.path.exists(paper_md_path):
                with open(paper_md_path, "r", encoding="utf-8") as f:
                    content = f.read()
            self._send_json({"content": content, "format": "markdown", "status": "success"})

        elif self.path.startswith("/api/paper/download"):
            # Makale Dosyası İndirme (.tex, .md, .bib)
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["markdown"])[0].lower()
            paper_dir = os.path.join(os.path.dirname(__file__), "..", "paper")

            if fmt == "latex" or fmt == "tex":
                fpath = os.path.join(paper_dir, "drosophila_digital_twin_preprint.tex")
                fname = "drosophila_digital_twin_preprint.tex"
                ctype = "application/x-tex"
            elif fmt == "bibtex" or fmt == "bib":
                fpath = os.path.join(paper_dir, "references.bib")
                fname = "references.bib"
                ctype = "text/plain; charset=utf-8"
            else:
                fpath = os.path.join(paper_dir, "drosophila_digital_twin_preprint.md")
                fname = "drosophila_digital_twin_preprint.md"
                ctype = "text/markdown; charset=utf-8"

            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404, "Dosya bulunamadı")

        elif self.path.startswith("/api/export_pdf_report"):
            # Preklinik İlaç ve Simülasyon Sonuçları PDF Raporu Dışa Aktarımı
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            rtype = qs.get("type", ["full"])[0].lower()

            with SIM_LOCK:
                active_snap = platform_instance.get_current_snapshot() if platform_instance else None
                active_drug = platform_instance.active_drug if platform_instance else None

            try:
                pdf_bytes = generate_pdf_report(
                    report_type=rtype,
                    active_telemetry=active_snap,
                    active_drug_profile=active_drug
                )
                now_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"Drosophila_In_Silico_Oncology_Report_{now_ts}.pdf"

                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as ex:
                print(f"[ERROR] PDF raporu olusturulurken hata: {ex}")
                self.send_error(500, f"PDF Olusturma Hatasi: {ex}")

        elif self.path.startswith("/api/export_dataset"):
            # Veri setini CSV veya JSON olarak dışa aktar
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            fmt = qs.get("format", ["csv"])[0].lower()
            compounds = data_manager.get_all_compounds()

            if fmt == "json":
                body = json.dumps(compounds, indent=2, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="drosophila_anticancer_dataset_{len(compounds)}compounds.json"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                # CSV Export
                output = io.StringIO()
                if compounds:
                    writer = csv.DictWriter(output, fieldnames=list(compounds[0].keys()))
                    writer.writeheader()
                    for c in compounds:
                        writer.writerow(c)
                body = output.getvalue().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="drosophila_anticancer_dataset_{len(compounds)}compounds.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        elif self.path.startswith("/api/molecule_svg"):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            smiles = qs.get("smiles", [""])[0].strip()
            svg_content = render_molecule_svg(smiles)
            body = svg_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if self.path == "/api/step":
            dt = float(payload.get("dt", 1.0))
            with SIM_LOCK:
                snap = platform_instance.step(dt_seconds=dt)
                cancer_list, hemocyte_list = serialize_cells(platform_instance)

            self._send_json({
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list,
                "lysed_bursts": snap.get("lysed_bursts", [])
            })

        elif self.path == "/api/set_molecule":
            mol_name = payload.get("molecule", "DeNovo_Champion")
            force_reset = payload.get("reset", True)
            if mol_name == "DeNovo_Champion":
                mol_key = "CC1=NC=C(C=C1)CCN(C)C(=O)CF"
            else:
                mol_key = mol_name

            with SIM_LOCK:
                viable = [c for c in platform_instance.cancer_cells if c.state.value not in ["apoptotic", "lysed"]]
                if len(viable) == 0 or not platform_instance.host_alive or force_reset:
                    platform_instance.reset(molecule_name_or_smiles=mol_key)
                else:
                    platform_instance.set_active_drug(mol_key)
                prof = platform_instance.active_drug
                snap = platform_instance.get_current_snapshot()
                cancer_list, hemocyte_list = serialize_cells(platform_instance)

            self._send_json({
                "status": "updated",
                "name": prof.name,
                "smiles": prof.smiles,
                "kd_micromolar": prof.kd_micromolar,
                "logP": prof.logP,
                "qsar_toxicity_risk": prof.qsar_toxicity_risk,
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list
            })

        elif self.path == "/api/set_cocktail":
            cid = payload.get("cocktail_id", "immuno_mek_synergy")
            custom_data = payload.get("cocktail_data")
            with SIM_LOCK:
                platform_instance.reset()
                reg = platform_instance.set_cocktail(cid, custom_cocktail=custom_data)
                snap = platform_instance.get_current_snapshot()
                cancer_list, hemocyte_list = serialize_cells(platform_instance)
            self._send_json({
                "status": "cocktail_updated",
                "cocktail": reg,
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list
            })

        elif self.path == "/api/cocktails/generate":
            # Otonom Yapay Zeka Sinerjik Kokteyl Sentezi
            objective = payload.get("objective", "mek_bypass_triple")
            comp_count = int(payload.get("component_count", 3))
            try:
                new_cocktail = cocktail_synthesizer.generate_cocktail(
                    objective=objective,
                    component_count=comp_count
                )
                cid = new_cocktail.get("id")
                if cid:
                    platform_instance.COCKTAIL_REGIMENS[cid] = new_cocktail
                self._send_json({
                    "status": "success",
                    "cocktail": new_cocktail,
                    "all_cocktails": list(platform_instance.COCKTAIL_REGIMENS.values())
                })
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/cocktails/evaluate":
            # Özel kombinasyon ve dozaj için farmakolojik sinerji analizi
            components = payload.get("components", [])
            target_kill = float(payload.get("target_kill", 0.95))
            try:
                eval_res = cocktail_synthesizer.evaluate_combination_pharmacology(components, target_kill=target_kill)
                self._send_json({"status": "success", "evaluation": eval_res})
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/set_treatment_modality":
            mod_id = payload.get("modality", "targeted_small_molecule")
            force_reset = payload.get("reset", False)
            with SIM_LOCK:
                viable = [c for c in platform_instance.cancer_cells if c.state.value not in ["apoptotic", "lysed"]]
                if len(viable) == 0 or not platform_instance.host_alive or force_reset:
                    platform_instance.reset(modality=mod_id)
                mod_res = platform_instance.set_treatment_modality(mod_id)
                snap = platform_instance.get_current_snapshot()
                cancer_list, hemocyte_list = serialize_cells(platform_instance)
            self._send_json({
                "status": "modality_updated",
                "modality": mod_res,
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list
            })

        elif self.path == "/api/apply_radiation":
            dose = float(payload.get("dose_gy", 8.0))
            with SIM_LOCK:
                rad_res = platform_instance.apply_radiation_pulse(dose_gy=dose)
                snap = platform_instance.get_current_snapshot()
                cancer_list, hemocyte_list = serialize_cells(platform_instance)
            self._send_json({
                "status": "radiation_applied",
                "radiation_result": rad_res,
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list,
                "lysed_bursts": snap.get("lysed_bursts", [])
            })

        elif self.path == "/api/set_dose":
            dose = float(payload.get("dose_uM", 2.5))
            with SIM_LOCK:
                platform_instance.drug_dose_uM = dose
                snap = platform_instance.get_current_snapshot()
            self._send_json({"status": "dose_updated", "dose": dose, "telemetry": snap})

        elif self.path == "/api/reset":
            with SIM_LOCK:
                snap = platform_instance.reset()
                cancer_list, hemocyte_list = serialize_cells(platform_instance)

            self._send_json({
                "status": "reset",
                "telemetry": snap,
                "cancer_cells": cancer_list,
                "hemocyte_agents": hemocyte_list
            })
        elif self.path == "/api/online/fetch_compound":
            query = payload.get("query", "").strip()
            if not query:
                self._send_json({"status": "error", "message": "Bileşik adı veya PubChem CID belirtilmedi."}, status=400)
            else:
                try:
                    comp = fetch_from_pubchem(query)
                    save_compound_to_db(comp)
                    self._send_json({"status": "success", "compound": comp})
                except Exception as ex:
                    self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/predict_qsar":
            smiles = payload.get("smiles", "").strip()
            if not smiles:
                self._send_json({"status": "error", "message": "SMILES dizilimi girilmedi."}, status=400)
            else:
                try:
                    from pipeline.pubchem_connector import PubChemConnector
                    conn_qsar = PubChemConnector()
                    prof = conn_qsar.parse_molecule(smiles)
                    pot_class = "Yüksek Potans (<1 µM)" if prof.kd_micromolar < 0.25 else ("Orta Potans (1-25 µM)" if prof.kd_micromolar <= 2.5 else "Düşük Potans (>25 µM)")
                    lip_rules = compute_lipinski_rules(prof)
                    self._send_json({
                        "status": "success",
                        "lipinski": lip_rules,
                        "profile": {
                            "smiles": prof.smiles,
                            "canonical_smiles": prof.canonical_smiles,
                            "name": prof.name,
                            "molecular_weight": prof.molecular_weight,
                            "logP": prof.logP,
                            "tpsa": prof.tpsa,
                            "hbd": prof.h_bond_donors,
                            "hba": prof.h_bond_acceptors,
                            "rotb": prof.rotatable_bonds,
                            "kd_micromolar": prof.kd_micromolar,
                            "qsar_toxicity_risk": prof.qsar_toxicity_risk,
                            "bioavailability_score": prof.bioavailability_score,
                            "potency_class": pot_class,
                            "lipinski": lip_rules
                        }
                    })
                except Exception as ex:
                    self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/denovo/generate":
            # De Novo Yapay Zeka Analog Tasarımı ve Optimizasyonu
            parent = payload.get("parent_smiles", "Nicotine").strip()
            strategy = payload.get("strategy", "balanced").strip()
            count = int(payload.get("count", 4))
            try:
                candidates = denovo_generator.generate_targeted_analogs(
                    parent_smiles_or_name=parent,
                    strategy=strategy,
                    count=count
                )
                self._send_json({"status": "success", "candidates": candidates})
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/ai/predict_multidrug":
            # Yapay Zeka Coklu Ilac ve Otonom Dozaj Tahmin Motoru
            pathway = payload.get("pathway", "ras_mek")
            count = int(payload.get("count", 4))
            try:
                res = dose_optimizer.predict_multidrug_batch(pathway_key=pathway, count=count)
                self._send_json(res)
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/ai/optimize_dose":
            # Tekil molekul veya aktif ilac icin otonom dozaj hesaplama
            target = payload.get("molecule") or payload.get("smiles") or (platform_instance.active_drug.smiles if platform_instance else "Nicotine")
            try:
                dose_res = dose_optimizer.predict_for_molecule(target)
                self._send_json({"status": "success", **dose_res})
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/benchmark/run":
            # In silico benchmark kosusu yurut ve liderlik tablosunu guncelle
            mol = payload.get("molecule", "").strip()
            all_flag = payload.get("all", False)
            try:
                if all_flag:
                    benchmark_engine.populate_comprehensive_benchmarks()
                    self._send_json({"status": "success", "benchmarks": benchmark_engine.get_leaderboard(50)})
                elif mol:
                    rec = benchmark_engine.run_benchmark_for_molecule(mol)
                    self._send_json({"status": "success", "record": rec, "benchmarks": benchmark_engine.get_leaderboard(50)})
                else:
                    current_mol = platform_instance.active_drug.smiles
                    rec = benchmark_engine.run_benchmark_for_molecule(current_mol)
                    self._send_json({"status": "success", "record": rec, "benchmarks": benchmark_engine.get_leaderboard(50)})
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/benchmark/record":
            # Canli 3D simulatorden gelen telemetriyi liderlik tablosuna kaydet
            name = payload.get("molecule_name", platform_instance.active_drug.name)
            trig = float(payload.get("time_to_trigger_s", 1.0))
            hem = int(payload.get("hemocytes_produced", 65))
            clr = float(payload.get("tumor_clearance_pct", 100.0))
            tox = float(payload.get("toxicity_pct", 5.0))

            fit = 0.35 * clr + 0.30 * (100.0 - tox) + 0.20 * (10.0 / (trig + 0.1)) + 0.15 * min(100.0, hem * 1.5)
            fit = round(float(np.clip(fit, 0.0, 100.0)), 1)

            rec = {
                "molecule_name": name,
                "time_to_trigger_s": trig,
                "hemocytes_produced": hem,
                "tumor_clearance_pct": clr,
                "toxicity_pct": tox,
                "fitness_score": fit
            }
            try:
                benchmark_engine.save_benchmark_record(rec)
                self._send_json({"status": "success", "record": rec, "benchmarks": benchmark_engine.get_leaderboard(50)})
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/docking/run":
            # 3D Moleküler Kenetlenme Simülasyonu
            rec_key = payload.get("receptor", "nAChR_alpha7")
            mol_input = payload.get("molecule", "CC1=NC=C(C=C1)CCN(C)C(=O)CF")
            try:
                result = docking_engine.run_docking(receptor_key=rec_key, molecule_or_smiles=mol_input)
                self._send_json(result)
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/clinical_trial/run":
            # Sanal Klinik Deney ve Kaplan-Meier Simülasyonu
            cohort_size = int(payload.get("cohort_size", 100))
            selected_arms = payload.get("selected_arms", None)
            time_horizon = int(payload.get("time_horizon_days", 60))
            try:
                results = clinical_trial_engine.run_trial(
                    cohort_size_per_arm=cohort_size,
                    selected_arms=selected_arms,
                    time_horizon_days=time_horizon
                )
                self._send_json(results)
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/translational/predict":
            # Drosophila -> İnsan Translasyonel İlaç & Kokteyl Aktarılabilirlik Analizi
            is_cocktail = payload.get("is_cocktail", False)
            cohort = payload.get("tcga_cohort") or payload.get("tcga_cohort_id") or "TCGA-PAAD"
            cocktail_id = payload.get("cocktail_id")
            cocktail_data = payload.get("cocktail_data")
            mol = payload.get("molecule") or payload.get("molecule_name") or "DeNovo_Champion_Mol1"

            if is_cocktail or cocktail_id or cocktail_data or (mol and str(mol).startswith("cocktail_")) or (mol in platform_instance.COCKTAIL_REGIMENS):
                target_c = cocktail_data or cocktail_id or mol
                try:
                    result = translational_engine.translate_cocktail(
                        cocktail_data_or_id=target_c,
                        target_tcga_cohort=cohort
                    )
                    self._send_json(result)
                except Exception as ex:
                    self._send_json({"status": "error", "message": str(ex)}, status=500)
            else:
                target_gene = payload.get("target_gene")
                try:
                    result = translational_engine.translate_molecule(
                        molecule_name_or_smiles=mol,
                        target_tcga_cohort=cohort,
                        target_gene=target_gene
                    )
                    self._send_json(result)
                except Exception as ex:
                    self._send_json({"status": "error", "message": str(ex)}, status=500)

        elif self.path == "/api/translational/predict_cocktail":
            # Sinerjik Kokteyller İçin Özel Çok Hedefli Translasyonel Köprü
            cohort = payload.get("tcga_cohort") or payload.get("tcga_cohort_id") or "TCGA-PAAD"
            cocktail_id = payload.get("cocktail_id", "kras_g12d_vertical_blockade")
            cocktail_data = payload.get("cocktail_data")
            try:
                result = translational_engine.translate_cocktail(
                    cocktail_data_or_id=cocktail_data or cocktail_id,
                    target_tcga_cohort=cohort
                )
                self._send_json(result)
            except Exception as ex:
                self._send_json({"status": "error", "message": str(ex)}, status=500)

        else:
            self.send_error(404, "Endpoint bulunamadı")

    def log_message(self, format, *args):
        return


def start_server():
    server = HTTPServer((HOST, PORT), DashboardRequestHandler)
    print("=" * 80)
    print("  DROSOPHILA IN SILICO NEURO-IMMUNE DIGITAL TWIN PLATFORM")
    print(f"  Web Konsolu Yayında: http://{HOST}:{PORT}")
    print("  Tarayıcınız otomatik olarak açılıyor...")
    print("=" * 80)

    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatıldı.")
        server.server_close()


if __name__ == "__main__":
    start_server()
