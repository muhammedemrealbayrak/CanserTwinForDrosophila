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
            "is_resistant": (getattr(c, "clone_type", "sensitive") != "sensitive" or getattr(c, "resistance_score", 0.0) > 0.4)
        }
        for c in platform.cancer_cells
        if c.state.value not in ["apoptotic", "lysed"]
    ]
    hemocyte_list = [
        {"id": h.id, "subtype": h.subtype.value, "pos": [round(float(x), 1) for x in h.position]}
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

        elif self.path == "/api/compounds":
            # Veritabanından tüm 22 bileşiği çek
            compounds = data_manager.get_all_compounds()
            self._send_json(compounds)

        elif self.path == "/api/ml_analytics":
            # ML Model metrikleri ve analitiği
            json_path = os.path.join(data_manager.RESULTS_DIR, "classification_summary.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                summary = data_manager.ml_summary
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
