# -*- coding: utf-8 -*-
"""
===============================================================================
DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D & C# .NET EXPORTER
Module: unity_exporter.py
Author: Muhammed Emre Albayrak
License: MIT
===============================================================================
Extracts FlyWire connectome circuits, 3D anatomical neuropil volumes, and tumor
microenvironment agents into Unity 3D scene JSON and packages C# source assets.
"""

import os
import json
import zipfile
import io
import math
from typing import Dict, Any, List


class UnityConnectomeExporter:
    """
    Drosophila dijital ikiz modelini Unity 3D ve C# .NET simülasyon ortamına
    aktaran ve JSON sahne tanımı ile C# kaynak dosyalarını paketleyen modül.
    """

    NEUROPIL_ANCHORS = {
        "AntennalLobe_Left": {"x": -70.0, "y": 110.0, "z": -30.0, "radius": 28.0, "color": "#ff007f", "function": "Koku ve kemosensör duyu entegrasyonu"},
        "AntennalLobe_Right": {"x": 70.0, "y": 110.0, "z": -30.0, "radius": 28.0, "color": "#ff007f", "function": "Koku ve kemosensör duyu entegrasyonu"},
        "MushroomBody_Left": {"x": -120.0, "y": 40.0, "z": 20.0, "radius": 36.0, "color": "#00f0ff", "function": "Hafıza, öğrenme ve nörogenez nişi (Onkogenez odağı)"},
        "MushroomBody_Right": {"x": 120.0, "y": 40.0, "z": 20.0, "radius": 36.0, "color": "#00f0ff", "function": "Hafıza, öğrenme ve nörogenez nişi"},
        "CentralComplex_FanShaped": {"x": 0.0, "y": 15.0, "z": 40.0, "radius": 32.0, "color": "#ffb703", "function": "Yön bulma, motor koordinasyon ve lokomosyon kontrolü"},
        "CentralComplex_Ellipsoid": {"x": 0.0, "y": 35.0, "z": 20.0, "radius": 24.0, "color": "#fb8500", "function": "Uzamsal oryantasyon ve görsel pusula"},
        "OpticLobe_Left": {"x": -210.0, "y": -10.0, "z": 0.0, "radius": 55.0, "color": "#7000ff", "function": "Görsel lob (Medulla, Lobula ve Lamina fotoreseptörleri)"},
        "OpticLobe_Right": {"x": 210.0, "y": -10.0, "z": 0.0, "radius": 55.0, "color": "#7000ff", "function": "Görsel lob"},
        "SubesophagealZone": {"x": 0.0, "y": -90.0, "z": -60.0, "radius": 42.0, "color": "#00ff9d", "function": "Tat alma, beslenme ve otonomik visseral kontrol"},
        "ParsIntercerebralis": {"x": 0.0, "y": 80.0, "z": 50.0, "radius": 22.0, "color": "#ff0055", "function": "Nöroendokrin merkez (İnsülin benzeri peptitler / DILP salgısı)"}
    }

    def __init__(self, csharp_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csharp_dir = csharp_dir or os.path.join(base_dir, "unity_integration", "csharp")
        self.package_dir = os.path.join(base_dir, "unity_integration", "package")

    def generate_scene_json(self, neuron_count: int = 350, tumor_cells: int = 50, hemocytes: int = 25) -> Dict[str, Any]:
        """
        Unity 3D sahnelerinde doğrudan GameObject hiyerarşisi oluşturabilecek
        tam tanımlı uzamsal ve biyolojik sahne sözlüğü üretir.
        """
        import random
        rng = random.Random(42)

        # 1. Nöropil Hacim Küpleri / Küreleri
        neuropil_nodes = []
        for name, meta in self.NEUROPIL_ANCHORS.items():
            neuropil_nodes.append({
                "id": name,
                "label": name.replace("_", " "),
                "center": {"x": meta["x"], "y": meta["y"], "z": meta["z"]},
                "bounding_radius_um": meta["radius"],
                "color_hex": meta["color"],
                "biological_function": meta["function"]
            })

        # 2. Nöron Ajanları & Sinapslar
        neurons = []
        region_keys = list(self.NEUROPIL_ANCHORS.keys())

        for i in range(neuron_count):
            r_key = region_keys[i % len(region_keys)]
            anchor = self.NEUROPIL_ANCHORS[r_key]
            
            # Anchor etrafında Gauss dağılımı
            dx = rng.gauss(0, anchor["radius"] * 0.45)
            dy = rng.gauss(0, anchor["radius"] * 0.45)
            dz = rng.gauss(0, anchor["radius"] * 0.45)

            nt_choice = "Acetylcholine" if rng.random() < 0.70 else ("GABA" if rng.random() < 0.88 else "Octopamine")

            neurons.append({
                "id": i,
                "label": f"Neu_{r_key[:4]}_{i}",
                "neuropil": r_key,
                "transmitter": nt_choice,
                "position": {"x": round(anchor["x"] + dx, 2), "y": round(anchor["y"] + dy, 2), "z": round(anchor["z"] + dz, 2)},
                "basal_voltage_mv": round(-65.0 + rng.uniform(-3, 3), 1),
                "synaptic_targets": [rng.randint(0, neuron_count - 1) for _ in range(rng.randint(2, 6))]
            })

        # 3. Tümör Hücresi Ajanları (Mantar Cisimciği Kök Hücre Nişi)
        tumor_agents = []
        mb_anchor = self.NEUROPIL_ANCHORS["MushroomBody_Left"]
        for t_id in range(tumor_cells):
            tdx = rng.gauss(0, 14.0)
            tdy = rng.gauss(0, 14.0)
            tdz = rng.gauss(0, 14.0)
            is_stem = (t_id < 4)

            tumor_agents.append({
                "id": t_id,
                "state": "CancerStemCell" if is_stem else "Proliferating",
                "position": {"x": round(mb_anchor["x"] + tdx, 2), "y": round(mb_anchor["y"] + tdy, 2), "z": round(mb_anchor["z"] + tdz, 2)},
                "vitality": 1.0,
                "chemo_resistance": 0.75 if is_stem else 0.20,
                "cd47_shield_level": 0.88,
                "lactate_output": 0.08
            })

        # 4. Hemosit (İmmün Makrofaj) Ajanları
        hemocyte_agents = []
        for h_id in range(hemocytes):
            hdx = rng.gauss(0, 45.0)
            hdy = rng.gauss(0, 45.0)
            hdz = rng.gauss(0, 45.0)
            hemocyte_agents.append({
                "id": h_id,
                "position": {"x": round(mb_anchor["x"] + hdx, 2), "y": round(mb_anchor["y"] + hdy, 2), "z": round(mb_anchor["z"] + hdz, 2)},
                "state": "Patrolling",
                "phagocytic_capacity": 5.0,
                "speed": round(rng.uniform(2.0, 3.8), 2)
            })

        return {
            "format": "DrosophilaTwin_UnityScene_v1.0",
            "metadata": {
                "author": "Muhammed Emre Albayrak",
                "engine": "Unity 3D / C# .NET",
                "scale_units": "Micrometers (1 Unity Unit = 10 um)",
                "total_neuropils": len(neuropil_nodes),
                "total_neurons": len(neurons),
                "total_tumor_cells": len(tumor_agents),
                "total_hemocytes": len(hemocyte_agents)
            },
            "neuropils": neuropil_nodes,
            "neurons": neurons,
            "tumor_microenvironment": {
                "primary_site": "MushroomBody_Left",
                "cells": tumor_agents
            },
            "immune_system": {
                "hemocytes": hemocyte_agents
            }
        }

    def get_csharp_scripts(self) -> Dict[str, str]:
        """
        C# scriptlerinin içeriğini sözlük olarak döndürür (Web arayüzü kod görüntüleyici için).
        """
        scripts = {}
        if os.path.exists(self.csharp_dir):
            for fname in os.listdir(self.csharp_dir):
                if fname.endswith(".cs"):
                    fpath = os.path.join(self.csharp_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        scripts[fname] = f.read()
        return scripts

    def create_unity_package_zip(self) -> bytes:
        """
        Unity projesine doğrudan sürüklenebilecek tüm C# scriptleri,
        UPM manifestini ve sahne JSON dosyasını içeren bir ZIP arşivi üretir.
        """
        mem_file = io.BytesIO()
        scene_data = self.generate_scene_json()

        with zipfile.ZipFile(mem_file, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Sahne JSON'u ekle
            zf.writestr("DrosophilaTwin/ConnectomeSceneExport.json", json.dumps(scene_data, indent=2, ensure_ascii=False))

            # 2. C# Scriptlerini ekle
            scripts = self.get_csharp_scripts()
            for fname, content in scripts.items():
                zf.writestr(f"DrosophilaTwin/Scripts/{fname}", content)

            # 3. UPM Manifest ve README ekle
            pkg_json = os.path.join(self.package_dir, "package.json")
            if os.path.exists(pkg_json):
                with open(pkg_json, "r", encoding="utf-8") as f:
                    zf.writestr("DrosophilaTwin/package.json", f.read())

            readme_md = os.path.join(self.package_dir, "README.md")
            if os.path.exists(readme_md):
                with open(readme_md, "r", encoding="utf-8") as f:
                    zf.writestr("DrosophilaTwin/README.md", f.read())

        mem_file.seek(0)
        return mem_file.getvalue()


# Global Singleton
unity_exporter = UnityConnectomeExporter()
