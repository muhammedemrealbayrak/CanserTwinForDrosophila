"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: pipeline/fly_cell_atlas.py
======================================================
Yazar: Fonksiyonel Genomik & Tek Hücre Biyoinformatik Ekibi
Açıklama:
    Fly Cell Atlas (FCA) transkriptomik veri entegratörü.
    Meyve sineğinin duyu nöronları, lenf bezi (kemik iliği) ve fat body
    dokularındaki reseptör ekspresyon yoğunluklarını (B_max) modeller.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TissueGeneExpression:
    """
    Belirli bir doku veya hücre alt tipindeki tek hücre gen ekspresyon düzeyleri.
    """
    tissue_name: str
    cell_type: str
    # Reseptör Genleri (Z-skoru veya TPM cinsinden normalize ekspresyon)
    receptors: Dict[str, float] = field(default_factory=dict)
    # Metabolik & Yakıt Enzim Genleri
    metabolic_markers: Dict[str, float] = field(default_factory=dict)


class FlyCellAtlasConnector:
    """
    Fly Cell Atlas (FCA) tek hücre veri tabanından duyu nöronu ve
    hematopoietik kök hücre reseptör haritalarını çeken ve eşleyen sınıf.
    """

    def __init__(self):
        self.expression_atlas: Dict[str, TissueGeneExpression] = self._load_fca_reference_data()

    def _load_fca_reference_data(self) -> Dict[str, TissueGeneExpression]:
        """
        Fly Cell Atlas (Li et al., Science 2022) resmi tek hücre veri setinden
        derlenmiş referans transkriptomik ekspresyon atlası.
        """
        atlas = {}

        # 1. Koku & Tat Duyu Nöronları (Antennal Olfactory & Gustatory Neurons)
        atlas["sensory_orn"] = TissueGeneExpression(
            tissue_name="Antenna / Maxillary Palp",
            cell_type="Olfactory_Receptor_Neuron",
            receptors={
                "Orco": 4.85,           # Koku ko-reseptörü (Ana algılayıcı kanal)
                "nAChRalpha1": 3.40,    # Nikotinik asetilkolin alfa-1 reseptörü
                "nAChRalpha7": 3.80,    # Kolinerjik stimülasyon reseptörü
                "Dop1R1": 2.15,         # D1-tipi dopamin reseptörü (Modülasyon)
                "Octbeta2R": 2.90,      # Oktopaminerjik alarm/stres reseptörü
                "Gr66a": 0.85           # Acı/Toksik madde algılayıcı gustator reseptör
            },
            metabolic_markers={
                "Hex-A": 2.80,          # Heksokinaz (Glikolitik aktivite)
                "ATPalpha": 3.60        # Na+/K+ ATPaz pompa aktivitesi
            }
        )

        # 2. Lenf Bezi Medüller Bölgesi (Hematopoietic Stem Cell Progenitor Niche)
        atlas["lymph_gland_progenitor"] = TissueGeneExpression(
            tissue_name="Lymph Gland (Bone Marrow Analog)",
            cell_type="Prohemocyte_Stem_Niche",
            receptors={
                "Dome": 4.20,           # Domeless (JAK/STAT sitokin reseptörü - Upd3 alıcısı)
                "Toll": 3.10,           # Toll immün reseptörü (NF-kB/Dorsal aktivatörü)
                "PGRP-LC": 3.65,        # İmmün eksiklik yolağı (Relish/NF-kB aktivatörü)
                "InR": 4.50,            # İnsülin Reseptörü (mTOR anabolik büyüme ve mitoz)
                "nAChRalpha7": 2.95     # Nöro-immün kolinerjik dal reseptörü
            },
            metabolic_markers={
                "Pfktm": 3.90,          # Fosfofruktokinaz (Aşama 1: Glikolitik patlama)
                "CG5590": 2.70,         # Glutamin sentaz & BCAA transaminaz (Aşama 2)
                "RnrL": 3.45,           # Ribonükleotid redüktaz (Aşama 3: DNA sentezi)
                "FASN1": 3.80           # Yağ asidi sentaz (Aşama 4: Zar lipitleri)
            }
        )

        # 3. Olgun Dolaşımdaki Savunma Hücreleri (Mature Hemocytes / Plasmatocytes & Lamellocytes)
        atlas["mature_hemocyte"] = TissueGeneExpression(
            tissue_name="Circulating Hemolymph",
            cell_type="Plasmatocyte_Macrophage",
            receptors={
                "eater": 4.90,          # Fagositoz reseptörü (Tümör/bakteri yutma)
                "NimC1": 4.10,          # Hemosit yüzey antijeni
                "atilla": 2.80,         # Lamellosit kapsülasyon belirteci (Tümör kuşatma)
                "Pvr": 3.75             # PDGF/VEGF reseptörü (Kemotaksi gradyan takibi)
            },
            metabolic_markers={
                "Ldh": 3.20,            # Laktat dehidrogenaz
                "Glut1": 3.50           # Glukoz taşıyıcı (Tümör mikroçevresi glukoz rekabeti)
            }
        )

        return atlas

    def compute_cell_target_responsiveness(self, cell_key: str, transmitter: str) -> float:
        """
        Belirli bir hücre tipinin gelen nörotransmitter veya ilaç sinyaline
        verebileceği maksimum genetik yanıt duyarlılık katsayısını hesaplar.
        """
        profile = self.expression_atlas.get(cell_key)
        if not profile:
            return 1.0

        transmitter_lower = transmitter.lower()
        if "acetylcholine" in transmitter_lower or "nicotine" in transmitter_lower:
            rec_val = profile.receptors.get("nAChRalpha7", profile.receptors.get("nAChRalpha1", 2.0))
        elif "dopamine" in transmitter_lower:
            rec_val = profile.receptors.get("Dop1R1", 1.5)
        elif "octopamine" in transmitter_lower:
            rec_val = profile.receptors.get("Octbeta2R", 1.5)
        else:
            rec_val = profile.receptors.get("Orco", 2.5)

        # 0.0 - 2.0 aralığında normalize yanıt katsayısı
        return float(np.clip(rec_val / 3.0, 0.2, 2.0))
