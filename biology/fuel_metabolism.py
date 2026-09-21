"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: biology/fuel_metabolism.py
======================================================
Yazar: Hücresel Biyoenerjetik & Sistem Metabolizması Ekibi
Açıklama:
    Kemik İliği / Lenf Bezi 4 Aşamalı Moleküler Yakıt Hiyerarşisi.
    Hücre bölünmesi ve savunma hücresi inşasında şu sıralamayı saniye saniye tüketir:
      - Aşama 1 (0-15 Dk): Glukoz ve ATP (Bölünme için anlık enerji patlaması).
      - Aşama 2 (15-30 Dk): Glutamin ve BCAA Aminoasitleri (Relish/NF-kB ve Stat92E aktivasyonu).
      - Aşama 3 (30-60 Dk): Nükleotidler, Çinko ve Demir (DNA replikasyonu ve klonal çoğalma).
      - Aşama 4 (60-90 Dk): Lipitler ve Yağ Asitleri (3D dış zar inşası ve egress).
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class BiochemicalFuelPool:
    """
    Hücresel yakıt rezervleri ve konsantrasyonları (mM veya uM).
    """
    # Aşama 1: Anlık Enerji Patlaması (0 - 15 dk)
    atp_mM: float = 5.0
    glucose_mM: float = 6.5
    
    # Aşama 2: Aminoasit & Transkripsiyon İndüksiyonu (15 - 30 dk)
    glutamine_mM: float = 0.85
    bcaa_mM: float = 0.55             # Dallı zincirli aminoasitler (Leu, Ile, Val)
    
    # Aşama 3: Genetik Sentez & Klonal Çoğalma (30 - 60 dk)
    nucleotides_mM: float = 0.45      # dNTPs havuzu
    zinc_uM: float = 19.0             # Zn-Finger nükleaz kofaktörü
    iron_uM: float = 24.0             # Ribonükleotid redüktaz Fe kofaktörü
    
    # Aşama 4: 3D Dış Zar Lipitleri (60 - 90 dk)
    lipids_fatty_acids_mM: float = 2.8 # Fosfolipit & seramid yapı taşları

    # Sistemik İmmün Toksisite & Sitokin Düzeyi
    pro_inflammatory_cytokines: float = 1.0


class HierarchicalFuelMetabolismEngine:
    """
    4 Aşamalı Metabolik Yakıt Tüketim Motoru.
    """

    def __init__(self, pool: Optional[BiochemicalFuelPool] = None):
        self.pool = pool if pool is not None else BiochemicalFuelPool()
        
        # Transkripsiyon faktörü aktivasyon seviyeleri [0.0 - 1.0]
        self.relish_nfkb_activation: float = 0.05
        self.stat92e_activation: float = 0.05

        # Kümülatif harcanan substratlar
        self.cumulative_spent: Dict[str, float] = {
            "atp": 0.0, "glucose": 0.0,
            "glutamine": 0.0, "bcaa": 0.0,
            "nucleotides": 0.0, "zinc": 0.0, "iron": 0.0,
            "lipids": 0.0
        }

    def get_stage_info(self, elapsed_seconds: float) -> Tuple[int, str]:
        """
        Simülasyon zamanına göre aktif metabolik aşamayı belirler (90 dk / 5400 s döngü).
        """
        clock = elapsed_seconds % 5400.0
        if clock < 900.0:
            return 1, "Aşama 1 (0-15 Dk): Glukoz ve ATP (Anlık Enerji Patlaması)"
        elif clock < 1800.0:
            return 2, "Aşama 2 (15-30 Dk): Glutamin ve BCAA (Relish/Stat92E Aktivasyonu)"
        elif clock < 3600.0:
            return 3, "Aşama 3 (30-60 Dk): Nükleotidler, Çinko & Demir (DNA Replikasyonu)"
        else:
            return 4, "Aşama 4 (60-90 Dk): Lipitler ve Yağ Asitleri (3D Dış Zar İnşası)"

    def step_metabolism(
        self,
        dt_seconds: float,
        elapsed_seconds: float,
        progenitor_demand_flux: float
    ) -> Dict[str, float]:
        """
        Bir saniyelik metabolik adımı işletir.
        
        Returns:
            Dict: Aşama verimliliği, tf_aktivasyonu ve kalan yakıt oranları.
        """
        stage_idx, stage_name = self.get_stage_info(elapsed_seconds)
        burn_multiplier = 1.0 + progenitor_demand_flux * 0.5

        efficiency = 1.0

        # -------------------------------------------------------------
        # AŞAMA 1: Glukoz ve ATP (0 - 15 dk)
        # -------------------------------------------------------------
        if stage_idx == 1:
            km_atp, km_glc = 0.8, 1.0
            avail_atp = self.pool.atp_mM / (self.pool.atp_mM + km_atp)
            avail_glc = self.pool.glucose_mM / (self.pool.glucose_mM + km_glc)
            efficiency = float(avail_atp * avail_glc)

            spend_atp = min(self.pool.atp_mM, 0.0035 * burn_multiplier * dt_seconds)
            spend_glc = min(self.pool.glucose_mM, 0.0045 * burn_multiplier * dt_seconds)
            self.pool.atp_mM -= spend_atp
            self.pool.glucose_mM -= spend_glc
            self.cumulative_spent["atp"] += spend_atp
            self.cumulative_spent["glucose"] += spend_glc

        # -------------------------------------------------------------
        # AŞAMA 2: Glutamin ve BCAA Aminoasitleri (15 - 30 dk)
        # -------------------------------------------------------------
        elif stage_idx == 2:
            km_gln, km_bcaa = 0.20, 0.15
            avail_gln = self.pool.glutamine_mM / (self.pool.glutamine_mM + km_gln)
            avail_bcaa = self.pool.bcaa_mM / (self.pool.bcaa_mM + km_bcaa)
            efficiency = float(avail_gln * avail_bcaa)

            # Relish/NF-kB ve Stat92E indüksiyonu
            self.relish_nfkb_activation = float(np.clip(
                self.relish_nfkb_activation + (avail_gln * 0.04 - 0.005) * dt_seconds, 0.0, 1.0
            ))
            self.stat92e_activation = float(np.clip(
                self.stat92e_activation + (avail_bcaa * 0.04 - 0.005) * dt_seconds, 0.0, 1.0
            ))

            spend_gln = min(self.pool.glutamine_mM, 0.0022 * burn_multiplier * dt_seconds)
            spend_bcaa = min(self.pool.bcaa_mM, 0.0018 * burn_multiplier * dt_seconds)
            self.pool.glutamine_mM -= spend_gln
            self.pool.bcaa_mM -= spend_bcaa
            self.cumulative_spent["glutamine"] += spend_gln
            self.cumulative_spent["bcaa"] += spend_bcaa

        # -------------------------------------------------------------
        # AŞAMA 3: Nükleotidler, Çinko ve Demir (30 - 60 dk)
        # -------------------------------------------------------------
        elif stage_idx == 3:
            km_nt, km_zn, km_fe = 0.10, 4.0, 5.0
            avail_nt = self.pool.nucleotides_mM / (self.pool.nucleotides_mM + km_nt)
            avail_zn = self.pool.zinc_uM / (self.pool.zinc_uM + km_zn)
            avail_fe = self.pool.iron_uM / (self.pool.iron_uM + km_fe)
            tf_boost = 0.5 * (self.relish_nfkb_activation + self.stat92e_activation)
            efficiency = float(avail_nt * avail_zn * avail_fe * (0.4 + 0.6 * tf_boost))

            spend_nt = min(self.pool.nucleotides_mM, 0.0020 * burn_multiplier * dt_seconds)
            spend_zn = min(self.pool.zinc_uM, 0.028 * burn_multiplier * dt_seconds)
            spend_fe = min(self.pool.iron_uM, 0.032 * burn_multiplier * dt_seconds)
            self.pool.nucleotides_mM -= spend_nt
            self.pool.zinc_uM -= spend_zn
            self.pool.iron_uM -= spend_fe
            self.cumulative_spent["nucleotides"] += spend_nt
            self.cumulative_spent["zinc"] += spend_zn
            self.cumulative_spent["iron"] += spend_fe

        # -------------------------------------------------------------
        # AŞAMA 4: Lipitler ve Yağ Asitleri (60 - 90 dk)
        # -------------------------------------------------------------
        elif stage_idx == 4:
            km_lip = 0.40
            avail_lip = self.pool.lipids_fatty_acids_mM / (self.pool.lipids_fatty_acids_mM + km_lip)
            efficiency = float(avail_lip)

            spend_lip = min(self.pool.lipids_fatty_acids_mM, 0.0032 * burn_multiplier * dt_seconds)
            self.pool.lipids_fatty_acids_mM -= spend_lip
            self.cumulative_spent["lipids"] += spend_lip

        return {
            "stage_idx": stage_idx,
            "stage_name": stage_name,
            "stage_efficiency": efficiency,
            "relish_nfkb": self.relish_nfkb_activation,
            "stat92e": self.stat92e_activation,
            "pool_atp": self.pool.atp_mM,
            "pool_glucose": self.pool.glucose_mM,
            "pool_bcaa": self.pool.bcaa_mM,
            "pool_lipids": self.pool.lipids_fatty_acids_mM
        }
