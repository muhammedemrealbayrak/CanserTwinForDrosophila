"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: biology/lymph_gland.py
======================================================
Yazar: İmmüno-Hematoloji & Hücre Diferansiyasyon Ekibi
Açıklama:
    Drosophila Lenf Bezi (Lymph Gland - Kemik İliği Analoğu).
    Nöral eferent uyarımı ve 4 aşamalı metabolik yakıt akısını alarak
    kök hücrelerden (pro-hemosit) olgun savunma hücreleri (Plazmatosit ve
    Lamellosit) sentezleyen ve 3D dolaşıma salan organ motoru.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from biology.fuel_metabolism import HierarchicalFuelMetabolismEngine, BiochemicalFuelPool


@dataclass
class HemocyteProductionMetrics:
    """Üretim ve farklılaşma metrikleri."""
    active_progenitors: float = 30.0
    plasmatocytes_produced: int = 0
    lamellocytes_produced: int = 0
    total_cells_egressed: int = 0
    hematopoietic_stem_reserve: float = 100.0  # % Kök hücre / progenitör rezervi (Kemik iliği / Lenf bezi)
    myelosuppression_events: int = 0


class LymphGlandOrgan:
    """
    Lenf Bezi (Hematopoietik Organ) Simülatörü.
    """

    def __init__(self, fuel_engine: Optional[HierarchicalFuelMetabolismEngine] = None):
        self.fuel_engine = fuel_engine if fuel_engine is not None else HierarchicalFuelMetabolismEngine()
        self.metrics = HemocyteProductionMetrics()
        
        # Diferansiyasyon boru hattı (Maturation pipeline)
        self.pipeline_stage1_primed: float = 25.0
        self.pipeline_stage2_transcribing: float = 18.0
        self.pipeline_stage3_replicating: float = 12.0
        self.pipeline_stage4_membrane_assembly: float = 8.0

    def step_hematopoiesis(
        self,
        dt_seconds: float,
        elapsed_seconds: float,
        neural_efferent_drive: float,
        drug_immune_boost: float = 0.0,
        myelosuppressive_toxicity: float = 0.0,
        sirtuin_shield: float = 0.0,
        cholinergic_boost: float = 0.0
    ) -> Tuple[int, int]:
        """
        Saniye saniye lenf bezi hücresel üretim adımı.
        
        Args:
            dt_seconds: Zaman adımı (s).
            elapsed_seconds: Toplam simülasyon zamanı (s).
            neural_efferent_drive: Beyinden gelen kolinerjik/nöral eferent sinyal [0.0 - 2.5].
            drug_immune_boost: İlacın bağışıklık organını doğrudan tetikleme katsayısı.
            myelosuppressive_toxicity: Kemoterapötik veya aşırı kinaz sitotoksisitesinin kök hücre hasarı.
            sirtuin_shield: Resveratrol / Sir2 aktivasyonu ile sağlanan sitoproteksiyon kalkanı [0.0 - 1.0].
            cholinergic_boost: Tracey (2002) / Aonuma (2020) nAChR nöro-immün rejenerasyon sürüşü.
            
        Returns:
            (new_plasmatocytes: int, new_lamellocytes: int)
        """
        # 1. Metabolik yakıt tüketimini işlet
        demand_flux = neural_efferent_drive + drug_immune_boost
        metabolism_snap = self.fuel_engine.step_metabolism(
            dt_seconds=dt_seconds,
            elapsed_seconds=elapsed_seconds,
            progenitor_demand_flux=demand_flux
        )
        fuel_eff = metabolism_snap["stage_efficiency"]

        # 2. Miyelosüpresyon & Kök Hücre Rezervi Dinamikleri (Bishayee 2009 & Tracey 2002)
        eff_myelo = myelosuppressive_toxicity * max(0.0, 1.0 - sirtuin_shield)
        if eff_myelo > 0.02:
            # Sitotoksik hasar kök hücre rezervini tüketir ve aşama 2-3 hücrelerini lizise uğratır
            stem_damage = eff_myelo * 2.8 * dt_seconds
            self.metrics.hematopoietic_stem_reserve = max(0.0, self.metrics.hematopoietic_stem_reserve - stem_damage)
            self.metrics.myelosuppression_events += 1
            self.pipeline_stage2_transcribing = max(0.0, self.pipeline_stage2_transcribing - eff_myelo * 3.5 * dt_seconds)
            self.pipeline_stage3_replicating = max(0.0, self.pipeline_stage3_replicating - eff_myelo * 4.5 * dt_seconds)
        else:
            # Homeostatik Kök Hücre Rejenerasyonu (Kolinerjik sürüş ile 2.5x hızlanır)
            recovery_flux = (0.22 + 0.48 * cholinergic_boost) * fuel_eff * dt_seconds
            self.metrics.hematopoietic_stem_reserve = min(100.0, self.metrics.hematopoietic_stem_reserve + recovery_flux)

        # Rezerv faktörü: Rezerv <%25 ise hematopoez durma noktasına gelir (Ağır Miyelosüpresyon / Aplasia)
        reserve_ratio = float(np.clip(self.metrics.hematopoietic_stem_reserve / 100.0, 0.0, 1.0))
        stem_multiplier = reserve_ratio ** 1.6

        # 3. Aşama 1: Nöral uyarım ile pro-hemosit bölünme başlangıcı
        induction_flux = (0.4 + neural_efferent_drive * 0.9 + drug_immune_boost * 0.6) * fuel_eff * stem_multiplier
        self.pipeline_stage1_primed += induction_flux * 0.20 * dt_seconds

        # 4. Aşama 1 -> 2 Aktarımı (Glutamin/BCAA transkripsiyon faktör hazırlığı)
        adv_1_to_2 = min(self.pipeline_stage1_primed * 0.04 * dt_seconds, self.pipeline_stage1_primed)
        self.pipeline_stage1_primed -= adv_1_to_2
        self.pipeline_stage2_transcribing += adv_1_to_2 * fuel_eff

        # 5. Aşama 2 -> 3 Aktarımı (DNA Sentezi ve Amplifikasyon)
        adv_2_to_3 = min(self.pipeline_stage2_transcribing * 0.035 * dt_seconds, self.pipeline_stage2_transcribing)
        self.pipeline_stage2_transcribing -= adv_2_to_3
        # Relish & Stat92E faktörleri klonal çoğalmayı 1.8x katlar
        tf_factor = 0.5 * (metabolism_snap["relish_nfkb"] + metabolism_snap["stat92e"])
        clonal_expansion = 1.7 * (0.5 + 0.5 * tf_factor)
        self.pipeline_stage3_replicating += adv_2_to_3 * clonal_expansion * fuel_eff

        # 6. Aşama 3 -> 4 Aktarımı (Lipit Dış Zar İnşası)
        adv_3_to_4 = min(self.pipeline_stage3_replicating * 0.03 * dt_seconds, self.pipeline_stage3_replicating)
        self.pipeline_stage3_replicating -= adv_3_to_4
        self.pipeline_stage4_membrane_assembly += adv_3_to_4 * fuel_eff

        # 7. Olgun Savunma Hücrelerinin Sistemik Egress'i (3D Dokuya Çıkış)
        egress_probability_rate = 0.05 * fuel_eff * stem_multiplier
        potential_cells = self.pipeline_stage4_membrane_assembly * egress_probability_rate * dt_seconds

        # Poisson / Stokastik tam sayı dönüşümü
        int_cells = int(np.floor(potential_cells))
        if np.random.rand() < (potential_cells - int_cells):
            int_cells += 1

        self.pipeline_stage4_membrane_assembly = max(0.0, self.pipeline_stage4_membrane_assembly - int_cells)

        # Lamellosit vs Plazmatosit oranı (%85 Plazmatosit, %15 Tümör Kapsülleyici Lamellosit)
        new_lamello = int(int_cells * 0.20)
        new_plasma = int_cells - new_lamello

        self.metrics.plasmatocytes_produced += new_plasma
        self.metrics.lamellocytes_produced += new_lamello
        self.metrics.total_cells_egressed += int_cells

        return new_plasma, new_lamello
