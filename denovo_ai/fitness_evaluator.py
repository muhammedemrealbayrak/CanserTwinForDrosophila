"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: denovo_ai/fitness_evaluator.py
======================================================
Yazar: Takviyeli Öğrenme & Optimizasyon Ekibi
Açıklama:
    De Novo İlaç Adayları için Çok Amaçlı (Multi-Objective) Ödül Fonksiyonu.
    Hedefler:
      1. Maksimum Nöral Tetikleme Hızı (Zamana karşı yarış)
      2. Maksimum Hemosit (Savunma Hücresi) Üretim Verimi
      3. Minimum Sağlıklı Hücre Toksisitesi / Yan Etki
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
from pipeline.pubchem_connector import MolecularProfile


@dataclass
class CandidateEvaluationResult:
    """Tek bir moleküler adayın simülasyon performans çıktısı."""
    candidate_name: str
    smiles: str
    time_to_trigger_s: float         # Bağışıklığı uyarma başlangıç süresi (s)
    total_hemocytes_produced: int     # Üretilen savunma hücresi
    tumor_clearance_pct: float        # Tümör küçülme başarısı (%)
    systemic_toxicity_pct: float      # Sağlıklı doku hasar skoru (%)
    multiobjective_fitness_score: float # Birleşik ödül skoru


class MolecularFitnessEvaluator:
    """
    Simülatör telemetrisini alarak molekülleri puanlayan ve en efektif
    sentetik formülü seçen değerlendirici sınıf.
    """

    def __init__(self, weight_speed: float = 1.2, weight_efficacy: float = 1.0, weight_toxicity: float = 2.5):
        self.w_speed = weight_speed
        self.w_eff = weight_efficacy
        self.w_tox = weight_toxicity

    def compute_fitness(
        self,
        profile: MolecularProfile,
        time_to_trigger_s: float,
        hemocytes_produced: int,
        tumor_clearance_pct: float,
        toxicity_score: float,
        host_alive: bool = True,
        clinical_outcome: Optional[str] = None
    ) -> CandidateEvaluationResult:
        """
        Çok amaçlı onkolojik ödül fonksiyonunu hesaplar:
          1. Tümör Eradikasyonu ve Klonal Temizleme (Maks 40 Puan)
          2. Nöro-İmmün Tetikleme Hızı ve Hemosit Üretimi (Maks 30 Puan)
          3. Güvenlik, Düşük Toksisite ve Konakçı Canlılığı (Maks 30 Puan)
        Nüks (Relaps), Tümör İstila veya Ölüm durumlarında orantılı cezalar uygulanır.
        """
        total_tox = max(profile.qsar_toxicity_risk, toxicity_score)

        if not host_alive or total_tox >= 0.40 or clinical_outcome == "HOST_LETHALITY_OVERDOSE":
            # Konakçıyı öldüren bileşiklere fatalite cezası
            final_fitness = round(float(max(0.0, (1.0 - min(1.0, total_tox)) * 12.0)), 1)
            return CandidateEvaluationResult(
                candidate_name=profile.name,
                smiles=profile.smiles,
                time_to_trigger_s=round(time_to_trigger_s, 1),
                total_hemocytes_produced=hemocytes_produced,
                tumor_clearance_pct=0.0,
                systemic_toxicity_pct=round(total_tox * 100.0, 1),
                multiobjective_fitness_score=final_fitness
            )

        # 1. Hız ve İmmün Mobilizasyon Skoru (Maksimum 30 Puan)
        speed_ratio = max(0.0, (15.0 - min(15.0, time_to_trigger_s)) / 14.0)
        hemocyte_ratio = min(1.0, hemocytes_produced / 70.0)
        speed_score = (speed_ratio * 18.0) + (hemocyte_ratio * 12.0)

        # 2. Tümör Küçülme ve Klonal Temizleme Skoru (Maksimum 40 Puan)
        clearance_ratio = float(np.clip(tumor_clearance_pct / 100.0, 0.0, 1.0))
        clearance_score = clearance_ratio * 40.0

        # 3. Güvenlik & Toksisite Skoru (Maksimum 30 Puan): <%8 toksisite tam puan alır
        safety_score = max(0.0, (1.0 - (total_tox / 0.40)) * 30.0)

        raw_fitness = speed_score + clearance_score + safety_score

        # 4. Klinik Sonuç Cezaları
        if clinical_outcome == "TUMOR_RELAPSE_RESISTANT":
            raw_fitness = max(15.0, raw_fitness - 28.0)  # Nüks cezası
        elif clinical_outcome == "TUMOR_PROGRESSION_ESCAPE":
            raw_fitness = max(10.0, raw_fitness - 35.0)  # İstila cezası
        elif clinical_outcome == "COMPLETE_REMISSION":
            raw_fitness = min(100.0, raw_fitness + 5.0)  # Tam remisyon bonusu

        final_fitness = round(float(np.clip(raw_fitness, 0.0, 100.0)), 1)

        return CandidateEvaluationResult(
            candidate_name=profile.name,
            smiles=profile.smiles,
            time_to_trigger_s=round(time_to_trigger_s, 1),
            total_hemocytes_produced=hemocytes_produced,
            tumor_clearance_pct=round(tumor_clearance_pct, 1),
            systemic_toxicity_pct=round(total_tox * 100.0, 1),
            multiobjective_fitness_score=final_fitness
        )
