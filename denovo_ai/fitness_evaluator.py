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
from typing import Dict, List, Any
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
        host_alive: bool = True
    ) -> CandidateEvaluationResult:
        """
        Çok amaçlı ödül fonksiyonunu hesaplar:
        R = w_speed * (60.0 / time_to_trigger) + w_eff * (hemocytes / 50) + w_clear * (tumor_clear / 100) - w_tox * (tox * 10)
        Eğer konakçı aşırı toksisiteden öldüyse (host_alive=False veya tox>=45%), skor sıfırlanır.
        """
        total_tox = max(profile.qsar_toxicity_risk, toxicity_score)

        if not host_alive or total_tox >= 0.45:
            # Konakçıyı öldüren bileşiklere fatalite cezası
            final_fitness = round(float(max(0.0, (1.0 - total_tox) * 15.0)), 1)
            return CandidateEvaluationResult(
                candidate_name=profile.name,
                smiles=profile.smiles,
                time_to_trigger_s=round(time_to_trigger_s, 1),
                total_hemocytes_produced=hemocytes_produced,
                tumor_clearance_pct=0.0,
                systemic_toxicity_pct=round(total_tox * 100.0, 1),
                multiobjective_fitness_score=final_fitness
            )

        # Hız Skoru (Maksimum 40 Puan): 15 saniyeden ne kadar hızlı tetiklerse o kadar yüksek
        speed_ratio = max(0.0, (15.0 - min(15.0, time_to_trigger_s)) / 14.0)
        speed_score = speed_ratio * 40.0

        # Savunma Verimi Skoru (Maksimum 35 Puan): Hemosit üretimi ve tümör küçülmesi
        hemocyte_ratio = min(1.0, hemocytes_produced / 70.0)
        efficacy_score = (hemocyte_ratio * 25.0) + (tumor_clearance_pct * 0.10)

        # Güvenlik & Toksisite Skoru (Maksimum 25 Puan): Düşük toksisite yüksek puan
        safety_score = max(0.0, (1.0 - total_tox) * 25.0)

        # Toplam Birleşik Skor (0 - 100)
        final_fitness = round(float(np.clip(speed_score + efficacy_score + safety_score, 0.0, 100.0)), 1)

        return CandidateEvaluationResult(
            candidate_name=profile.name,
            smiles=profile.smiles,
            time_to_trigger_s=round(time_to_trigger_s, 1),
            total_hemocytes_produced=hemocytes_produced,
            tumor_clearance_pct=round(tumor_clearance_pct, 1),
            systemic_toxicity_pct=round(total_tox * 100.0, 1),
            multiobjective_fitness_score=final_fitness
        )
