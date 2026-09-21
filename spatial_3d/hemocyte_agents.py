"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: spatial_3d/hemocyte_agents.py
======================================================
Yazar: Hücresel İmmünoloji & Biyofizik Simülasyon Ekibi
Açıklama:
    3D Savunma Hücresi Ajanları (Hemositler):
      1. Plazmatositler (Makrofaj analogları): Kemotaksi ile tümör odağına göç,
         sitotoksisite ve fagositoz.
      2. Lamellositler (Büyük kapsülleyiciler): Agresif tümör nodülünü kuşatma,
         bölünmeyi bloke etme ve melanizasyon lizisi.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from spatial_3d.cancer_microenvironment import CancerCell3D, CancerState


class HemocyteSubtype(Enum):
    PLASMATOCYTE = "plasmatocyte"
    LAMELLOCYTE = "lamellocyte"


@dataclass
class HemocyteAgent3D:
    """
    3D Savunma Hücresi Ajanı.
    """
    id: int
    subtype: HemocyteSubtype
    position: np.ndarray             # [x, y, z] koordinatı (um)
    radius_um: float = 7.0
    cytotoxic_energy: float = 100.0  # Sitotoksik rezerv [0.0 - 100.0]
    exhaustion_index: float = 0.0    # Tükenmişlik skoru [0.0 - 1.0]
    kills_count: int = 0
    chemotaxis_drive: float = 0.88   # Kemotaktik gradyan takip hassasiyeti

    def step_patrol_and_attack(
        self,
        dt: float,
        cancer_cells: List[CancerCell3D],
        domain_bounds: np.ndarray,
        fuel_efficiency: float = 1.0,
        anti_cd47_active: bool = False,
        potency_multiplier: float = 1.0
    ) -> Optional[int]:
        """
        Kemotaksi ile en yakın canlı kanser hücresine yönelir ve temas halinde saldırır.
        Tükenmişlik, hedef klonunun immün-kaçış özellikleri, Anti-CD47 kontrol noktası
        blokajını ve sinerjik kokteyl potens çarpanını simüle eder.
        
        Returns:
            Etkisiz hale getirilen (öldürülen) kanser hücresi ID'si veya None.
        """
        if self.exhaustion_index >= 1.0 or self.cytotoxic_energy <= 4.0:
            return None

        # Hedef olabilecek canlı kanser hücrelerini filtrele
        viable_targets = [c for c in cancer_cells if c.state not in (CancerState.APOPTOTIC, CancerState.LYSED)]
        if not viable_targets:
            drift = (np.random.rand(3) - 0.5) * 0.35 * dt
            self.position = np.clip(self.position + drift, 5.0, domain_bounds - 5.0)
            return None

        # En yakın hedefi bul
        target_positions = np.array([c.position for c in viable_targets])
        deltas = target_positions - self.position
        dists = np.linalg.norm(deltas, axis=1)
        nearest_idx = int(np.argmin(dists))
        min_dist = dists[nearest_idx]
        target_cell = viable_targets[nearest_idx]

        # Temas / Etkileşim Yarıçapı
        contact_threshold = self.radius_um + target_cell.radius_um + 2.0

        if min_dist <= contact_threshold:
            # Hedefin Direnç ve İmmün-Kaçış Özellikleri
            evasion_mod = 1.0
            if getattr(target_cell, "clone_type", "") == "immune_evasive":
                if anti_cd47_active:
                    # Anti-CD47 monoklonal kalkanı "Don't eat me" sinyalini tamamen siler
                    evasion_mod = 1.0
                else:
                    # Draper/NimC1 reseptör kaçışı: %70 hasar sönümleme
                    evasion_mod = 0.30
            
            # Matriks ve glikokaliks direnç kalkanı
            target_res = getattr(target_cell, "resistance_score", 0.0)
            resistance_shield = max(0.25, 1.0 - 0.55 * target_res)

            # Tükenmişlik etkisi: Yorgun hemositler daha az hasar verir
            exhaustion_pen = max(0.15, 1.0 - 0.85 * self.exhaustion_index)
            cd47_boost = 2.2 if anti_cd47_active else 1.0
            potency = max(0.5, float(potency_multiplier))

            # Saldırı Mekaniği
            if self.subtype == HemocyteSubtype.LAMELLOCYTE:
                # Lamellosit Kapsülasyonu
                target_cell.state = CancerState.ENCAPSULATED
                strike = 18.0 * exhaustion_pen * evasion_mod * resistance_shield * cd47_boost * potency * dt
                target_cell.health -= strike
                self.cytotoxic_energy -= 8.0 * dt
                self.exhaustion_index = min(1.0, self.exhaustion_index + 0.08 * dt)
            else:
                # Plazmatosit Fagositoz / Sitotoksisite
                strike = 22.0 * exhaustion_pen * evasion_mod * resistance_shield * cd47_boost * potency * dt
                target_cell.health -= strike
                self.cytotoxic_energy -= 10.0 * dt
                self.exhaustion_index = min(1.0, self.exhaustion_index + 0.10 * dt)

            if target_cell.health <= 0.0:
                target_cell.state = CancerState.LYSED
                self.kills_count += 1
                return target_cell.id

            return None

        # Kemotaktik Yönelme
        direction_unit = deltas[nearest_idx] / (min_dist + 1e-6)
        noise = (np.random.rand(3) - 0.5) * 0.3
        
        speed_um_s = (0.35 if self.subtype == HemocyteSubtype.PLASMATOCYTE else 0.22) * max(0.2, fuel_efficiency)
        motion = (self.chemotaxis_drive * direction_unit + (1.0 - self.chemotaxis_drive) * noise)
        norm = np.linalg.norm(motion)
        if norm > 0:
            motion = (motion / norm) * speed_um_s * dt

        self.position = np.clip(self.position + motion, 5.0, domain_bounds - 5.0)
        return None
