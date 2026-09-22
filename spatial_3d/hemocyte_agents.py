"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: spatial_3d/hemocyte_agents.py
======================================================
Yazar: Hücresel İmmünoloji & Biyofizik Simülasyon Ekibi
Açıklama:
    3D Savunma Hücresi Ajanları (Hemositler):
      1. Plazmatositler (Makrofaj analogları): Kemotaksi ile tümör odağına göç,
         fagositoz döngüsü ve refrakter süreler.
      2. Lamellositler (Büyük kapsülleyiciler): Agresif tümör nodülünü kuşatma,
         bölünmeyi bloke etme ve melanizasyon lizisi.
    Biyofiziksel Gerçekçilik:
      - Fagositoz refrakter (bekleme) periyodu (3-5 saniye)
      - TME Laktat Asidozu felci (hız düşüşü ve hızlanan tükenmişlik)
      - CD47 / Draper "Don't eat me" kontrol noktası engellemesi
      - Sonlu sitotoksik rezerv ve tükenmişlik apoptozu
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
    refractory_timer_s: float = 0.0  # Fagositoz / saldırı sonrası bekleme süresi
    age_s: float = 0.0               # Hemosit operasyonel yaşı
    max_lifespan_s: float = 65.0     # Doğal hücre yaşam döngüsü / apoptoz
    is_apoptotic: bool = False       # Tükenmişlik veya yaşlanma sonucu hücresel ölüm

    def step_patrol_and_attack(
        self,
        dt: float,
        cancer_cells: List[CancerCell3D],
        domain_bounds: np.ndarray,
        local_lactate_mM: float = 0.0,
        fuel_efficiency: float = 1.0,
        anti_cd47_active: bool = False,
        potency_multiplier: float = 1.0
    ) -> Optional[int]:
        """
        Kemotaksi ile en yakın canlı kanser hücresine yönelir ve temas halinde saldırır.
        Refrakter periyot, TME laktat asidozu, CD47 kontrol noktası blokajı ve
        tükenmişlik dinamiklerini tam gerçekçilikle simüle eder.
        
        Returns:
            Etkisiz hale getirilen (öldürülen) kanser hücresi ID'si veya None.
        """
        self.age_s += dt
        if self.is_apoptotic or self.age_s >= self.max_lifespan_s or self.exhaustion_index >= 1.0 or self.cytotoxic_energy <= 3.0:
            self.is_apoptotic = True
            return None

        # Refrakter bekleme süresi işletimi (Fagositoz / lizis döngü gecikmesi)
        if self.refractory_timer_s > 0.0:
            self.refractory_timer_s = max(0.0, self.refractory_timer_s - dt)
            # Refrakter sürede hücre hafifçe sürüklenir ama saldıramaz
            drift = (np.random.rand(3) - 0.5) * 0.15 * dt
            self.position = np.clip(self.position + drift, 5.0, domain_bounds - 5.0)
            return None

        # TME Laktat Asidozunun Hemosit Üzerindeki Felç Edici Etkisi:
        # Yüksek laktat ortamı hemosit hızını %70'e kadar keser ve tükenmeyi 2.5x hızlandırır.
        acidosis_speed_factor = max(0.30, 1.0 - 0.20 * min(3.5, local_lactate_mM))
        exhaustion_multiplier = 1.0 + min(2.5, 0.40 * local_lactate_mM)

        # Hedef olabilecek canlı kanser hücrelerini filtrele
        viable_targets = [c for c in cancer_cells if c.state not in (CancerState.APOPTOTIC, CancerState.LYSED)]
        if not viable_targets:
            drift = (np.random.rand(3) - 0.5) * 0.25 * dt
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
            # 1. CD47 / Draper "Don't Eat Me" İmmün Kontrol Noktası Kaçışı
            evasion_mod = 1.0
            if getattr(target_cell, "clone_type", "") == "immune_evasive":
                if anti_cd47_active:
                    # Anti-CD47 / Evorpacept mimetiği "Don't eat me" sinyalini tamamen yıkar
                    evasion_mod = 1.0
                else:
                    # Kanser hücresi Draper/NimC1 fagositoz reseptörünü kilitler: %85 hasar sönümleme!
                    evasion_mod = 0.15

            # 2. Matriks ve glikokaliks direnç kalkanı
            target_res = getattr(target_cell, "resistance_score", 0.0)
            resistance_shield = max(0.20, 1.0 - 0.60 * target_res)

            # 3. Tükenmişlik ve Potens Çarpanı
            exhaustion_pen = max(0.12, 1.0 - 0.88 * self.exhaustion_index)
            cd47_boost = 1.35 if anti_cd47_active else 1.0
            potency = max(0.5, float(potency_multiplier))

            # 4. Biyolojik Olarak Kalibre Edilmiş Saldırı Hasarı (Dengeli oran)
            if self.subtype == HemocyteSubtype.LAMELLOCYTE:
                # Lamellosit Kapsülasyonu: Bölünmeyi dondurur, melanizasyon hasarı verir
                target_cell.state = CancerState.ENCAPSULATED
                base_strike = 3.6
                strike = base_strike * exhaustion_pen * evasion_mod * resistance_shield * cd47_boost * potency * dt
                target_cell.health -= strike
                self.cytotoxic_energy -= 4.5 * dt
                self.exhaustion_index = min(1.0, self.exhaustion_index + (0.045 * exhaustion_multiplier * dt))
                # Kapsülasyon sonrası 2.5 saniye refrakter bekleme
                self.refractory_timer_s = 2.5
            else:
                # Plazmatosit Fagositozu / Sitotoksisite
                base_strike = 4.5
                strike = base_strike * exhaustion_pen * evasion_mod * resistance_shield * cd47_boost * potency * dt
                target_cell.health -= strike
                self.cytotoxic_energy -= 5.5 * dt
                self.exhaustion_index = min(1.0, self.exhaustion_index + (0.055 * exhaustion_multiplier * dt))
                # Fagositer lizis döngüsü için 3.5 saniye refrakter bekleme
                self.refractory_timer_s = 3.5

            if target_cell.health <= 0.0:
                target_cell.state = CancerState.LYSED
                self.kills_count += 1
                # Bir hücreyi tamamen yuttuktan sonra sindirim için ek refrakter süre
                self.refractory_timer_s = 4.5
                return target_cell.id

            return None

        # Kemotaktik Yönelme (TME laktat asidozu ile hız modüle edilir)
        direction_unit = deltas[nearest_idx] / (min_dist + 1e-6)
        noise = (np.random.rand(3) - 0.5) * 0.25
        
        base_speed = 0.30 if self.subtype == HemocyteSubtype.PLASMATOCYTE else 0.20
        speed_um_s = base_speed * max(0.2, fuel_efficiency) * acidosis_speed_factor
        
        motion = (self.chemotaxis_drive * direction_unit + (1.0 - self.chemotaxis_drive) * noise)
        norm = np.linalg.norm(motion)
        if norm > 0:
            motion = (motion / norm) * speed_um_s * dt

        self.position = np.clip(self.position + motion, 5.0, domain_bounds - 5.0)
        return None
