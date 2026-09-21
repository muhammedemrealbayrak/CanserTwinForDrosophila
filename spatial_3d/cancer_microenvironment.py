"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: spatial_3d/cancer_microenvironment.py
======================================================
Yazar: 3D Mekansal Onkoloji & Reaksiyon-Difüzyon Ekibi
Açıklama:
    3D Tümör Mikroçevresi (TME) ve Kanser Hücre Modeli.
    p53 apoptoz direnci ve KRAS (Ras85D) hiper-proliferasyon mutasyonlarını,
    kaşeksi (PIF/Upd3) proteoliz salgısını ve 3D ilaç difüzyon ızgarasını yönetir.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class CancerState(Enum):
    PROLIFERATING = "proliferating"
    ENCAPSULATED = "encapsulated"  # Lamellositler tarafından sarılmış/hapsedilmiş
    APOPTOTIC = "apoptotic"
    LYSED = "lysed"


class CancerCloneType(Enum):
    SENSITIVE = "sensitive"
    RESISTANT_MEK = "resistant_mek"
    RESISTANT_EFFLUX = "resistant_efflux"
    IMMUNE_EVASIVE = "immune_evasive"


@dataclass
class CancerCell3D:
    """
    3D Agresif Malign Kanser Ajanı (Heterojen Klonal Popülasyon).
    """
    id: int
    position: np.ndarray             # [x, y, z] koordinatları (um)
    radius_um: float = 9.5
    state: CancerState = CancerState.PROLIFERATING
    health: float = 100.0            # Canlılık skoru [0.0 - 100.0]
    clone_type: str = "sensitive"    # "sensitive", "resistant_mek", "resistant_efflux", "immune_evasive"
    resistance_score: float = 0.05   # [0.0 - 1.0] Direnç düzeyi
    p53_mutated: bool = True         # Apoptoz direnci
    kras_mutated: bool = True        # Ras85D hiperaktif mitoz
    division_timer_s: float = 0.0
    division_threshold_s: float = 90.0 # Dinamik bölünme döngü süresi (s)
    cachectic_shed_rate: float = 0.003 # Kas/fat body eriten PIF/Upd3 toksini salınımı (saniyede birikim)
    _visual_reported: bool = False

    def step(
        self,
        dt: float,
        local_drug_conc: float,
        mek_inhibited: bool = False,
        dna_damaged: bool = False
    ) -> Tuple[bool, float]:
        """
        Kanser hücresini dt süresince günceller.
        
        Args:
            dt: Zaman adımı (s)
            local_drug_conc: Hücre konumundaki yerel ilaç konsantrasyonu (uM)
            mek_inhibited: MEK inhibitörü (Trametinib) varlığı (MEK bypass direncini kırar)
            dna_damaged: DNA çapraz bağlayıcı (Cisplatin) varlığı
            
        Returns:
            (ready_to_divide: bool, cachectic_toxin_shed: float)
        """
        if self.state in (CancerState.APOPTOTIC, CancerState.LYSED):
            return False, 0.0

        # 1. Klonal Direnç ve Hücre İçi İlaç Konsantrasyonu
        conc = local_drug_conc
        if self.clone_type == "resistant_efflux":
            # ABC Çoklu İlaç Taşıyıcıları (Mdr49/Mdr65) ilacı dışarı pompalar
            conc *= 0.20

        eff_res = self.resistance_score
        if self.clone_type == "resistant_mek":
            if mek_inhibited:
                # Sinerjik kokteyl MEK bypass'ını etkisiz kılar
                eff_res = min(eff_res, 0.15)
            else:
                eff_res = max(eff_res, 0.82)
        elif self.clone_type == "resistant_efflux":
            eff_res = max(eff_res, 0.75)
        elif self.clone_type == "immune_evasive":
            eff_res = max(eff_res, 0.35)

        if dna_damaged:
            # Sisplatin DNA adduct oluşturarak direnci hafifletir
            eff_res *= 0.40

        # Sitotoksik hasar (Direnç ne kadar yüksekse hasar o kadar düşer)
        damage_multiplier = max(0.04, 1.0 - eff_res)
        p53_mod = 0.70 if self.p53_mutated else 1.0
        drug_damage = conc * 0.045 * damage_multiplier * p53_mod * dt
        self.health -= drug_damage

        # 2. Sub-letal ilaç baskısı altında kazanılmış direnç (Darwinian adaptation)
        if conc > 0.05 and self.health > 15.0:
            self.resistance_score = min(0.95, self.resistance_score + 0.0015 * dt)

        if self.health <= 0.0:
            self.state = CancerState.APOPTOTIC
            return False, 0.0

        # 3. Kaşeksi toksini (PIF / Upd3 / Eiger)
        kras_mult = 1.4 if self.kras_mutated else 1.0
        toxin = self.cachectic_shed_rate * kras_mult * dt

        # Lamellositler tarafından hapsedildiyse mitoz bloke edilir
        if self.state == CancerState.ENCAPSULATED:
            return False, toxin * 0.4

        # 4. Dinamik Mitotik İlerleme:
        # İlaç dozu yüksek ve hücre hassassa mitoz durur (G1/S blokajı).
        # Ancak ilaç yetersizse veya hücre dirençliyse tümör agresif şekilde bölünür!
        drug_arrest = max(0.0, 1.0 - (conc * (1.0 - eff_res) / 1.2))
        mitotic_speed = (1.4 if self.kras_mutated else 1.0) * drug_arrest
        self.division_timer_s += dt * mitotic_speed

        if self.division_timer_s >= self.division_threshold_s and self.health > 40.0:
            self.division_timer_s = 0.0
            return True, toxin

        return False, toxin


class SpatialMicroenvironment3D:
    """
    3D Doku Hacmi ve Reaksiyon-Difüzyon Alanı.
    """

    def __init__(self, domain_size_um: float = 500.0, grid_resolution: int = 16):
        self.domain_size = domain_size_um
        self.grid_res = grid_resolution
        self.dx = domain_size_um / grid_resolution
        self.bounds = np.array([domain_size_um, domain_size_um, domain_size_um])

        # 3D İlaç Konsantrasyon Matrisi (uM)
        self.drug_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)
        self.chemokine_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)

        # Merkezi vasküler giriş noktası
        self.vessel_idx = (grid_resolution // 2, grid_resolution // 2, grid_resolution // 2)

    def inject_drug(self, dose_rate: float, dt: float, logP: float):
        """
        İlacı vasküler porttan dokuya enjekte eder.
        logP (lipofilisite) membran geçişini hızlandırır.
        """
        penetration_boost = 1.0 + 0.12 * logP
        inflow = dose_rate * penetration_boost * 2.5 * dt
        cx, cy, cz = self.vessel_idx
        self.drug_field[cx, cy, cz] += inflow

    def diffuse_fields(self, dt: float, diffusion_coeff: float = 75.0, clearance_rate: float = 0.001):
        """
        3D Fickian Laplacian diferansiyel denklemini çözer:
        dC/dt = D * Laplacian(C) - k_clear * C
        """
        substeps = 2
        sub_dt = dt / substeps
        r = (diffusion_coeff * sub_dt) / (self.dx ** 2)

        for _ in range(substeps):
            laplacian = (
                np.roll(self.drug_field, 1, axis=0) + np.roll(self.drug_field, -1, axis=0) +
                np.roll(self.drug_field, 1, axis=1) + np.roll(self.drug_field, -1, axis=1) +
                np.roll(self.drug_field, 1, axis=2) + np.roll(self.drug_field, -1, axis=2) -
                6.0 * self.drug_field
            )
            self.drug_field += r * laplacian - (clearance_rate * sub_dt * self.drug_field)
            np.clip(self.drug_field, 0.0, None, out=self.drug_field)

    def sample_drug_at(self, pos_um: np.ndarray) -> float:
        """Sürekli 3D uzaydaki bir koordinattaki yerel ilaç miktarını okur."""
        idx = np.clip((pos_um / self.dx).astype(int), 0, self.grid_res - 1)
        return float(self.drug_field[idx[0], idx[1], idx[2]])
