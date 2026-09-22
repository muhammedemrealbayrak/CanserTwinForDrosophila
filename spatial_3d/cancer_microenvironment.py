"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: spatial_3d/cancer_microenvironment.py
======================================================
Yazar: 3D Mekansal Onkoloji & Reaksiyon-Difüzyon Ekibi
Açıklama:
    3D Tümör Mikroçevresi (TME) ve Heterojen Klonal Kanser Modeli.
    - Hill-Langmuir sigmoidal farmakodinamik sitotoksisite
    - 4 Ayrı Biyolojik Direnç Ekseni (Duyarlı, MEK/SHP2 Bypass Direnci, ABC Eflüks, CD47 İmmün Kaçış)
    - Ras85D (KRAS) hiperaktif mitozu (hızlı bölünme) ve p53 apoptoz direnci
    - Warburg etkisi kaynaklı dinamik 3D Laktat Asidoz Alanı (TME immün felci)
    - Kaşeksi faktörleri (Upd3/Dawdle/Eiger) salınımı
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
    RESISTANT_MEK = "resistant_mek"        # SHP2/Corkscrew veya RTK bypass aktivasyonu
    RESISTANT_EFFLUX = "resistant_efflux"  # ABC Çoklu İlaç Taşıyıcıları (Mdr49/Mdr65)
    IMMUNE_EVASIVE = "immune_evasive"      # CD47 / Draper "Don't eat me" kalkanı


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
    p53_mutated: bool = True         # Apoptoz direnci (p53 mutasyonu)
    kras_mutated: bool = True        # Ras85D hiperaktif mitoz
    division_timer_s: float = 0.0
    division_threshold_s: float = 48.0 # Agresif Ras85D bölünme döngü süresi (s)
    cachectic_shed_rate: float = 0.005 # Upd3 / PIF / Dawdle kaşeksi sitokin salınımı
    lactate_shed_rate: float = 0.08   # Warburg glikoliz laktat salınımı (mM/s)
    _visual_reported: bool = False

    def step(
        self,
        dt: float,
        local_drug_conc: float,
        target_ec50: float = 0.15,
        mek_inhibited: bool = False,
        shp2_inhibited: bool = False,
        kras_inhibited: bool = False,
        dna_damaged: bool = False,
        metabolic_starved: bool = False
    ) -> Tuple[bool, float, float]:
        """
        Kanser hücresini dt süresince günceller.
        
        Args:
            dt: Zaman adımı (s)
            local_drug_conc: Hücre konumundaki yerel ilaç konsantrasyonu (uM)
            target_ec50: İlacın hedef reseptör/kinaz afinitesine dayalı EC50 değeri (uM)
            mek_inhibited: MEK inhibitörü (Trametinib) varlığı
            shp2_inhibited: SHP2 allosterik inhibitörü (RMC-4550) veya dikey blokaj
            kras_inhibited: Doğrudan KRAS G12D inhibitörü (MRTX1133)
            dna_damaged: DNA hasar verici (Sisplatin / PARP+ATR sentetik ölümcüllük)
            metabolic_starved: HK2 / GLS1 metabolik açlık (2-DG / Telaglenastat)
            
        Returns:
            (ready_to_divide: bool, cachectic_toxin_shed: float, lactate_shed: float)
        """
        if self.state in (CancerState.APOPTOTIC, CancerState.LYSED):
            return False, 0.0, 0.0

        # 1. Hücre İçi İlaç Konsantrasyonu ve ABC Eflüks Pompaları
        conc = max(0.0, local_drug_conc)
        if self.clone_type == "resistant_efflux":
            if metabolic_starved:
                # ATP krizinde ABC pompaları (Mdr49/Mdr65) kilitlenir, ilaç dışarı atılamaz!
                conc *= 0.85
            else:
                # Aktif eflüks: İlacın %80'ini hücre dışına pompalar
                conc *= 0.20

        # 2. Biyolojik Klonal Direnç ve Yolak Bypass Eksenleri
        eff_res = self.resistance_score
        
        if self.clone_type == "resistant_mek":
            # MEK Monoterapisi İkilemi:
            # Hücrede SHP2/Corkscrew ve RTK adaptif geri besleme aktivasyonu mevcuttur.
            # Yalnızca MEK inhibe edilirse (Trametinib tek başına), bu klon MEK'i bypass ederek
            # yüksek dirençle (%86) hayatta kalır ve çoğalmaya devam eder (Relaps)!
            # Ancak SHP2 veya dikey KRAS inhibitörü eşzamanlı verilirse bypass kırılır!
            if shp2_inhibited or kras_inhibited:
                # Sinerjik dikey kilit direnci ezer
                eff_res = min(eff_res, 0.18)
            elif mek_inhibited:
                # MEK tek başına uygulandığında dirençli klon hayatta kalır
                eff_res = max(eff_res, 0.86)
            else:
                eff_res = max(eff_res, 0.75)

        elif self.clone_type == "resistant_efflux":
            if metabolic_starved or dna_damaged:
                eff_res = min(eff_res, 0.25)
            else:
                eff_res = max(eff_res, 0.80)

        elif self.clone_type == "immune_evasive":
            eff_res = max(eff_res, 0.30)

        elif self.clone_type == "sensitive":
            eff_res = min(eff_res, 0.15)

        # Sisplatin veya PARP+ATR sentetik ölümcüllük DNA adductları oluşturur
        if dna_damaged:
            eff_res *= 0.45

        # 3. Hill-Langmuir Farmakodinamik Sitotoksisite Fonksiyonu
        # E(C) = E_max * (C^h / (EC50^h + C^h))
        ec50 = max(0.08, float(target_ec50))  # uM
        hill_h = 1.6
        emax_damage_per_s = 6.2  # Maksimum sitotoksik hasar hızı (HP/s)
        
        if conc > 0.001:
            pd_kill = emax_damage_per_s * (conc ** hill_h) / ((ec50 ** hill_h) + (conc ** hill_h))
        else:
            pd_kill = 0.0

        # Direnç ve p53 modülasyonu
        damage_multiplier = max(0.03, 1.0 - eff_res)
        p53_mod = 0.72 if self.p53_mutated else 1.0
        
        step_damage = pd_kill * damage_multiplier * p53_mod * dt
        self.health -= step_damage

        # 4. Darwinian Adaptif Direnç Kazanımı:
        # Sub-letal tekil ilaç baskısı altında sensitive klonlar direnç kazanır
        if conc > 0.08 and self.health > 20.0 and self.clone_type == "sensitive":
            self.resistance_score = min(0.92, self.resistance_score + 0.0012 * dt)

        # Hücre ölümü kontrolü
        if self.health <= 0.0:
            self.state = CancerState.APOPTOTIC
            return False, 0.0, 0.0

        # 5. Kaşeksi Toksini ve TME Warburg Laktat Salınımı
        kras_mult = 1.35 if self.kras_mutated else 1.0
        cachectic_toxin = self.cachectic_shed_rate * kras_mult * dt
        lactate_shed = self.lactate_shed_rate * dt

        # Lamellositler tarafından kuşatılıp hapsedildiyse (kapsülasyon) mitoz ve salgılar baskılanır
        if self.state == CancerState.ENCAPSULATED:
            return False, cachectic_toxin * 0.35, lactate_shed * 0.30

        # 6. Agresif Ras85D Mitotik Bölünme Dinamiği:
        # Etkili ilaç baskısı varsa (yüksek konsantrasyon + duyarlı hücre) G1/S fazı kilitlenir.
        # Ancak dirençli klonlar (örn. MEK monoterapisinde resistant_mek) bölünmeye hızla devam eder!
        effective_inhibition = min(1.0, (conc * (1.0 - eff_res)) / 0.18)
        
        # Dikey KRAS/SHP2 kilidi veya metabolik açlık mitozu sıfırlar
        if (shp2_inhibited and mek_inhibited) or (kras_inhibited and shp2_inhibited) or metabolic_starved:
            effective_inhibition = 1.0

        drug_arrest = max(0.0, 1.0 - effective_inhibition)
        mitotic_speed = (1.45 if self.kras_mutated else 1.0) * drug_arrest
        self.division_timer_s += dt * mitotic_speed

        if self.division_timer_s >= self.division_threshold_s and self.health > 18.0:
            self.division_timer_s = 0.0
            self.health = max(25.0, self.health * 0.72)
            return True, cachectic_toxin, lactate_shed

        return False, cachectic_toxin, lactate_shed


class SpatialMicroenvironment3D:
    """
    3D Doku Hacmi, Reaksiyon-Difüzyon Ağı ve Warburg Laktat Asidoz Alanı.
    """

    def __init__(self, domain_size_um: float = 500.0, grid_resolution: int = 16):
        self.domain_size = domain_size_um
        self.grid_res = grid_resolution
        self.dx = domain_size_um / grid_resolution
        self.bounds = np.array([domain_size_um, domain_size_um, domain_size_um])

        # 3D İlaç Konsantrasyon Matrisi (uM)
        self.drug_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)
        # 3D Warburg Laktat Asidoz Matrisi (mM)
        self.lactate_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)
        # 3D Kemokin / DAMP Gradyan Matrisi
        self.chemokine_field = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)

        # Merkezi vasküler giriş noktası
        self.vessel_idx = (grid_resolution // 2, grid_resolution // 2, grid_resolution // 2)

    def inject_drug(self, dose_rate: float, dt: float, logP: float):
        """
        İlacı vasküler porttan dokuya enjekte eder.
        logP (lipofilisite) membran geçişini hızlandırır.
        """
        penetration_boost = 1.0 + 0.12 * logP
        inflow = dose_rate * penetration_boost * 1.8 * dt
        cx, cy, cz = self.vessel_idx
        self.drug_field[cx, cy, cz] += inflow

    def deposit_lactate(self, pos_um: np.ndarray, amount: float):
        """Kanser hücresinin bulunduğu 3D ızgara hücresine laktat biriktirir."""
        idx = np.clip((pos_um / self.dx).astype(int), 0, self.grid_res - 1)
        self.lactate_field[idx[0], idx[1], idx[2]] += float(amount)

    def clear_lactate(self, clearance_boost: float = 1.0, dt: float = 1.0):
        """
        MCT1 inhibitörü (AZD3965) veya sinerjik kokteyl varlığında
        laktat asidozunu hızla nötralize eder.
        """
        base_clearance = 0.02 * clearance_boost * dt
        self.lactate_field = np.maximum(0.0, self.lactate_field * (1.0 - base_clearance))

    def diffuse_fields(
        self,
        dt: float,
        diffusion_coeff: float = 75.0,
        clearance_rate: float = 0.003,
        mct1_inhibited: bool = False
    ):
        """
        3D Fickian Laplacian diferansiyel denklemini çözer:
        dC/dt = D * Laplacian(C) - k_clear * C
        Hem ilaç hem de laktat asidoz alanlarını reaksiyon-difüzyonla yayar.
        """
        substeps = 2
        sub_dt = dt / substeps
        r_drug = (diffusion_coeff * sub_dt) / (self.dx ** 2)
        r_lactate = ((diffusion_coeff * 0.7) * sub_dt) / (self.dx ** 2)

        lactate_kclear = 0.04 if mct1_inhibited else 0.002

        for _ in range(substeps):
            # İlaç difüzyonu
            lap_drug = (
                np.roll(self.drug_field, 1, axis=0) + np.roll(self.drug_field, -1, axis=0) +
                np.roll(self.drug_field, 1, axis=1) + np.roll(self.drug_field, -1, axis=1) +
                np.roll(self.drug_field, 1, axis=2) + np.roll(self.drug_field, -1, axis=2) -
                6.0 * self.drug_field
            )
            self.drug_field += r_drug * lap_drug - (clearance_rate * sub_dt * self.drug_field)
            np.clip(self.drug_field, 0.0, None, out=self.drug_field)

            # Laktat difüzyonu ve klerensi
            lap_lactate = (
                np.roll(self.lactate_field, 1, axis=0) + np.roll(self.lactate_field, -1, axis=0) +
                np.roll(self.lactate_field, 1, axis=1) + np.roll(self.lactate_field, -1, axis=1) +
                np.roll(self.lactate_field, 1, axis=2) + np.roll(self.lactate_field, -1, axis=2) -
                6.0 * self.lactate_field
            )
            self.lactate_field += r_lactate * lap_lactate - (lactate_kclear * sub_dt * self.lactate_field)
            np.clip(self.lactate_field, 0.0, None, out=self.lactate_field)

    def sample_drug_at(self, pos_um: np.ndarray) -> float:
        """Sürekli 3D uzaydaki bir koordinattaki yerel ilaç miktarını okur (uM)."""
        idx = np.clip((pos_um / self.dx).astype(int), 0, self.grid_res - 1)
        return float(self.drug_field[idx[0], idx[1], idx[2]])

    def sample_lactate_at(self, pos_um: np.ndarray) -> float:
        """Sürekli 3D uzaydaki yerel laktat konsantrasyonunu okur (mM)."""
        idx = np.clip((pos_um / self.dx).astype(int), 0, self.grid_res - 1)
        return float(self.lactate_field[idx[0], idx[1], idx[2]])
