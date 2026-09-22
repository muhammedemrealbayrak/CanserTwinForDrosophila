"""
Drosophila In Silico Neuro-Immune Digital Twin & De Novo Drug Discovery Engine
Modül: platform_engine.py
=============================================================================
Yazar: Baş Sistem Biyoloğu & Yapay Zeka Baş Mimarı
Açıklama:
    Tüm katmanları (Kimya, Nöro-Konektom, Transkriptomik, 4-Aşamalı Yakıt,
    3D Ajan Simülasyonu ve De Novo Optimizasyon) tek bir çok ölçekli
    (multi-scale) zaman adımında birleştiren ana platform motoru.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Alt Katman İthalatları
from pipeline.pubchem_connector import PubChemConnector, MolecularProfile
from pipeline.fly_cell_atlas import FlyCellAtlasConnector
from pipeline.flywire_circuit import FlyWireCircuitSimulator
from biology.fuel_metabolism import HierarchicalFuelMetabolismEngine, BiochemicalFuelPool
from biology.lymph_gland import LymphGlandOrgan
from spatial_3d.cancer_microenvironment import CancerCell3D, SpatialMicroenvironment3D, CancerState
from spatial_3d.hemocyte_agents import HemocyteAgent3D, HemocyteSubtype
from denovo_ai.fitness_evaluator import MolecularFitnessEvaluator, CandidateEvaluationResult


class DrosophilaInSilicoPlatform:
    """
    Ana Dijital İkiz Orkestrasyon Motoru.
    """

    def __init__(
        self,
        active_compound_smiles_or_name: str = "DeNovo_HighSpeed_Agonist",
        initial_tumor_burden: int = 150,
        domain_size_um: float = 500.0,
        grid_resolution: int = 16
    ):
        # 1. Konnektörler ve Katman Başlatıcılar
        self.pubchem = PubChemConnector()
        self.fca = FlyCellAtlasConnector()
        self.evaluator = MolecularFitnessEvaluator()

        # 2. Aktif İlaç Molekülü
        self.active_drug: MolecularProfile = self.pubchem.parse_molecule(active_compound_smiles_or_name)
        self.drug_dose_uM: float = 2.5

        # 3. Nöral ve Biyolojik Motorlar
        self.neural_circuit = FlyWireCircuitSimulator(
            sensory_latency_ms=10.0,
            interneuron_latency_ms=25.0,
            central_latency_ms=150.0,
            resonant_frequency_hz=45.0
        )
        self.fuel_engine = HierarchicalFuelMetabolismEngine()
        self.lymph_gland = LymphGlandOrgan(self.fuel_engine)

        # 4. 3D Mekansal Alan
        self.domain_size = domain_size_um
        self.spatial_tme = SpatialMicroenvironment3D(domain_size_um, grid_resolution)

        # 5. Ajan Koleksiyonları
        self.cancer_cells: List[CancerCell3D] = []
        self.hemocyte_agents: List[HemocyteAgent3D] = []
        self.next_agent_id = 1

        # 6. Zaman ve Telemetri Kaydı
        self.elapsed_time_s: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self.time_to_first_trigger_s: Optional[float] = None
        self.initial_tumor_count = initial_tumor_burden
        self.lowest_tumor_count = initial_tumor_burden
        self.active_cocktail: Optional[Dict[str, Any]] = None
        self.newly_lysed_events: List[List[float]] = []

        # Konakçı Canlılığı ve Toksik Ölüm Eşiği (Biyolojik Gerçekçilik)
        self.host_alive: bool = True
        self.lethal_toxicity_threshold: float = 0.45  # %45 toksisite konakçı için ölümcüldür
        self.cumulative_cachectic_toxin: float = 0.0

        # Onkolojik Tedavi Modalitesi & Radyoterapi
        self.active_modality: str = "targeted_small_molecule"
        self.radiation_pulses_applied: int = 0

        # 7. Sistemi Başlat
        self._initialize_tissue(initial_tumor_burden)

    def _initialize_tissue(self, tumor_count: int):
        """3D Doku alanına heterojen klonlardan oluşan başlangıç tümör nodülü ve yerleşik hemositleri yerleştirir."""
        center = np.array([self.domain_size / 2.0] * 3)
        tumor_radius = 65.0  # um

        # Kanser Hücrelerini Heterojen Klon Dağılımıyla Merkeze Yerleştir
        # %68 Duyarlı (Sensitive), %16 MEK-Bypass Dirençli (SHP2/RTK), %10 ABC-Efflux, %6 CD47 İmmün-Kaçış
        for _ in range(tumor_count):
            offset = np.random.normal(0.0, tumor_radius / 2.5, size=3)
            pos = np.clip(center + offset, 15.0, self.domain_size - 15.0)

            roll = np.random.rand()
            if roll < 0.68:
                c_type = "sensitive"
                c_res = float(np.random.uniform(0.02, 0.10))
            elif roll < 0.84:
                c_type = "resistant_mek"
                c_res = float(np.random.uniform(0.78, 0.90))
            elif roll < 0.94:
                c_type = "resistant_efflux"
                c_res = float(np.random.uniform(0.72, 0.88))
            else:
                c_type = "immune_evasive"
                c_res = float(np.random.uniform(0.45, 0.65))

            c = CancerCell3D(
                id=self.next_agent_id,
                position=pos,
                radius_um=float(np.random.uniform(8.5, 11.0)),
                clone_type=c_type,
                resistance_score=c_res,
                p53_mutated=True,
                kras_mutated=True,
                division_threshold_s=float(np.random.uniform(42.0, 56.0))
            )
            self.cancer_cells.append(c)
            self.next_agent_id += 1

        # Başlangıç Devriye Hemositleri (Çeperde)
        for _ in range(int(tumor_count * 0.20)):
            pos = np.random.uniform(20.0, self.domain_size - 20.0, size=3)
            h = HemocyteAgent3D(
                id=self.next_agent_id,
                subtype=HemocyteSubtype.PLASMATOCYTE,
                position=pos
            )
            self.hemocyte_agents.append(h)
            self.next_agent_id += 1

    def reset(
        self,
        initial_tumor_burden: Optional[int] = None,
        molecule_name_or_smiles: Optional[str] = None,
        dose_uM: Optional[float] = None,
        modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tüm sistemi, 3D dokuyu, hücreleri ve fizyolojik göstergeleri temiz başlangıç durumuna sıfırlar."""
        if molecule_name_or_smiles:
            self.active_drug = self.pubchem.parse_molecule(molecule_name_or_smiles)
        self.drug_dose_uM = dose_uM if dose_uM is not None else 2.5
        if modality and modality in self.TREATMENT_MODALITIES:
            self.active_modality = modality

        burden = initial_tumor_burden if initial_tumor_burden is not None else self.initial_tumor_count
        self.initial_tumor_count = burden
        self.lowest_tumor_count = burden
        self.elapsed_time_s = 0.0
        self.history = []
        self.time_to_first_trigger_s = None
        self.newly_lysed_events = []
        self.radiation_pulses_applied = 0

        # Toksisite ve konakçı canlılığı sıfırlama
        self.cumulative_cachectic_toxin = 0.0
        self.host_alive = True

        # Biyolojik motorları sıfırla
        self.fuel_engine = HierarchicalFuelMetabolismEngine()
        self.lymph_gland = LymphGlandOrgan(self.fuel_engine)
        self.spatial_tme = SpatialMicroenvironment3D(self.domain_size, self.spatial_tme.grid_res)

        # Hücre listelerini sıfırla ve yeniden oluştur
        self.cancer_cells = []
        self.hemocyte_agents = []
        self.next_agent_id = 1
        self._initialize_tissue(burden)

        return self.get_current_snapshot()

    def get_current_snapshot(self) -> Dict[str, Any]:
        """Zamanı ilerletmeden mevcut platform telemetri durumunu üretir."""
        viable_cancer = [c for c in self.cancer_cells if c.state not in (CancerState.APOPTOTIC, CancerState.LYSED)]
        viable_cancer_count = len(viable_cancer)
        sensitive_count = sum(1 for c in viable_cancer if getattr(c, "clone_type", "sensitive") == "sensitive")
        resistant_count = viable_cancer_count - sensitive_count

        active_hemocyte_count = sum(1 for h in self.hemocyte_agents if h.exhaustion_index < 1.0 and h.cytotoxic_energy > 4.0)
        stage_idx, stage_desc = self.fuel_engine.get_stage_info(self.elapsed_time_s)

        dose_factor = (self.drug_dose_uM / 2.5) ** 0.8
        direct_drug_tox = self.active_drug.qsar_toxicity_risk * dose_factor
        if self.active_modality == "cytotoxic_chemotherapy":
            direct_drug_tox = max(direct_drug_tox, 0.32 * dose_factor)
        elif self.active_modality == "metabolic_starvation":
            direct_drug_tox = max(direct_drug_tox, 0.10 * dose_factor)

        cachexia_burden = min(0.30, (self.cumulative_cachectic_toxin / 300.0) * 0.20)
        raw_tox = direct_drug_tox + cachexia_burden
        if self.active_cocktail:
            tox_red = self.active_cocktail.get("toxicity_reduction_pct", 0.0) / 100.0
            raw_tox *= (1.0 - tox_red)

        systemic_tox = float(np.clip(raw_tox, 0.0, 1.0))

        if systemic_tox >= self.lethal_toxicity_threshold:
            self.host_alive = False
            host_vitality = 0.0
            clinical_outcome = "HOST_LETHALITY_OVERDOSE"
            clinical_status_text = "☠️ ORGANİZMA ÖLÜMÜ (AŞIRI TOKSİSİTE)"
        else:
            self.host_alive = True
            host_vitality = max(0.0, round((1.0 - (systemic_tox / self.lethal_toxicity_threshold)) * 100.0, 1))
            if viable_cancer_count == 0:
                clinical_outcome = "COMPLETE_REMISSION"
                clinical_status_text = "🟢 TAM REMİSYON"
            elif viable_cancer_count >= int(self.initial_tumor_count * 1.60):
                clinical_outcome = "TUMOR_PROGRESSION_ESCAPE"
                clinical_status_text = "🔴 TEDAVİ BAŞARISIZ: TÜMÖR İSTİLASI"
            elif self.lowest_tumor_count <= int(self.initial_tumor_count * 0.80) and viable_cancer_count >= int(self.lowest_tumor_count * 1.25) and resistant_count >= 0.35 * max(1, viable_cancer_count):
                clinical_outcome = "TUMOR_RELAPSE_RESISTANT"
                clinical_status_text = "🟠 DİRENÇLİ NÜKS / RELAPS"
            elif viable_cancer_count <= int(self.initial_tumor_count * 0.45):
                clinical_outcome = "PARTIAL_RESPONSE"
                clinical_status_text = "🟡 KISMİ YANIT"
            else:
                clinical_outcome = "STABLE_DISEASE"
                clinical_status_text = "⚪ DURAĞAN HASTALIK"

        return {
            "time_s": self.elapsed_time_s,
            "drug_name": self.active_cocktail["name"] if self.active_cocktail else self.active_drug.name,
            "smiles": self.active_drug.smiles,
            "receptor_occupancy": 0.0 if self.elapsed_time_s == 0.0 else 0.5,
            "lysed_bursts": [],
            "active_cocktail": self.active_cocktail,
            "efferent_hz": 0.0 if not self.host_alive or self.elapsed_time_s == 0.0 else 42.5,
            "marrow_drive": 0.0 if not self.host_alive or self.elapsed_time_s == 0.0 else 0.5,
            "kcg_membrane_mv": -65.0,
            "kcg_calcium_nm": 100.0,
            "snpf_release_pct": 0.0,
            "ach_quanta_nm": 5.0,
            "spikes_per_sec": 0,
            "voltage_trace": [-65.0],
            "calcium_trace": [100.0],
            "flywire_neuron_info": {
                "root_id": "720575940608530955",
                "cell_class": "Kenyon_Cell",
                "cell_sub_class": "KCg",
                "cell_type": "KCg-m",
                "dataset": "FAFB v783",
                "side": "left",
                "known_transmitters": "acetylcholine; sNPF"
            },
            "initial_tumor_count": self.initial_tumor_count,
            "cancer_cells": viable_cancer_count,
            "sensitive_cancer_cells": sensitive_count,
            "resistant_cancer_cells": resistant_count,
            "active_hemocytes": active_hemocyte_count,
            "total_egressed_hemocytes": self.lymph_gland.metrics.total_cells_egressed,
            "stage_idx": stage_idx,
            "stage_name": stage_desc,
            "pool_atp": self.fuel_engine.pool.atp_mM,
            "pool_glucose": self.fuel_engine.pool.glucose_mM,
            "pool_bcaa": self.fuel_engine.pool.bcaa_mM,
            "pool_lipids": self.fuel_engine.pool.lipids_fatty_acids_mM,
            "toxicity_pct": systemic_tox * 100.0,
            "host_alive": self.host_alive,
            "host_vitality_pct": host_vitality,
            "clinical_outcome": clinical_outcome,
            "clinical_status_text": clinical_status_text,
            "active_modality": self.active_modality,
            "modality_info": self.TREATMENT_MODALITIES.get(self.active_modality, {}),
            "radiation_pulses_applied": self.radiation_pulses_applied
        }

    COCKTAIL_REGIMENS: Dict[str, Any] = {
        "immuno_mek_synergy": {
            "id": "immuno_mek_synergy",
            "name": "Immuno-MEK Sinerjisi (F-NAc + Trametinib + Kurkumin)",
            "primary_smiles": "NC(=O)CN1CCC[C@H]1c2cncc(F)c2",
            "components": [
                {"name": "F-NAc (De Novo Agonist)", "dose": "0.85 µM", "dose_uM": 0.85, "target": "nAChR / KCg-m", "role": "nAChR / KCg-m", "mechanism": "185ms Refleksle Hemosit Üretimi", "dri_fold": 8.1},
                {"name": "Trametinib", "dose": "5.0 nM", "dose_uM": 0.005, "target": "MEK1/2 Kinaz", "role": "MEK1/2 Kinaz", "mechanism": "KRAS/MAPK Proliferasyon Blokajı", "dri_fold": 8.1},
                {"name": "Curcumin", "dose": "4.2 µM", "dose_uM": 4.2, "target": "NF-kB / STAT", "role": "NF-kB / STAT", "mechanism": "Kaşeksi Kalkanı & Doku Koruması", "dri_fold": 14.4}
            ],
            "synergy_index_ci": 0.34,
            "chou_talalay_ci": 0.34,
            "synergy_label": "Süper Sinerji (CI < 0.45)",
            "synergy_description": "Derin Moleküler Kooperasyon & Çoklu Yolak Kilitlenmesi",
            "bliss_excess_score": 0.038,
            "bliss_observed_kill": 0.985,
            "bliss_expected_kill": 0.947,
            "toxicity_reduction_pct": 88.0,
            "toxicity_shield_pct": 0.88,
            "potency_boost": 2.4,
            "target_potency_multiplier": 2.4,
            "dri_profile": {
                "F-NAc": {"dose_uM": 0.85, "dri_fold": 8.1, "sparing_pct": 87.6},
                "Trametinib": {"dose_uM": 0.005, "dri_fold": 8.1, "sparing_pct": 87.6},
                "Curcumin": {"dose_uM": 4.2, "dri_fold": 14.4, "sparing_pct": 93.1}
            },
            "crosstalk_interactions": [
                {"agent_a": "F-NAc", "agent_b": "Trametinib", "targets": "nAChR ⟷ MEK", "coupling_strength": 0.32},
                {"agent_a": "Trametinib", "agent_b": "Curcumin", "targets": "MEK ⟷ NF-kB", "coupling_strength": 0.22}
            ],
            "clinical_rationale": "Chou-Talalay CI = 0.34 (Süper Sinerji). Mantar cisimciğinden gelen hemosit patlaması ile hücre içi onkogenik MEK blokajı birleşir; kurkumin kaşeksiyi %88 oranında önler.",
            "description": "Mantar cisimciğinden gelen hemosit patlaması ile hücre içi onkogenik MEK blokajı birleşir; kurkumin kaşeksiyi %88 oranında önler."
        },
        "soft_drug_chemo_immune": {
            "id": "soft_drug_chemo_immune",
            "name": "Soft-Drug Kemo-İmmün (MCN + Sisplatin + Resveratrol)",
            "primary_smiles": "COC(=O)N1CCC[C@H]1c2cccnc2",
            "components": [
                {"name": "MCN (Karbamat Agonist)", "dose": "1.4 µM", "dose_uM": 1.4, "target": "nAChR", "role": "nAChR", "mechanism": "Esteraz-Klerensli Hızlı Egress", "dri_fold": 6.4},
                {"name": "Cisplatin", "dose": "0.35 µM", "dose_uM": 0.35, "target": "DNA Adducts", "role": "DNA Adducts", "mechanism": "Tümör Hücresi DNA Çapraz Bağlama", "dri_fold": 6.8},
                {"name": "Resveratrol", "dose": "7.5 µM", "dose_uM": 7.5, "target": "SIRT1", "role": "SIRT1", "mechanism": "Sağlıklı Hücre Apoptoz Direnci", "dri_fold": 9.5}
            ],
            "synergy_index_ci": 0.42,
            "chou_talalay_ci": 0.42,
            "synergy_label": "Süper Sinerji (CI < 0.45)",
            "synergy_description": "Metronomik Kemo-İmmün Koruma",
            "bliss_excess_score": 0.032,
            "bliss_observed_kill": 0.978,
            "bliss_expected_kill": 0.946,
            "toxicity_reduction_pct": 82.0,
            "toxicity_shield_pct": 0.82,
            "potency_boost": 2.1,
            "target_potency_multiplier": 2.1,
            "dri_profile": {
                "MCN": {"dose_uM": 1.4, "dri_fold": 6.4, "sparing_pct": 84.4},
                "Cisplatin": {"dose_uM": 0.35, "dri_fold": 6.8, "sparing_pct": 85.3},
                "Resveratrol": {"dose_uM": 7.5, "dri_fold": 9.5, "sparing_pct": 89.5}
            },
            "crosstalk_interactions": [
                {"agent_a": "MCN", "agent_b": "Cisplatin", "targets": "nAChR ⟷ DNA", "coupling_strength": 0.28},
                {"agent_a": "Cisplatin", "agent_b": "Resveratrol", "targets": "DNA ⟷ SIRT1", "coupling_strength": 0.26}
            ],
            "clinical_rationale": "Chou-Talalay CI = 0.42. Düşük doz kemoterapötik ile immün yanıt sinerjiye girer; Sisplatin dozu 6.8x azaltılarak toksisite <%5'te tutulur.",
            "description": "Düşük doz kemoterapötik ile immün yanıt sinerjiye girer; Sisplatin dozu 6.8x azaltılarak toksisite <%5'te tutulur."
        },
        "denovo_triple_shield": {
            "id": "denovo_triple_shield",
            "name": "De Novo Triple-Shield (AI Şampiyon + Vorinostat + EGCG)",
            "primary_smiles": "CC1=NC=C(C=C1)CCN(C)C(=O)CF",
            "components": [
                {"name": "De Novo Şampiyon", "dose": "0.85 µM", "dose_uM": 0.85, "target": "KCg-m Gamma Lobe", "role": "KCg-m Gamma Lobe", "mechanism": "1.0s Hızlı Refleks ve Eferent Sürüş", "dri_fold": 8.1},
                {"name": "Vorinostat", "dose": "1.2 µM", "dose_uM": 1.2, "target": "HDAC Sınıf I/II", "role": "HDAC Sınıf I/II", "mechanism": "Epigenetik Kanser Hücresi Farklılaşması", "dri_fold": 7.4},
                {"name": "EGCG", "dose": "5.5 µM", "dose_uM": 5.5, "target": "Antioksidan", "role": "Antioksidan", "mechanism": "Mitokondriyal Membran Stabilizasyonu", "dri_fold": 10.8}
            ],
            "synergy_index_ci": 0.38,
            "chou_talalay_ci": 0.38,
            "synergy_label": "Süper Sinerji (CI < 0.45)",
            "synergy_description": "Epigenetik Kırılma & Mitokondriyal Kalkan",
            "bliss_excess_score": 0.035,
            "bliss_observed_kill": 0.982,
            "bliss_expected_kill": 0.947,
            "toxicity_reduction_pct": 89.0,
            "toxicity_shield_pct": 0.89,
            "potency_boost": 2.3,
            "target_potency_multiplier": 2.3,
            "dri_profile": {
                "De Novo Şampiyon": {"dose_uM": 0.85, "dri_fold": 8.1, "sparing_pct": 87.6},
                "Vorinostat": {"dose_uM": 1.2, "dri_fold": 7.4, "sparing_pct": 86.5},
                "EGCG": {"dose_uM": 5.5, "dri_fold": 10.8, "sparing_pct": 90.7}
            },
            "crosstalk_interactions": [
                {"agent_a": "Vorinostat", "agent_b": "EGCG", "targets": "HDAC ⟷ Mitochondria", "coupling_strength": 0.20}
            ],
            "clinical_rationale": "Chou-Talalay CI = 0.38. HDAC baskılaması tümörün epigenetik savunmasını kırar, EGCG mitokondri membranını korur ve hemositler nodülü hızla temizler.",
            "description": "HDAC baskılaması tümörün epigenetik savunmasını kırar, EGCG mitokondri membranını korur ve hemositler nodülü hızla temizler."
        },
        "kras_g12d_vertical_blockade": {
            "id": "kras_g12d_vertical_blockade",
            "name": "Pan-RAS / KRAS G12D & SHP2 Dikey Blokaj (MRTX1133 + RMC-4550 + Ponsegromab)",
            "primary_smiles": "OC1=CC(C2=C(F)C3=NC(OC[C@@]45CCCN4C[C@H](F)C5)=NC(N6CC(N7)CCC7C6)=C3C=N2)=C8C(C#C)=C(F)C=CC8=C1",
            "components": [
                {"name": "MRTX1133 (KRAS G12D)", "dose": "2.5 nM", "dose_uM": 0.0025, "target": "KRAS G12D Switch-II", "role": "KRAS G12D Switch-II", "mechanism": "Asp12 Tuz Köprüsü & MAPK Kilitlenmesi", "dri_fold": 9.2},
                {"name": "RMC-4550 (SHP2)", "dose": "2.0 nM", "dose_uM": 0.002, "target": "SHP2 / PTPN11", "role": "SHP2 / PTPN11", "mechanism": "Allosterik Adaptif RTK Direnç Kırıcı", "dri_fold": 8.5},
                {"name": "Ponsegromab Mimetic", "dose": "0.08 µM", "dose_uM": 0.08, "target": "GDF15/GFRAL & Upd3", "role": "GDF15/GFRAL & Upd3", "mechanism": "NEJM 2024 Kaşeksi Kalkanı", "dri_fold": 12.0}
            ],
            "synergy_index_ci": 0.18,
            "chou_talalay_ci": 0.18,
            "synergy_label": "Ultra Sinerji (CI < 0.25)",
            "synergy_description": "Pan-RAS Dikey Yolak Çöküşü & Sıfır Kaşeksi",
            "bliss_excess_score": 0.046,
            "bliss_observed_kill": 0.992,
            "bliss_expected_kill": 0.946,
            "toxicity_reduction_pct": 92.0,
            "toxicity_shield_pct": 0.92,
            "potency_boost": 3.0,
            "target_potency_multiplier": 3.0,
            "dri_profile": {
                "MRTX1133": {"dose_uM": 0.0025, "dri_fold": 9.2, "sparing_pct": 89.1},
                "RMC-4550": {"dose_uM": 0.002, "dri_fold": 8.5, "sparing_pct": 88.2},
                "Ponsegromab": {"dose_uM": 0.08, "dri_fold": 12.0, "sparing_pct": 91.7}
            },
            "crosstalk_interactions": [
                {"agent_a": "MRTX1133", "agent_b": "RMC-4550", "targets": "KRAS_G12D ⟷ SHP2", "coupling_strength": 0.38},
                {"agent_a": "MRTX1133", "agent_b": "Ponsegromab", "targets": "KRAS_G12D ⟷ GDF15_Cachexia", "coupling_strength": 0.32}
            ],
            "clinical_rationale": "Chou-Talalay CI = 0.18 (Ultra Sinerji). MRTX1133 ile KRAS G12D kilitlenirken, allosterik SHP2 inhibitörü (RMC-4550) adaptif RTK direncini sıfırlar; NEJM 2024 GDF15 kalkanı kaşeksiyi %92 önler.",
            "description": "MRTX1133 ile KRAS G12D kilitlenirken, allosterik SHP2 inhibitörü (RMC-4550) adaptif RTK direncini sıfırlar; NEJM 2024 GDF15 kalkanı kaşeksiyi %92 önler."
        },
        "synthetic_lethality_parp_atr": {
            "id": "synthetic_lethality_parp_atr",
            "name": "Sentetik Ölümcüllük PARP & ATR (Olaparib + Ceralasertib + Resveratrol)",
            "primary_smiles": "O=C(N1CCN(CC1)C(=O)c1cc(ccc1F)Cc1nnc(c2c1cccc2)O)C1CC1",
            "components": [
                {"name": "Olaparib", "dose": "25 nM", "dose_uM": 0.025, "target": "PARP1/2", "role": "PARP1/2", "mechanism": "DNA Tek Zincir Onarım Tuzaklaması", "dri_fold": 7.8},
                {"name": "Ceralasertib", "dose": "20 nM", "dose_uM": 0.020, "target": "ATR Kinaz", "role": "ATR Kinaz", "mechanism": "Replikasyon Çatalı Çöküşü & Sentetik Ölüm", "dri_fold": 8.2},
                {"name": "Resveratrol", "dose": "7.5 µM", "dose_uM": 7.5, "target": "SIRT1 Kalkanı", "role": "SIRT1 Kalkanı", "mechanism": "Sağlıklı Nöron ve Hemosit Sitoproteksiyonu", "dri_fold": 9.5}
            ],
            "synergy_index_ci": 0.18,
            "chou_talalay_ci": 0.18,
            "synergy_label": "Ultra Sinerji (CI < 0.25)",
            "synergy_description": "DNA Hasar Yanıtı Katastrofisi (Synthetic Lethality)",
            "bliss_excess_score": 0.026,
            "bliss_observed_kill": 0.988,
            "bliss_expected_kill": 0.962,
            "toxicity_reduction_pct": 86.0,
            "toxicity_shield_pct": 0.86,
            "potency_boost": 2.8,
            "target_potency_multiplier": 2.8,
            "dri_profile": {
                "Olaparib": {"dose_uM": 0.025, "dri_fold": 7.8, "sparing_pct": 87.2},
                "Ceralasertib": {"dose_uM": 0.020, "dri_fold": 8.2, "sparing_pct": 87.8},
                "Resveratrol": {"dose_uM": 7.5, "dri_fold": 9.5, "sparing_pct": 89.5}
            },
            "crosstalk_interactions": [
                {"agent_a": "Olaparib", "agent_b": "Ceralasertib", "targets": "PARP ⟷ ATR", "coupling_strength": 0.38}
            ],
            "clinical_rationale": "Chou-Talalay CI = 0.18. Olaparib tek zincir onarımını dondururken Ceralasertib replikasyon kontrol noktasını patlatır; kanser hücreleri sentetik ölümcüllükle erir.",
            "description": "Olaparib tek zincir onarımını dondururken Ceralasertib replikasyon kontrol noktasını patlatır; kanser hücreleri sentetik ölümcüllükle erir."
        }
    }

    def set_active_drug(self, smiles_or_name: str, dose_uM: float = 2.5):
        """Test edilen kimyasal molekülü ve dozajı dinamik olarak değiştirir."""
        self.active_drug = self.pubchem.parse_molecule(smiles_or_name)
        self.drug_dose_uM = dose_uM
        self.active_cocktail = None

    def set_cocktail(self, cocktail_id: str, custom_cocktail: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Kombinasyon terapisi protokolünü yükler ve sinerjiyi aktif eder."""
        if custom_cocktail:
            self.COCKTAIL_REGIMENS[cocktail_id] = custom_cocktail
            self.active_cocktail = custom_cocktail
        elif cocktail_id in self.COCKTAIL_REGIMENS:
            self.active_cocktail = self.COCKTAIL_REGIMENS[cocktail_id]
        else:
            self.active_cocktail = None
            return None

        primary = (
            self.active_cocktail.get("primary_smiles") or
            (self.active_cocktail.get("components") and self.active_cocktail["components"][0].get("smiles")) or
            "CC1=NC=C(C=C1)CCN(C)C(=O)CF"
        )
        self.active_drug = self.pubchem.parse_molecule(primary)
        first_comp = self.active_cocktail.get("components", [{}])[0]
        self.drug_dose_uM = float(first_comp.get("dose_uM", 2.0))
        return self.active_cocktail

    TREATMENT_MODALITIES: Dict[str, Any] = {
        "targeted_small_molecule": {
            "id": "targeted_small_molecule",
            "name": "Hedefe Yönelik Küçük Molekül (Standart Monoterapi)",
            "category": "Targeted Monotherapy",
            "badge_color": "#00f0ff",
            "clinical_summary": "Seçici nAChR/KCg-m reseptör agonisti; nöro-immün refleks ve kemotaktik hemosit göçünü uyarır.",
            "target_pathway": "nAChRα7 -> KCg-m Nöro-Hematopoetik Aks",
            "advantages": "Düşük sağlıklı doku hasarı, fizyolojik bağışıklık mobilizasyonu",
            "limitations": "Monoterapide MEK bypass ve ABC efflux dirençli klonlar nüks (relaps) yaratabilir"
        },
        "cytotoxic_chemotherapy": {
            "id": "cytotoxic_chemotherapy",
            "name": "Sitotoksik Kemoterapi (Sisplatin + Paklitaksel)",
            "category": "Cytotoxic Chemotherapy",
            "badge_color": "#ff2a6d",
            "clinical_summary": "Yüksek doz DNA alkilleyici (Sisplatin) ve mikrotübül dondurucu (Paklitaksel). Klon direnci gözetmeksizin tümör DNA'sını çapraz bağlar.",
            "target_pathway": "DNA Adducts & M-Fazı Mitotik İğ İplikçikleri",
            "advantages": "Dirençli klonları ve hızlı bölünen tüm hücreleri kuvvetle lize eder",
            "limitations": "Yüksek doku toksisitesi (%30-38), hemosit öncüllerini baskılama riski"
        },
        "immunotherapy_cd47": {
            "id": "immunotherapy_cd47",
            "name": "Kontrol Noktası İmmünoterapisi (Anti-CD47 Don't-Eat-Me Blokajı)",
            "category": "Immune Checkpoint Blockade",
            "badge_color": "#00ff9d",
            "clinical_summary": "Kanser hücrelerinin 'beni yeme' (Don't Eat Me) kalkanını yıkar. Draper/NimC1 kaçışını sıfırlayarak hemosit fagositozunu 2.2x artırır.",
            "target_pathway": "CD47 - SIRPα / Draper Fagositoz Kontrol Noktası",
            "advantages": "İmmün-kaçış klonlarını savunmasız bırakır, toksisitesi son derece düşüktür (<%8)",
            "limitations": "Yalnızca bağışıklık sistemi yeterli yakıta (ATP/BCAA) sahip olduğunda etkilidir"
        },
        "metabolic_starvation": {
            "id": "metabolic_starvation",
            "name": "Metabolik Warburg Açlık Terapisi (2-Deoksiglukoz / 2-DG)",
            "category": "Metabolic Oncology",
            "badge_color": "#ffb703",
            "clinical_summary": "Hekzokinaz-II enzimini kilitler. Tümörün aşırı glikoz açlığını (Warburg etkisi) keserek ATP krizine sokar ve hücre bölünmesini kilitler.",
            "target_pathway": "Hekzokinaz-II (HK2) & Aerobik Glikoliz",
            "advantages": "Tümör mitozunu sıfıra indirir, kaşeksi salınımını durdurur",
            "limitations": "Tek başına hücreleri hızlı parçalamaz; sitotoksik ajanlarla birleştirilmelidir"
        },
        "metronomic_rescue": {
            "id": "metronomic_rescue",
            "name": "Metronomik Multimodal Kurtarma Protokolü (AI Şampiyon Reçetesi)",
            "category": "Metronomic Multi-modal",
            "badge_color": "#d946ef",
            "clinical_summary": "Düşük doz sürekli kemoterapi + Anti-CD47 immün kalkan kırıcı + Nöral kolinerjik ateşleme + Kurkumin kaşeksi kalkanı. Direnci sıfırlar.",
            "target_pathway": "Çoklu Eşzamanlı Hedefleme (Kinaz + DNA + CD47 + nAChR)",
            "advantages": "Toksisiteyi <%8 tutarken p53/KRAS ve MEK bypass klonlarını tamamen temizler",
            "limitations": "Çoklu bileşen formülasyonu ve hassas farmakokinetik senkronizasyon gerektirir"
        }
    }

    def set_treatment_modality(self, modality_id: str) -> Dict[str, Any]:
        """Tedavi rejimini dinamik olarak değiştirir."""
        if modality_id in self.TREATMENT_MODALITIES:
            self.active_modality = modality_id
            if modality_id == "targeted_small_molecule":
                self.set_active_drug("CC1=NC=C(C=C1)CCN(C)C(=O)CF", dose_uM=2.5)
                self.active_cocktail = None
            elif modality_id == "cytotoxic_chemotherapy":
                self.set_active_drug("Cisplatin", dose_uM=2.0)
                self.active_drug.qsar_toxicity_risk = 0.28
                self.active_cocktail = None
            elif modality_id == "metronomic_rescue":
                self.set_cocktail("immuno_mek_synergy")
                self.host_alive = True
                self.cumulative_cachectic_toxin = 0.0
            elif modality_id == "immunotherapy_cd47":
                self.active_cocktail = None
                self.host_alive = True
            elif modality_id == "metabolic_starvation":
                self.set_active_drug("2-Deoxyglucose", dose_uM=2.0)
                self.active_drug.qsar_toxicity_risk = 0.10
                self.active_cocktail = None
            return self.TREATMENT_MODALITIES[modality_id]
        return self.TREATMENT_MODALITIES["targeted_small_molecule"]

    def apply_radiation_pulse(self, dose_gy: float = 8.0) -> Dict[str, Any]:
        """
        Stereotaktik Radyoterapi (SABR / IMRT) darbesi uygular.
        Tümör dokusundaki DNA çift zincir kırıklarını (DSB) tetikleyerek
        kanser hücrelerini direnç mekanizmasından bağımsız olarak anında vurur.
        """
        self.radiation_pulses_applied += 1
        damaged_count = 0
        destroyed_count = 0

        for c in self.cancer_cells:
            if c.state in (CancerState.APOPTOTIC, CancerState.LYSED):
                continue
            damaged_count += 1
            rad_damage = float(min(75.0, dose_gy * np.random.uniform(5.5, 7.5)))
            c.health -= rad_damage
            if c.health <= 0.0:
                c.state = CancerState.LYSED
                destroyed_count += 1
                self.newly_lysed_events.append([round(float(x), 1) for x in c.position])

        # Geçici sistemik radyasyon doku stresi
        self.cumulative_cachectic_toxin += dose_gy * 0.35

        return {
            "status": "radiation_applied",
            "dose_gy": dose_gy,
            "damaged_cells": damaged_count,
            "destroyed_cells": destroyed_count,
            "remaining_cancer_cells": sum(1 for c in self.cancer_cells if c.state not in (CancerState.APOPTOTIC, CancerState.LYSED)),
            "total_pulses": self.radiation_pulses_applied
        }

    def step(self, dt_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Tüm sistemi çok ölçekli olarak 1 saniye ilerletir:
          1. İlaç-Reseptör Doygunluğu (Hill-Langmuir & FCA duyarlılığı)
          2. 185 ms'lik milisaniyelik aferent nöral döngü & 45 Hz eferent ateşleme
          3. 4 Aşamalı Lenf Bezi yakıt tüketimi & Hemosit sentezi
          4. 3D İlaç difüzyonu ve Kanser hücresi büyümesi/apoptozu
          5. 3D Hemosit kemotaksisi, tümör kapsülasyonu ve fagositoz
        """
        # 1. Reseptör Bağlanma Kinetiği (Assosiasyon / Birikim Eğrisi: 1 - exp(-k_on * [L] * t))
        kd = self.active_drug.kd_micromolar
        hill_n = self.active_drug.hill_coefficient
        steady_state_occupancy = float((self.drug_dose_uM ** hill_n) / ((kd ** hill_n) + (self.drug_dose_uM ** hill_n)))
        
        # Moleküler bağlanma hızı (k_on ve MW gecikmesi)
        tau_bind_s = max(0.5, (self.active_drug.molecular_weight / 120.0) * (kd / 0.1))
        current_occupancy = steady_state_occupancy * (1.0 - np.exp(-(self.elapsed_time_s + 1e-3) / tau_bind_s))
        
        # FCA Duyu nöron duyarlılık katsayısı ile modüle et
        fca_sensitivity = self.fca.compute_cell_target_responsiveness("sensory_orn", self.active_drug.name)
        effective_potency = float(np.clip(current_occupancy * fca_sensitivity, 0.0, 1.0))
        if self.active_cocktail:
            effective_potency = float(np.clip(effective_potency * self.active_cocktail.get("potency_boost", 1.0), 0.0, 1.0))

        # 2. Milisaniyelik Nöral Devre Çözümlemesi (FlyWire KCg-m 185 ms Refleks Döngüsü)
        if self.host_alive:
            neural_data = self.neural_circuit.simulate_macro_second_response(
                ligand_potency=effective_potency,
                duration_s=dt_seconds
            )
            efferent_hz = neural_data["efferent_firing_hz"]
            marrow_drive = neural_data["marrow_stimulation_drive"]
        else:
            # Konakçı ölüyse nöral devre ve eferent ateşleme durur (0 Hz)
            neural_data = {
                "kcg_membrane_mv": -70.0,
                "kcg_calcium_nm": 60.0,
                "snpf_release_pct": 0.0,
                "ach_quanta_nm": 0.0,
                "efferent_firing_hz": 0.0,
                "marrow_stimulation_drive": 0.0,
                "spikes_per_sec": 0,
                "voltage_trace": [-70.0],
                "calcium_trace": [60.0]
            }
            efferent_hz = 0.0
            marrow_drive = 0.0

        if marrow_drive >= 0.85 and self.time_to_first_trigger_s is None:
            self.time_to_first_trigger_s = self.elapsed_time_s

        # Mekanistik İlaç ve Kokteyl Hedef Eşleştirmesi
        modality = self.active_modality
        drug_name_lower = (self.active_drug.name or "").lower()
        drug_smiles_lower = (self.active_drug.smiles or "").lower()
        comp_text = f"{drug_name_lower} {drug_smiles_lower} "
        if self.active_cocktail and self.active_cocktail.get("components"):
            comp_text += " ".join([
                (c.get("name", "") + " " + c.get("target", "") + " " + c.get("mechanism", "") + " " + c.get("target_key", "")).lower()
                for c in self.active_cocktail["components"]
            ])

        is_immune_agonist = bool(
            any(k in comp_text for k in ["nachr", "agonist", "kcg", "f-nac", "mcn", "nicotine", "denovo"])
        )

        # 3. Lenf Bezi (Kemik İliği) 4-Aşamalı Yakıt Tüketimi ve Savunma Hücresi Üretimi
        if self.host_alive:
            drug_boost = min(1.5, self.drug_dose_uM * 0.25) if is_immune_agonist else 0.0
            new_plasma, new_lamello = self.lymph_gland.step_hematopoiesis(
                dt_seconds=dt_seconds,
                elapsed_seconds=self.elapsed_time_s,
                neural_efferent_drive=marrow_drive,
                drug_immune_boost=drug_boost
            )
        else:
            new_plasma, new_lamello = 0, 0

        # Yeni üretilen savunma hücrelerini çeper damarlardan 3D dokuya dök (Egress)
        for _ in range(new_plasma):
            pos = self._random_boundary_position()
            self.hemocyte_agents.append(HemocyteAgent3D(id=self.next_agent_id, subtype=HemocyteSubtype.PLASMATOCYTE, position=pos))
            self.next_agent_id += 1

        for _ in range(new_lamello):
            pos = self._random_boundary_position()
            self.hemocyte_agents.append(HemocyteAgent3D(id=self.next_agent_id, subtype=HemocyteSubtype.LAMELLOCYTE, position=pos, radius_um=12.0))
            self.next_agent_id += 1

        is_mct1_acidosis_cleared = bool(
            modality == "metronomic_rescue" or
            any(k in comp_text for k in ["azd3965", "mct1", "laktat", "acidosis"])
        )
        is_anti_cd47 = bool(
            modality in ("immunotherapy_cd47", "metronomic_rescue") or
            any(k in comp_text for k in ["cd47", "checkpoint", "fagositoz", "evorpacept", "alx148", "draper"])
        )
        is_metabolic = bool(
            modality == "metabolic_starvation" or
            any(k in comp_text for k in ["2-deoxyglucose", "2-dg", "glikoliz", "warburg", "telaglenastat", "gls1", "cb-839"])
        )
        is_chemo = bool(
            modality in ("cytotoxic_chemotherapy", "metronomic_rescue") or
            any(k in comp_text for k in ["cisplatin", "adduct", "alkilat", "paclitaxel", "kemoterapi"])
        )
        is_synthetic_lethality = bool(
            any(k in comp_text for k in ["olaparib", "parp"]) and any(k in comp_text for k in ["ceralasertib", "atr"])
        )
        is_kras_dual_lock = bool(
            any(k in comp_text for k in ["mrtx1133", "kras_g12d", "pan-ras"]) and any(k in comp_text for k in ["rmc-4550", "shp2"])
        )
        is_shp2_inhibited = bool(
            is_kras_dual_lock or any(k in comp_text for k in ["rmc-4550", "shp2", "ptpn11"])
        )
        is_kras_inhibited = bool(
            any(k in comp_text for k in ["mrtx1133", "kras_g12d", "pan-ras"])
        )
        is_gdf15_shielded = bool(
            any(k in comp_text for k in ["ponsegromab", "gdf15", "curcumin", "kurkumin", "resveratrol"])
        )
        is_mek_inhibited = bool(
            is_kras_dual_lock or
            any(k in comp_text for k in ["trametinib", "cobimetinib", "selumetinib", "binimetinib", "mek1", "mek2", "mek_inhibitor", "allosterik mek", "mek1/2"])
        )
        dna_damaged = bool(
            is_chemo or is_synthetic_lethality or any(k in comp_text for k in ["cisplatin", "dna", "adduct", "alkilat", "olaparib"])
        )

        # 3D İlaç Enjeksiyonu ve Fickian Difüzyon (Asidoz klerensi ile)
        self.spatial_tme.inject_drug(dose_rate=self.drug_dose_uM, dt=dt_seconds, logP=self.active_drug.logP)
        self.spatial_tme.diffuse_fields(dt=dt_seconds, mct1_inhibited=is_mct1_acidosis_cleared)
        if is_mct1_acidosis_cleared:
            self.spatial_tme.clear_lactate(clearance_boost=2.5, dt=dt_seconds)

        # Farmakodinamik afiniteye bağlı EC50 penceresi (Düşük Kd & Düşük CI = Yüksek Potens)
        if self.active_cocktail:
            ci = float(self.active_cocktail.get("chou_talalay_ci", 0.50))
            base_kd = self.active_drug.kd_micromolar
            eff_ec50 = float(np.clip(base_kd * (ci * 2.5), 0.12, 4.0))
        else:
            eff_ec50 = float(np.clip(self.active_drug.kd_micromolar * 4.0, 0.10, 8.0))

        # 5. Kanser Hücre Güncellemeleri & Onkolojik Tedavi Modalitesi
        newly_divided: List[CancerCell3D] = []
        step_cachectic_toxin = 0.0

        for c in self.cancer_cells:
            if c.state in (CancerState.APOPTOTIC, CancerState.LYSED):
                continue

            # Modaliteye Özgü Sitotoksisite ve Sentetik Ölümcüllük
            if is_chemo and modality == "cytotoxic_chemotherapy":
                c.health -= 0.65 * dt_seconds

            if is_synthetic_lethality:
                c.health -= 1.4 * dt_seconds

            if is_kras_dual_lock:
                # KRAS G12D + SHP2 dikey blokajı onkogen bağımlısı hücrelerde kaskat apoptoz tetikler
                c.health -= 1.35 * dt_seconds

            if is_metabolic and modality == "metabolic_starvation":
                c.health -= 0.75 * dt_seconds

            # Biyolojik Klonal Hedef Eşleşmesi ve Sinerji Kooperasyonu
            local_drug = self.spatial_tme.sample_drug_at(c.position)
            
            # Hücrenin kokteyl bileşenlerinin mekanizmasına duyarlılık kontrolü
            is_target_matched = False
            if c.clone_type == "sensitive":
                is_target_matched = bool(is_mek_inhibited or is_kras_inhibited or is_chemo or is_synthetic_lethality or is_metabolic)
            elif c.clone_type == "resistant_mek":
                is_target_matched = bool(is_kras_dual_lock or (is_shp2_inhibited and (is_mek_inhibited or is_kras_inhibited)) or is_synthetic_lethality)
            elif c.clone_type == "resistant_efflux":
                is_target_matched = bool(is_metabolic or is_synthetic_lethality or (is_chemo and self.drug_dose_uM >= 1.5))
            elif c.clone_type == "immune_evasive":
                is_target_matched = bool(is_kras_dual_lock or is_synthetic_lethality or is_anti_cd47)

            # Sinerjik kokteyl (CI < 0.45) yalnızca hedefi tutan ve ilacın ulaştığı klonlarda apoptozu hızlandırır
            if self.active_cocktail and is_target_matched and local_drug > 0.02:
                ci_val = float(self.active_cocktail.get("chou_talalay_ci", 0.50))
                if ci_val < 0.45:
                    synergy_kill = float(np.clip(0.35 / ci_val, 0.40, 1.35)) * dt_seconds
                    c.health -= synergy_kill

            divided, toxin, lactate = c.step(
                dt=dt_seconds,
                local_drug_conc=local_drug,
                target_ec50=eff_ec50,
                mek_inhibited=is_mek_inhibited,
                shp2_inhibited=is_shp2_inhibited,
                kras_inhibited=is_kras_inhibited,
                dna_damaged=dna_damaged,
                metabolic_starved=is_metabolic
            )

            # Laktat birikimi
            self.spatial_tme.deposit_lactate(c.position, lactate)

            # Yalnızca dikey onkogenik kilit, sentetik ölümcüllük veya metabolik açlık mitozu tamamen durdurur
            if is_metabolic or is_kras_dual_lock or is_synthetic_lethality:
                divided = False

            step_cachectic_toxin += toxin

            if divided and self.host_alive:
                d_offset = (np.random.rand(3) - 0.5) * (c.radius_um * 2.1)
                d_pos = np.clip(c.position + d_offset, 15.0, self.domain_size - 15.0)
                
                # Yavru hücre mutasyon kalıtımı (Darwinian clonal evolution)
                d_type = c.clone_type
                d_res = c.resistance_score
                # Sinerjik çoklu hedefli kokteyller mutasyon kaçışını ve relapsı baskılar
                ci_factor = float(self.active_cocktail.get("chou_talalay_ci", 1.0)) if self.active_cocktail else 1.0
                mutation_chance = 0.12 * float(np.clip(ci_factor, 0.20, 1.0))
                if d_type == "sensitive" and np.random.rand() < mutation_chance:
                    d_type = str(np.random.choice(["resistant_mek", "resistant_efflux"]))
                    d_res = float(np.random.uniform(0.75, 0.90))
                else:
                    d_res = float(min(0.98, d_res * np.random.uniform(0.98, 1.05)))

                daughter = CancerCell3D(
                    id=self.next_agent_id,
                    position=d_pos,
                    radius_um=c.radius_um,
                    clone_type=d_type,
                    resistance_score=d_res,
                    division_threshold_s=float(np.random.uniform(42.0, 56.0))
                )
                newly_divided.append(daughter)
                self.next_agent_id += 1

        self.cancer_cells.extend(newly_divided)

        # 6. Hemosit Kemotaksisi, Kuşatma ve Sitotoksisite
        active_hemocyte_count = 0
        fuel_eff = 1.0
        if hasattr(self, "fuel_engine") and self.fuel_engine.pool.atp_mM < 1.0:
            fuel_eff = 0.35

        # MCT1 inhibisyonu ile laktat asidozu nötrlendiğinde hemositler çevikleşir
        if is_mct1_acidosis_cleared:
            fuel_eff = min(1.4, fuel_eff * 1.35)

        # Hemosit fagositer gücü: Yalnızca nöro-immün agonist varlığında nöral sürüşle artar
        hemocyte_potency = 1.30 if is_immune_agonist else 1.0

        for h in self.hemocyte_agents:
            if not h.is_apoptotic and h.exhaustion_index < 1.0 and h.cytotoxic_energy > 3.0:
                active_hemocyte_count += 1
                if self.host_alive:
                    local_lac = self.spatial_tme.sample_lactate_at(h.position)
                    h.step_patrol_and_attack(
                        dt=dt_seconds,
                        cancer_cells=self.cancer_cells,
                        domain_bounds=self.spatial_tme.bounds,
                        local_lactate_mM=local_lac,
                        fuel_efficiency=fuel_eff,
                        anti_cd47_active=is_anti_cd47,
                        potency_multiplier=hemocyte_potency
                    )

        # Tükenen veya ölen hemositleri temizle
        self.hemocyte_agents = [
            h for h in self.hemocyte_agents
            if not h.is_apoptotic and h.exhaustion_index < 1.0 and h.cytotoxic_energy > 2.0
        ]

        # Canlı kanser ve klon sayıları
        viable_cancer = [c for c in self.cancer_cells if c.state not in (CancerState.APOPTOTIC, CancerState.LYSED)]
        viable_cancer_count = len(viable_cancer)
        sensitive_count = sum(1 for c in viable_cancer if getattr(c, "clone_type", "sensitive") == "sensitive")
        resistant_count = viable_cancer_count - sensitive_count

        if viable_cancer_count < self.lowest_tumor_count:
            self.lowest_tumor_count = viable_cancer_count

        # Zamanı ilerlet
        self.elapsed_time_s += dt_seconds
        stage_idx, stage_desc = self.fuel_engine.get_stage_info(self.elapsed_time_s)

        # Yeni parçalanan kanser hücrelerinin 3D patlama koordinatları
        self.newly_lysed_events = []
        for c in self.cancer_cells:
            if c.state in (CancerState.APOPTOTIC, CancerState.LYSED) and not getattr(c, "_visual_reported", False):
                self.newly_lysed_events.append([round(float(x), 1) for x in c.position])
                c._visual_reported = True

        # Toksisite Hesaplaması (Doz faktörü + Kademeli Kaşeksi birikimi + Modalite yükü)
        # Sitoprotektif kalkan (Curcumin / Resveratrol / SIRT1 / Ponsegromab GDF15) kaşektik sitokin hasarını söndürür:
        if self.active_cocktail:
            shield_red = float(self.active_cocktail.get("toxicity_reduction_pct", 0.0)) / 100.0
            if is_gdf15_shielded:
                shield_red = max(shield_red, 0.94)
            step_cachectic_toxin *= (1.0 - (shield_red * 0.90))

        self.cumulative_cachectic_toxin += step_cachectic_toxin
        dose_factor = (self.drug_dose_uM / 2.5) ** 0.8
        direct_drug_tox = self.active_drug.qsar_toxicity_risk * dose_factor

        if modality == "cytotoxic_chemotherapy":
            direct_drug_tox = max(direct_drug_tox, 0.32 * dose_factor)
            self.cumulative_cachectic_toxin += 0.85 * dt_seconds  # Korunmasız kemoterapinin doku nekrozu yükü
        elif modality == "metabolic_starvation":
            direct_drug_tox = max(direct_drug_tox, 0.10 * dose_factor)

        cachexia_burden = min(0.30, (self.cumulative_cachectic_toxin / 300.0) * 0.20)
        raw_tox = direct_drug_tox + cachexia_burden

        if self.active_cocktail:
            tox_red = self.active_cocktail.get("toxicity_reduction_pct", 0.0) / 100.0
            raw_tox *= (1.0 - tox_red)

        systemic_tox = float(np.clip(raw_tox, 0.0, 1.0))

        # Konakçı Canlılık & Toksik Ölüm Kontrolü
        if systemic_tox >= self.lethal_toxicity_threshold:
            self.host_alive = False
            host_vitality = 0.0
            clinical_outcome = "HOST_LETHALITY_OVERDOSE"
            clinical_status_text = "☠️ ORGANİZMA ÖLÜMÜ (AŞIRI TOKSİSİTE)"
        else:
            self.host_alive = True
            host_vitality = max(0.0, round((1.0 - (systemic_tox / self.lethal_toxicity_threshold)) * 100.0, 1))

            # Klinik Sonuç Sınıflandırması
            if viable_cancer_count == 0:
                clinical_outcome = "COMPLETE_REMISSION"
                clinical_status_text = "🟢 TAM REMİSYON"
            elif viable_cancer_count >= int(self.initial_tumor_count * 1.60):
                clinical_outcome = "TUMOR_PROGRESSION_ESCAPE"
                clinical_status_text = "🔴 TEDAVİ BAŞARISIZ: TÜMÖR İSTİLASI"
            elif self.lowest_tumor_count <= int(self.initial_tumor_count * 0.80) and viable_cancer_count >= int(self.lowest_tumor_count * 1.25) and resistant_count >= 0.35 * max(1, viable_cancer_count):
                clinical_outcome = "TUMOR_RELAPSE_RESISTANT"
                clinical_status_text = "🟠 DİRENÇLİ NÜKS / RELAPS"
            elif viable_cancer_count <= int(self.initial_tumor_count * 0.45):
                clinical_outcome = "PARTIAL_RESPONSE"
                clinical_status_text = "🟡 KISMİ YANIT"
            else:
                clinical_outcome = "STABLE_DISEASE"
                clinical_status_text = "⚪ DURAĞAN HASTALIK"

        snapshot = {
            "time_s": self.elapsed_time_s,
            "drug_name": self.active_cocktail["name"] if self.active_cocktail else self.active_drug.name,
            "smiles": self.active_drug.smiles,
            "receptor_occupancy": current_occupancy,
            "lysed_bursts": self.newly_lysed_events,
            "active_cocktail": self.active_cocktail,
            "efferent_hz": efferent_hz,
            "marrow_drive": marrow_drive,
            "kcg_membrane_mv": neural_data["kcg_membrane_mv"],
            "kcg_calcium_nm": neural_data["kcg_calcium_nm"],
            "snpf_release_pct": neural_data["snpf_release_pct"],
            "ach_quanta_nm": neural_data["ach_quanta_nm"],
            "spikes_per_sec": neural_data["spikes_per_sec"],
            "voltage_trace": neural_data["voltage_trace"],
            "calcium_trace": neural_data["calcium_trace"],
            "flywire_neuron_info": {
                "root_id": "720575940608530955",
                "cell_class": "Kenyon_Cell",
                "cell_sub_class": "KCg",
                "cell_type": "KCg-m",
                "dataset": "FAFB v783",
                "side": "left",
                "known_transmitters": "acetylcholine; sNPF"
            },
            "initial_tumor_count": self.initial_tumor_count,
            "cancer_cells": viable_cancer_count,
            "sensitive_cancer_cells": sensitive_count,
            "resistant_cancer_cells": resistant_count,
            "active_hemocytes": active_hemocyte_count,
            "total_egressed_hemocytes": self.lymph_gland.metrics.total_cells_egressed,
            "stage_idx": stage_idx,
            "stage_name": stage_desc,
            "pool_atp": self.fuel_engine.pool.atp_mM,
            "pool_glucose": self.fuel_engine.pool.glucose_mM,
            "pool_bcaa": self.fuel_engine.pool.bcaa_mM,
            "pool_lipids": self.fuel_engine.pool.lipids_fatty_acids_mM,
            "toxicity_pct": systemic_tox * 100.0,
            "host_alive": self.host_alive,
            "host_vitality_pct": host_vitality,
            "clinical_outcome": clinical_outcome,
            "clinical_status_text": clinical_status_text,
            "active_modality": self.active_modality,
            "modality_info": self.TREATMENT_MODALITIES.get(self.active_modality, {}),
            "radiation_pulses_applied": self.radiation_pulses_applied
        }
        self.history.append(snapshot)
        return snapshot

    def _random_boundary_position(self) -> np.ndarray:
        """3D dokunun çeper damar noktalarından rastgele pozisyon üretir."""
        axis = np.random.randint(0, 3)
        pos = np.random.uniform(15.0, self.domain_size - 15.0, size=3)
        pos[axis] = 15.0 if np.random.rand() < 0.5 else (self.domain_size - 15.0)
        return pos

    def run_benchmark(self, duration_seconds: float = 600.0) -> CandidateEvaluationResult:
        """Belirtilen süre boyunca simülasyonu çalıştırıp molekülü skorlar."""
        for _ in range(int(duration_seconds)):
            snap = self.step(dt_seconds=1.0)
            if not self.host_alive:
                break  # Toksik ölüm halinde erken sonlanma

        final_snap = self.history[-1]
        trigger_time = self.time_to_first_trigger_s if self.time_to_first_trigger_s is not None else duration_seconds
        
        if not self.host_alive:
            cleared_pct = 0.0
        else:
            cleared_pct = max(0.0, ((self.initial_tumor_count - final_snap["cancer_cells"]) / self.initial_tumor_count) * 100.0)

        return self.evaluator.compute_fitness(
            profile=self.active_drug,
            time_to_trigger_s=trigger_time,
            hemocytes_produced=self.lymph_gland.metrics.total_cells_egressed,
            tumor_clearance_pct=cleared_pct,
            toxicity_score=final_snap["toxicity_pct"] / 100.0,
            host_alive=self.host_alive,
            clinical_outcome=final_snap.get("clinical_outcome")
        )
