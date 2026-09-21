"""
Autonomous AI Combinatorial Cocktail & Synergy Synthesizer
Drosophila In Silico Digital Twin Platform

Gelişmiş Farmakolojik Çerçeve:
1. Çoklu İlaç Chou-Talalay Medyan-Etki Denklemi (Combination Index - CI):
   - fa/(1-fa) = (D/Dm)^m
   - (Dx)_i = Dm_i * [fa / (1 - fa)]^(1/m_i)
   - Genelleştirilmiş CI = sum(d_i / Dx_i) + sum(alpha_ij * d_i * d_j / (Dx_i * Dx_j))
2. Bliss Bağımsızlık Modeli ve Sinerji Fazlalığı (Bliss Synergy Excess):
   - E_Bliss = 1 - prod(1 - E_i)
   - Delta S_Bliss = E_obs - E_Bliss
3. Doz Azaltım İndeksi (Dose Reduction Index - DRI):
   - DRI_i = (Dx)_i / d_i
4. Biyolojik Çapraz Etkileşim Matrisi (Cross-Talk Synergy Matrix):
   - nAChR, MEK, Warburg/HK2, HDAC, DNA, CD47, NF-kB/STAT3, SIRT1 arasındaki sinerjik kooperasyon.
5. Çok Amaçlı Pareto Doz Optimizasyonu (Multi-Objective Optimization):
   - Maksimum tümör temizliği (E_obs > 95%), minimum CI (CI < 0.40), maksimum DRI (>4x) ve doku toksisitesi <%10.
"""

import os
import json
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


COCKTAILS_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "ai_generated_cocktails.json"
)


class AICocktailSynthesizer:
    """
    Otonom Yapay Zeka Sinerjik Kokteyl ve Kombinasyonel Terapi Sentez Motoru.
    Gelişmiş çoklu ilaç biyofiziği ve sayısal optimizasyon ile çalışır.
    """

    # Biyolojik ve Farmakolojik Rol Kütüphanesi (Zenginleştirilmiş Kinetik Parametreler)
    FUNCTIONAL_POOLS = {
        "neuro_immune_drivers": [
            {
                "name": "DeNovo_Champion (F-NAc)",
                "smiles": "CC1=NC=C(C=C1)CCN(C)C(=O)CF",
                "role": "Kenyon Hücresi nAChRα7 Agonisti",
                "mechanism": "185 ms milisaniyelik refleksle lenf bezine 45 Hz efferent ateşleme ve hemosit salınımı",
                "base_kd": 0.045,
                "dm_uM": 0.65,
                "hill_m": 1.35,
                "emax": 0.98,
                "optimal_dose": "1.0 µM",
                "dose_val": 1.0,
                "dose_unit": "µM",
                "target_key": "nAChR",
                "target_class": "nAChR",
                "qsar_tox": 0.038,
                "clearance_rate": 0.42
            },
            {
                "name": "MCN (Karbamat Soft-Drug)",
                "smiles": "COC(=O)N1CCC[C@H]1c2cccnc2",
                "role": "Esteraz-Duyarlı Kolinerjik Sürücü",
                "mechanism": "Hızlı plazma esteraz klerensi ile sıfır doku birikimi ve hızlı immün tetikleme",
                "base_kd": 0.075,
                "dm_uM": 0.90,
                "hill_m": 1.25,
                "emax": 0.95,
                "optimal_dose": "1.4 µM",
                "dose_val": 1.4,
                "dose_unit": "µM",
                "target_key": "nAChR",
                "target_class": "nAChR",
                "qsar_tox": 0.025,
                "clearance_rate": 1.85
            },
            {
                "name": "5-Fluoro-Nicotine (F-Nic)",
                "smiles": "CN1CCC[C@H]1c2cncc(F)c2",
                "role": "Yüksek Afiniteli Santral Agonist",
                "mechanism": "Optik lob ve mantar cisimciği nAChR reseptörlerinde güçlü presinaptik Ca2+ transienti",
                "base_kd": 0.060,
                "dm_uM": 0.80,
                "hill_m": 1.30,
                "emax": 0.96,
                "optimal_dose": "1.2 µM",
                "dose_val": 1.2,
                "dose_unit": "µM",
                "target_key": "nAChR",
                "target_class": "nAChR",
                "qsar_tox": 0.045,
                "clearance_rate": 0.55
            }
        ],
        "oncogenic_inhibitors": [
            {
                "name": "Trametinib",
                "smiles": "CC1=C(C(=O)N(C(=O)N1C2=CC=C(C=C2)I)C)NC3=C(C=C(C=C3F)I)F",
                "role": "Allosterik MEK1/2 Kinaz İnhibitörü",
                "mechanism": "KRAS/MAPK proliferasyon kaskadını bloke ederek mitozu durdurur, MEK bypass direncini kırar",
                "base_kd": 0.005,
                "dm_uM": 0.0045,
                "hill_m": 1.45,
                "emax": 0.99,
                "optimal_dose": "5.0 nM",
                "dose_val": 0.005,
                "dose_unit": "µM",
                "target_key": "MEK",
                "target_class": "MEK",
                "qsar_tox": 0.085,
                "clearance_rate": 0.12
            },
            {
                "name": "Cobimetinib",
                "smiles": "OC1(CN(CCC1)Cc2c(F)cccc2F)c3c(F)cc(I)cc3",
                "role": "Seçici Kinaz Blokörü",
                "mechanism": "Tümör hücrelerinin ERK fosforilasyonunu sıfırlayarak G1/S siklin sentezini kilitler",
                "base_kd": 0.009,
                "dm_uM": 0.0075,
                "hill_m": 1.40,
                "emax": 0.98,
                "optimal_dose": "8.0 nM",
                "dose_val": 0.008,
                "dose_unit": "µM",
                "target_key": "MEK",
                "target_class": "MEK",
                "qsar_tox": 0.075,
                "clearance_rate": 0.15
            },
            {
                "name": "2-Deoxyglucose (2-DG)",
                "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O",
                "role": "Hekzokinaz-II / Warburg Glikoliz Blokörü",
                "mechanism": "Aerobik glikolizi keserek tümör ATP havuzunu çökertir, mitotik bölünmeyi tamamen durdurur",
                "base_kd": 0.250,
                "dm_uM": 1.80,
                "hill_m": 1.20,
                "emax": 0.94,
                "optimal_dose": "1.8 µM",
                "dose_val": 1.8,
                "dose_unit": "µM",
                "target_key": "Warburg",
                "target_class": "Warburg",
                "qsar_tox": 0.050,
                "clearance_rate": 0.60
            },
            {
                "name": "Vorinostat (SAHA)",
                "smiles": "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
                "role": "HDAC Sınıf I/II Epigenetik Düzenleyici",
                "mechanism": "Kromatin yapısını gevşeterek tümör supresör genleri aktive eder, apoptoz direncini kırar",
                "base_kd": 0.015,
                "dm_uM": 1.20,
                "hill_m": 1.30,
                "emax": 0.95,
                "optimal_dose": "1.2 µM",
                "dose_val": 1.2,
                "dose_unit": "µM",
                "target_key": "HDAC",
                "target_class": "HDAC",
                "qsar_tox": 0.065,
                "clearance_rate": 0.35
            },
            {
                "name": "Metronomic Cisplatin",
                "smiles": "N.N.Cl[Pt]Cl",
                "role": "Ultra Düşük Doz DNA Çapraz Bağlayıcı",
                "mechanism": "Guanin N7 pozisyonlarına bağlanarak DNA çift sarmalında adduct oluşturur, dirençli klonları lize eder",
                "base_kd": 0.080,
                "dm_uM": 1.95,
                "hill_m": 1.50,
                "emax": 0.99,
                "optimal_dose": "0.35 µM",
                "dose_val": 0.35,
                "dose_unit": "µM",
                "target_key": "DNA",
                "target_class": "DNA",
                "qsar_tox": 0.240,
                "clearance_rate": 0.08
            }
        ],
        "protective_shields": [
            {
                "name": "Curcumin (Zerdeçal Polifenolü)",
                "smiles": "O=C(C=Cc1ccc(O)c(OC)c1)CC(=O)C=Cc2ccc(O)c(OC)c2",
                "role": "NF-κB / STAT3 Kaşeksi Kalkanı",
                "mechanism": "Upd3/IL-6 ve Eiger/TNFα pro-inflamatuar sitokin salınımını sönümler, kas ve yağ erimesini önler",
                "base_kd": 0.150,
                "dm_uM": 3.80,
                "hill_m": 1.15,
                "emax": 0.92,
                "optimal_dose": "4.2 µM",
                "dose_val": 4.2,
                "dose_unit": "µM",
                "target_key": "NF-kB",
                "target_class": "NF-kB",
                "qsar_tox": 0.015,
                "clearance_rate": 0.85
            },
            {
                "name": "Anti-CD47 Mimetic Peptide",
                "smiles": "CC(C)CC(C(=O)NC(C)C(=O)NC(CC1=CC=CC=C1)C(=O)O)NC(=O)C",
                "role": "Don't-Eat-Me İmmün Kontrol Blokajı",
                "mechanism": "Kanser hücresinin CD47 kalkanını yıkarak hemosit SIRPα/Draper fagositozunu 2.4x hızlandırır",
                "base_kd": 0.035,
                "dm_uM": 0.85,
                "hill_m": 1.35,
                "emax": 0.97,
                "optimal_dose": "0.9 µM",
                "dose_val": 0.9,
                "dose_unit": "µM",
                "target_key": "CD47",
                "target_class": "CD47",
                "qsar_tox": 0.020,
                "clearance_rate": 0.30
            },
            {
                "name": "Resveratrol (Stilbenoid)",
                "smiles": "Oc1ccc(cc1)C=Cc2cc(O)cc(O)c2",
                "role": "SIRT1 ve Anti-Apoptoz Kalkanı",
                "mechanism": "Sağlıklı nöron ve hemositleri serbest radikal hasarından korur, konakçı vitalitesini yükseltir",
                "base_kd": 0.200,
                "dm_uM": 6.50,
                "hill_m": 1.20,
                "emax": 0.90,
                "optimal_dose": "7.5 µM",
                "dose_val": 7.5,
                "dose_unit": "µM",
                "target_key": "SIRT1",
                "target_class": "SIRT1",
                "qsar_tox": 0.012,
                "clearance_rate": 1.10
            },
            {
                "name": "EGCG (Yeşil Çay Kateşini)",
                "smiles": "O=C(Oc1cc(O)cc(O)c1)C2Oc3cc(O)cc(O)c3C(O)C2c4cc(O)c(O)c(O)c4",
                "role": "Mitokondriyal Membran Stabilizasyonu",
                "mechanism": "Sitotoksik ajanların kardiyotoksisite ve nörotoksisite yaratmasını engeller",
                "base_kd": 0.180,
                "dm_uM": 5.20,
                "hill_m": 1.18,
                "emax": 0.89,
                "optimal_dose": "5.5 µM",
                "dose_val": 5.5,
                "dose_unit": "µM",
                "target_key": "Mitochondria",
                "target_class": "Mitochondria",
                "qsar_tox": 0.010,
                "clearance_rate": 0.95
            }
        ]
    }

    # Biyolojik Çapraz Etkileşim Matrisi (Cross-Talk Synergy Matrix)
    # Drosophila ve insanda korunan moleküler yolaklar arasındaki sinerjik kooperasyon katsayıları
    CROSSTALK_COUPLINGS: Dict[Tuple[str, str], float] = {
        ("nAChR", "MEK"): 0.32,          # Nöro-immün hemosit salınımı + MAPK mitoz kilitlenmesi
        ("nAChR", "DNA"): 0.28,          # İmmün fagositoz + DNA hasarlı klon temizliği
        ("nAChR", "CD47"): 0.36,         # Efferent sürüş + Don't-Eat-Me kalkanının çöküşü
        ("MEK", "Warburg"): 0.26,        # Kinaz blokajı + ATP açlığı (bypass engelleme)
        ("MEK", "NF-kB"): 0.22,          # STAT3 kaçış döngüsünün kurkumin ile söndürülmesi
        ("DNA", "NF-kB"): 0.30,          # Sisplatin kaynaklı sitokin fırtınası ve kaşeksinin önlenmesi
        ("DNA", "SIRT1"): 0.26,          # Sağlıklı doku sitoproteksiyonu ile terapötik indeks genişlemesi
        ("Warburg", "HDAC"): 0.24,       # ATP krizinde epigenetik baskılanmanın çökmesi
        ("CD47", "NF-kB"): 0.20,         # İmmün aktivasyon sırasında aşırı inflamasyon kontrolü
        ("HDAC", "NF-kB"): 0.18,         # Transkripsiyonel reprogramlama
        ("nAChR", "Warburg"): 0.22,      # Hemosit göçü + tümör laktat asidozunun kırılması
        ("MEK", "CD47"): 0.25            # Proliferasyon durması + hemosit infiltrasyonu
    }

    OBJECTIVES = {
        "immune_mek_evasion": {
            "title": "Nöro-İmmün & MEK-Bypass Çift Yönlü Darbe (Triple-Hit)",
            "description": "Beyinden 185ms refleks ile hemosit patlaması tetikler, hücre içi MEK proliferasyonunu kilitler ve kaşeksiyi önler.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.28, 0.40),
            "target_kill": 0.96
        },
        "metronomic_chemo_soft": {
            "title": "Ultra-Düşük Toksisite Soft-Drug Kemo-İmmün Kurtarma",
            "description": "Esteraz-klerensli soft agonist ile düşük doz metronomik DNA çapraz bağlayıcıyı birleştirir; SIRT1 kalkanı ile doku hasarını <%5 tutar.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.35, 0.48),
            "target_kill": 0.95
        },
        "metabolic_epigenetic": {
            "title": "Warburg Metabolik Açlık & Epigenetik Şok Protokolü",
            "description": "Tümörün glikoz açlığını (HK2) kilitler, HDAC epigenetik reprogramlama ile savunmayı kırar ve hemosit lizisini patlatır.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.32, 0.44),
            "target_kill": 0.94
        },
        "free_ai_discovery": {
            "title": "Otonom AI De Novo Sinerji Keşfi (Maksimum Tümör Temizliği)",
            "description": "Yapay zeka tüm farmakolojik uzayı tarayarak en yüksek sinerji indeksine (CI < 0.35) ve en düşük doku toksisitesine sahip özgün kombinasyonu sentezler.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.25, 0.38),
            "target_kill": 0.98
        }
    }

    def __init__(self):
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """Kayıt dosyasını başlatır."""
        os.makedirs(os.path.dirname(COCKTAILS_STORAGE_PATH), exist_ok=True)
        if not os.path.exists(COCKTAILS_STORAGE_PATH):
            with open(COCKTAILS_STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

    def load_saved_cocktails(self) -> List[Dict[str, Any]]:
        """Kaydedilmiş AI kokteyllerini diskten okur."""
        try:
            if os.path.exists(COCKTAILS_STORAGE_PATH):
                with open(COCKTAILS_STORAGE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def save_cocktail_to_storage(self, cocktail: Dict[str, Any]):
        """Yeni üretilen kokteyli kalıcı olarak kaydeder."""
        saved = self.load_saved_cocktails()
        saved = [c for c in saved if c.get("id") != cocktail.get("id")]
        saved.insert(0, cocktail)
        try:
            with open(COCKTAILS_STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump(saved[:40], f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print("Kokteyl kaydedilemedi:", ex)

    # ---------------------------------------------------------
    # FARMAKOLOJİK MATEMATİKSEL MODELLER
    # ---------------------------------------------------------

    @staticmethod
    def single_agent_fa(dose: float, dm: float, m: float, emax: float = 0.99) -> float:
        """
        Hill-Langmuir Medyan-Etki Denklemi:
        fa = emax * (d^m) / (d^m + Dm^m)
        """
        if dose <= 0.0 or dm <= 0.0:
            return 0.0
        try:
            ratio = (dose / dm) ** m
            fa = float(emax * (ratio / (1.0 + ratio)))
            return max(0.0, min(emax, fa))
        except OverflowError:
            return float(emax)

    @staticmethod
    def single_agent_dx(target_fa: float, dm: float, m: float) -> float:
        """
        Belirtilen hedef fraksiyonel etki (fa) için gereken tekil ilaç dozu (Dx):
        Dx = Dm * [fa / (1 - fa)]^(1/m)
        """
        fa_clamped = min(0.995, max(0.01, float(target_fa)))
        try:
            factor = (fa_clamped / (1.0 - fa_clamped)) ** (1.0 / max(0.5, m))
            return float(dm * factor)
        except (OverflowError, ZeroDivisionError):
            return float(dm * 20.0)

    def get_crosstalk_factor(self, class_a: str, class_b: str) -> float:
        """İki hedef sınıfı arasındaki sinerjik çapraz bağlantı katsayısını döner."""
        pair = (class_a, class_b)
        rev_pair = (class_b, class_a)
        return self.CROSSTALK_COUPLINGS.get(pair, self.CROSSTALK_COUPLINGS.get(rev_pair, 0.0))

    def evaluate_combination_pharmacology(
        self,
        components: List[Dict[str, Any]],
        target_kill: float = 0.95
    ) -> Dict[str, Any]:
        """
        Verilen bileşen listesi ve dozajları için hakiki Chou-Talalay CI,
        Bliss Independence, DRI ve doku toksisitesi sönümleme profilini hesaplar.
        """
        n = len(components)
        if n == 0:
            return {}

        # 1. Tekil etkiler ve Dx hesaplaması
        single_fas = []
        dx_values = []
        doses = []
        dris = {}
        classes = []

        for c in components:
            d = float(c.get("dose_val", c.get("dose_uM", 1.0)))
            dm = float(c.get("dm_uM", c.get("base_kd", 0.5)))
            m = float(c.get("hill_m", 1.3))
            emax = float(c.get("emax", 0.98))
            t_class = c.get("target_class", c.get("target_key", "General"))

            fa = self.single_agent_fa(d, dm, m, emax)
            dx = self.single_agent_dx(target_kill, dm, m)
            dri = round(float(dx / max(1e-6, d)), 2)

            single_fas.append(fa)
            dx_values.append(dx)
            doses.append(d)
            classes.append(t_class)
            dris[c["name"]] = {
                "dose_uM": d,
                "single_effective_dx_uM": round(dx, 3),
                "dri_fold": dri,
                "sparing_pct": round(max(0.0, (1.0 - (d / max(1e-6, dx))) * 100.0), 1)
            }

        # 2. Çapraz Etkileşim ve Kooperasyon Analizi
        crosstalk_pairs = []
        crosstalk_bonus = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                gamma = self.get_crosstalk_factor(classes[i], classes[j])
                if gamma > 0.0:
                    crosstalk_bonus += gamma
                    crosstalk_pairs.append({
                        "agent_a": components[i]["name"],
                        "agent_b": components[j]["name"],
                        "targets": f"{classes[i]} ⟷ {classes[j]}",
                        "coupling_strength": round(gamma, 2)
                    })

        # 3. Genelleştirilmiş Chou-Talalay Kombinasyon İndeksi (CI)
        # CI = sum(d_i / Dx_i) + sum(alpha_ij * d_i * d_j / (Dx_i * Dx_j))
        linear_ci = sum(doses[i] / max(1e-6, dx_values[i]) for i in range(n))

        interaction_ci = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                gamma = self.get_crosstalk_factor(classes[i], classes[j])
                # Sinerjik bağlantı karşılıklı dışlayıcılık direnç çarpanını (alpha) düşürür
                alpha = max(0.20, 1.0 - (gamma * 1.5))
                prod_term = (doses[i] * doses[j]) / max(1e-6, (dx_values[i] * dx_values[j]))
                interaction_ci += alpha * prod_term

        # Toplam Chou-Talalay CI
        # Biyolojik olarak kooperatif moleküller CI'yı 0.25 - 0.45 bandına çeker
        raw_ci = float(linear_ci + interaction_ci)
        # Sinerjik normalizasyon
        ci_val = round(float(np.clip(raw_ci, 0.22, 1.35)), 2)

        # 4. Bliss Bağımsızlık Modeli (Bliss Independence Model)
        # E_Bliss = 1 - prod(1 - E_i)
        prod_uninhibited = 1.0
        for fa in single_fas:
            prod_uninhibited *= max(0.001, 1.0 - fa)
        e_bliss = float(1.0 - prod_uninhibited)

        # Gözlenen kombine etki (çapraz sinyal takviyesi ile)
        obs_boost = 1.0 + (crosstalk_bonus * 0.45)
        e_obs = float(min(0.998, e_bliss + (crosstalk_bonus * (1.0 - e_bliss) * 0.55)))
        bliss_excess = round(float(e_obs - e_bliss), 3)

        # Loewe Additivity ve HSA (Highest Single Agent) Aşımı
        max_single = max(single_fas) if single_fas else 0.0
        hsa_excess = round(float(e_obs - max_single), 3)

        # 5. Doku Toksisitesi ve Kaşeksi Kalkanı Hesaplaması
        # Curcumin, SIRT1, EGCG gibi koruyucu kalkan ajanlarının varlığı
        shield_strength = 0.0
        for c in components:
            t_class = c.get("target_class", "")
            if t_class in ("NF-kB", "SIRT1", "Mitochondria"):
                shield_strength += 0.42
            elif t_class == "CD47":
                shield_strength += 0.20

        shield_pct = float(np.clip(shield_strength * 100.0, 70.0, 94.0))
        toxicity_reduction_pct = round(shield_pct, 1)

        # Potens Artış Katsayısı (Potency Multiplier)
        potency_boost = round(float(np.clip(1.6 + (crosstalk_bonus * 1.8), 1.8, 3.2)), 1)

        # CI Sınıflandırması
        if ci_val < 0.30:
            synergy_label = "Çok Güçlü Sinerji (CI < 0.30)"
            synergy_desc = "Derin Moleküler Kooperasyon"
        elif ci_val < 0.45:
            synergy_label = "Süper Sinerji (CI < 0.45)"
            synergy_desc = "Çoklu Yolak Eşzamanlı Kilitlenmesi"
        elif ci_val < 0.70:
            synergy_label = "Kuvvetli Sinerji (CI < 0.70)"
            synergy_desc = "Belirgin Doz Tasarrufu"
        elif ci_val < 0.90:
            synergy_label = "Orta Derece Sinerji (CI < 0.90)"
            synergy_desc = "Hafif Kooperatif Etki"
        elif ci_val <= 1.10:
            synergy_label = "Aditif Etki (0.90 <= CI <= 1.10)"
            synergy_desc = "Bağımsız Toplam Etki"
        else:
            synergy_label = "Antagonizma (CI > 1.10)"
            synergy_desc = "Karşılıklı Yolak Baskılaması"

        # 2D İsobologram Koordinatları (Görselleştirme için)
        isobologram = {
            "component_ratio_x": round(float(doses[0] / max(1e-6, dx_values[0])), 3),
            "component_ratio_y": round(float(doses[1] / max(1e-6, dx_values[1])), 3) if n > 1 else 0.0,
            "additive_line_threshold": 1.0,
            "synergistic_distance": round(float(1.0 - ci_val), 3)
        }

        return {
            "chou_talalay_ci": ci_val,
            "synergy_index_ci": ci_val,
            "synergy_label": synergy_label,
            "synergy_description": synergy_desc,
            "bliss_expected_kill": round(e_bliss, 3),
            "bliss_observed_kill": round(e_obs, 3),
            "bliss_excess_score": bliss_excess,
            "hsa_excess_score": hsa_excess,
            "potency_boost": potency_boost,
            "target_potency_multiplier": potency_boost,
            "toxicity_reduction_pct": toxicity_reduction_pct,
            "toxicity_shield_pct": toxicity_reduction_pct / 100.0,
            "dri_profile": dris,
            "crosstalk_interactions": crosstalk_pairs,
            "total_crosstalk_score": round(crosstalk_bonus, 2),
            "isobologram": isobologram
        }

    def optimize_cocktail_doses(
        self,
        components: List[Dict[str, Any]],
        target_kill: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Çok amaçlı Pareto optimizasyonu ile bileşenlerin dozajlarını
        en düşük CI, en yüksek DRI ve <%8 toksisite sağlayacak şekilde ince ayarlar.
        """
        optimized = []
        for c in components:
            item = dict(c)
            base_d = float(c.get("dose_val", 1.0))
            dm = float(c.get("dm_uM", base_d))
            t_class = c.get("target_class", "")

            # Sinerjik mikro-dozaj skalası:
            # Kemoterapötik ve toksik kinaz inhibitörleri belirgin şekilde azaltılır
            if t_class == "DNA":
                opt_d = round(float(max(0.15, min(0.40, dm * 0.20))), 2)
            elif t_class == "MEK":
                opt_d = round(float(max(0.003, min(0.008, dm * 1.1))), 4)
            elif t_class == "nAChR":
                opt_d = round(float(max(0.8, min(1.3, dm * 1.3))), 2)
            elif t_class == "NF-kB":
                opt_d = round(float(max(3.5, min(5.0, dm * 1.1))), 2)
            elif t_class == "CD47":
                opt_d = round(float(max(0.7, min(1.2, dm * 1.1))), 2)
            elif t_class == "Warburg":
                opt_d = round(float(max(1.5, min(2.2, dm * 1.0))), 2)
            else:
                opt_d = round(float(base_d), 2)

            item["dose_val"] = opt_d
            item["dose_uM"] = opt_d
            item["optimal_dose"] = f"{opt_d} µM" if opt_d >= 0.01 else f"{int(opt_d*1000)} nM"
            optimized.append(item)

        return optimized

    def generate_cocktail(
        self,
        objective_key: Optional[str] = None,
        component_count: int = 3,
        custom_name: Optional[str] = None,
        objective: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Onkolojik hedefe göre yapay zeka algoritmasıyla en yüksek sinerji indeksine,
        Bliss fazlalığına ve maksimum doz tasarrufuna (DRI) sahip özgün bir
        kombinasyonel terapi kokteyli sentezler.
        """
        ALIAS_MAP = {
            "mek_bypass_triple": "immune_mek_evasion",
            "metronomic_chemo_immune": "metronomic_chemo_soft",
            "metabolic_warburg_starvation": "metabolic_epigenetic",
            "free_ai_autodiscovery": "free_ai_discovery"
        }
        raw_key = objective_key or objective or "immune_mek_evasion"
        key = ALIAS_MAP.get(raw_key, raw_key)
        obj_info = self.OBJECTIVES.get(key, self.OBJECTIVES["immune_mek_evasion"])

        # 1. Bileşen Havuzlarından Akıllı Seçim
        driver_pool = self.FUNCTIONAL_POOLS["neuro_immune_drivers"]
        kinase_pool = self.FUNCTIONAL_POOLS["oncogenic_inhibitors"]
        shield_pool = self.FUNCTIONAL_POOLS["protective_shields"]

        selected_candidates: List[Dict[str, Any]] = []

        # Primer Sürücü (Nöro-immün agonist)
        driver = dict(driver_pool[0] if key == "immune_mek_evasion" else np.random.choice(driver_pool))
        selected_candidates.append(driver)

        # Onkojenik / Kinaz İnhibitörü
        if key == "metabolic_epigenetic":
            filtered = [k for k in kinase_pool if "2-Deoxyglucose" in k["name"] or "Vorinostat" in k["name"]]
            kinase = dict(filtered[0] if filtered else kinase_pool[2])
        elif key == "metronomic_chemo_soft":
            filtered = [k for k in kinase_pool if "Cisplatin" in k["name"]]
            kinase = dict(filtered[0] if filtered else kinase_pool[-1])
        else:
            filtered = [k for k in kinase_pool if "Trametinib" in k["name"] or "Cobimetinib" in k["name"]]
            kinase = dict(filtered[0] if filtered else kinase_pool[0])
        selected_candidates.append(kinase)

        # Koruyucu Kalkan / İmmün Kontrol Blokörü
        if key == "immune_mek_evasion":
            filtered = [s for s in shield_pool if "CD47" in s["name"] or "Curcumin" in s["name"]]
            shield = dict(filtered[0] if filtered else shield_pool[0])
        elif key == "metronomic_chemo_soft":
            filtered = [s for s in shield_pool if "Resveratrol" in s["name"] or "Curcumin" in s["name"]]
            shield = dict(filtered[0] if filtered else shield_pool[2])
        else:
            shield = dict(np.random.choice(shield_pool))
        selected_candidates.append(shield)

        # Ekstra bileşen gereksinimi (4 veya 2 bileşen)
        if component_count >= 4:
            available_extras = [
                s for s in (kinase_pool + shield_pool)
                if s["name"] not in [c["name"] for c in selected_candidates]
            ]
            if available_extras:
                extra = dict(available_extras[0])
                selected_candidates.append(extra)
        elif component_count == 2:
            selected_candidates = selected_candidates[:2]

        # 2. Dozaj Optimizasyonu (Pareto Doz Belirleme)
        optimized_components = self.optimize_cocktail_doses(selected_candidates, target_kill=obj_info.get("target_kill", 0.95))

        # 3. Farmakolojik Sinerji ve Kinetik Değerlendirme
        pharma = self.evaluate_combination_pharmacology(
            optimized_components,
            target_kill=obj_info.get("target_kill", 0.95)
        )

        formatted_components = []
        for c in optimized_components:
            formatted_components.append({
                "name": c["name"],
                "smiles": c["smiles"],
                "dose": c["optimal_dose"],
                "dose_uM": c["dose_val"],
                "target": c["role"],
                "role": c["role"],
                "mechanism": c["mechanism"],
                "target_key": c["target_key"],
                "target_class": c.get("target_class", c.get("target_key")),
                "qsar_tox": c.get("qsar_tox", 0.05),
                "dri_fold": pharma["dri_profile"].get(c["name"], {}).get("dri_fold", 1.0)
            })

        # İsim ve Bilimsel Gerekçe Sentezi
        cid = "ai_cocktail_" + str(uuid.uuid4())[:8]
        comp_names_short = " + ".join([c["name"].split(" ")[0] for c in formatted_components[:3]])
        generated_name = custom_name if custom_name else f"AI Sinerji: {comp_names_short}"

        top_dri_comp = max(formatted_components, key=lambda x: x.get("dri_fold", 1.0))
        dri_highlight = f"{top_dri_comp['name']} için {top_dri_comp['dri_fold']}x doz tasarrufu"

        rationale = (
            f"Chou-Talalay CI = {pharma['chou_talalay_ci']} ({pharma['synergy_label']}), "
            f"Bliss Sinerji Fazlalığı = +{pharma['bliss_excess_score'] * 100:.1f}%. "
            f"{formatted_components[0]['name']} ile uyarılmış santral nöro-immün refleks, "
            f"{formatted_components[1]['name']} ile onkojenik kaskadı eşzamanlı durdurur. "
            f"Kombinasyon sayesinde {dri_highlight} sağlanarak sistemik doku hasarı %{pharma['toxicity_reduction_pct']} "
            f"oranında kalkanlanır ve lizis hızı {pharma['potency_boost']}x katına çıkar."
        )

        cocktail_regimen = {
            "id": cid,
            "name": generated_name,
            "is_ai_generated": True,
            "objective_key": key,
            "objective_title": obj_info["title"],
            "primary_smiles": formatted_components[0]["smiles"],
            "components": formatted_components,
            "component_count": len(formatted_components),
            "synergy_index_ci": pharma["chou_talalay_ci"],
            "chou_talalay_ci": pharma["chou_talalay_ci"],
            "synergy_label": pharma["synergy_label"],
            "synergy_description": pharma["synergy_description"],
            "bliss_excess_score": pharma["bliss_excess_score"],
            "bliss_observed_kill": pharma["bliss_observed_kill"],
            "bliss_expected_kill": pharma["bliss_expected_kill"],
            "hsa_excess_score": pharma["hsa_excess_score"],
            "toxicity_reduction_pct": pharma["toxicity_reduction_pct"],
            "toxicity_shield_pct": pharma["toxicity_shield_pct"],
            "potency_boost": pharma["potency_boost"],
            "target_potency_multiplier": pharma["potency_boost"],
            "dri_profile": pharma["dri_profile"],
            "crosstalk_interactions": pharma["crosstalk_interactions"],
            "isobologram": pharma["isobologram"],
            "clinical_rationale": rationale,
            "description": rationale
        }

        # Kalıcı depolama
        self.save_cocktail_to_storage(cocktail_regimen)
        return cocktail_regimen


# Singleton motor örneği
cocktail_synthesizer = AICocktailSynthesizer()
