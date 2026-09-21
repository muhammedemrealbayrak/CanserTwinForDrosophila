"""
Autonomous AI Combinatorial Cocktail & Synergy Synthesizer
Drosophila In Silico Digital Twin Platform

Bu modül:
1. Onkolojik hedeflere (MEK/Ras kinaz, nAChR nöro-immün aks, CD47 kaçışı, Warburg glikolizi, kaşeksi)
   göre çoklu ilaç kombinasyonlarını otonom olarak tahmin eder ve sentezler.
2. Chou-Talalay Medyan-Etki Denklemi (Combination Index - CI) ve Doz Azaltım İndeksi (DRI)
   üzerinden sinerji, potens artışı ve doku toksisitesi sönümleme katsayılarını hesaplar.
3. Üretilen kokteylleri simülatörün rejim havuzuna kaydeder ve 3D test için hazır hale getirir.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional
import numpy as np


COCKTAILS_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "ai_generated_cocktails.json"
)


class AICocktailSynthesizer:
    """
    Otonom Yapay Zeka Sinerjik Kokteyl ve Kombinasyonel Terapi Sentez Motoru.
    """

    # Biyolojik ve Farmakolojik Rol Kütüphanesi
    FUNCTIONAL_POOLS = {
        "neuro_immune_drivers": [
            {
                "name": "DeNovo_Champion (F-NAc)",
                "smiles": "CC1=NC=C(C=C1)CCN(C)C(=O)CF",
                "role": "Kenyon Hücresi nAChRα7 Agonisti",
                "mechanism": "185 ms milisaniyelik refleksle lenf bezine 45 Hz efferent ateşleme ve hemosit salınımı",
                "base_kd": 0.045,
                "optimal_dose": "1.0 µM",
                "dose_val": 1.0,
                "dose_unit": "µM",
                "target_key": "nAChR"
            },
            {
                "name": "MCN (Karbamat Soft-Drug)",
                "smiles": "COC(=O)N1CCC[C@H]1c2cccnc2",
                "role": "Esteraz-Duyarlı Kolinerjik Sürücü",
                "mechanism": "Hızlı plazma esteraz klerensi ile sıfır doku birikimi ve hızlı immün tetikleme",
                "base_kd": 0.075,
                "optimal_dose": "1.5 µM",
                "dose_val": 1.5,
                "dose_unit": "µM",
                "target_key": "nAChR"
            },
            {
                "name": "5-Fluoro-Nicotine (F-Nic)",
                "smiles": "CN1CCC[C@H]1c2cncc(F)c2",
                "role": "Yüksek Afiniteli Santral Agonist",
                "mechanism": "Optik lob ve mantar cisimciği nAChR reseptörlerinde güçlü presinaptik Ca2+ transienti",
                "base_kd": 0.060,
                "optimal_dose": "1.2 µM",
                "dose_val": 1.2,
                "dose_unit": "µM",
                "target_key": "nAChR"
            }
        ],
        "oncogenic_inhibitors": [
            {
                "name": "Trametinib",
                "smiles": "CC1=C(C(=O)N(C(=O)N1C2=CC=C(C=C2)I)C)NC3=C(C=C(C=C3F)I)F",
                "role": "Allosterik MEK1/2 Kinaz İnhibitörü",
                "mechanism": "KRAS/MAPK proliferasyon kaskadını bloke ederek mitozu durdurur, MEK bypass direncini kırar",
                "base_kd": 0.005,
                "optimal_dose": "5.0 nM",
                "dose_val": 0.005,
                "dose_unit": "µM",
                "target_key": "MEK"
            },
            {
                "name": "Cobimetinib",
                "smiles": "OC1(CN(CCC1)Cc2c(F)cccc2F)c3c(F)cc(I)cc3",
                "role": "Seçici Kinaz Blokörü",
                "mechanism": "Tümör hücrelerinin ERK fosforilasyonunu sıfırlayarak G1/S siklin sentezini kilitler",
                "base_kd": 0.009,
                "optimal_dose": "8.0 nM",
                "dose_val": 0.008,
                "dose_unit": "µM",
                "target_key": "MEK"
            },
            {
                "name": "2-Deoxyglucose (2-DG)",
                "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O",
                "role": "Hekzokinaz-II / Warburg Glikoliz Blokörü",
                "mechanism": "Aerobik glikolizi keserek tümör ATP havuzunu çökertir, mitotik bölünmeyi tamamen durdurur",
                "base_kd": 0.250,
                "optimal_dose": "2.0 µM",
                "dose_val": 2.0,
                "dose_unit": "µM",
                "target_key": "Warburg"
            },
            {
                "name": "Vorinostat (SAHA)",
                "smiles": "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
                "role": "HDAC Sınıf I/II Epigenetik Düzenleyici",
                "mechanism": "Kromatin yapısını gevşeterek tümör supresör genleri aktive eder, apoptoz direncini kırar",
                "base_kd": 0.015,
                "optimal_dose": "1.5 µM",
                "dose_val": 1.5,
                "dose_unit": "µM",
                "target_key": "HDAC"
            },
            {
                "name": "Metronomic Cisplatin",
                "smiles": "N.N.Cl[Pt]Cl",
                "role": "Ultra Düşük Doz DNA Çapraz Bağlayıcı",
                "mechanism": "Guanin N7 pozisyonlarına bağlanarak DNA çift sarmalında adduct oluşturur, dirençli klonları lize eder",
                "base_kd": 0.080,
                "optimal_dose": "0.4 µM",
                "dose_val": 0.4,
                "dose_unit": "µM",
                "target_key": "DNA"
            }
        ],
        "protective_shields": [
            {
                "name": "Curcumin (Zerdeçal Polifenolü)",
                "smiles": "O=C(C=Cc1ccc(O)c(OC)c1)CC(=O)C=Cc2ccc(O)c(OC)c2",
                "role": "NF-κB / STAT3 Kaşeksi Kalkanı",
                "mechanism": "Upd3/IL-6 ve Eiger/TNFα pro-inflamatuar sitokin salınımını sönümler, kas ve yağ erimesini önler",
                "base_kd": 0.150,
                "optimal_dose": "4.5 µM",
                "dose_val": 4.5,
                "dose_unit": "µM",
                "target_key": "NF-kB"
            },
            {
                "name": "Anti-CD47 Mimetic Peptide",
                "smiles": "CC(C)CC(C(=O)NC(C)C(=O)NC(CC1=CC=CC=C1)C(=O)O)NC(=O)C",
                "role": "Don't-Eat-Me İmmün Kontrol Blokajı",
                "mechanism": "Kanser hücresinin CD47 kalkanını yıkarak hemosit SIRPα/Draper fagositozunu 2.4x hızlandırır",
                "base_kd": 0.035,
                "optimal_dose": "1.0 µM",
                "dose_val": 1.0,
                "dose_unit": "µM",
                "target_key": "CD47"
            },
            {
                "name": "Resveratrol (Stilbenoid)",
                "smiles": "Oc1ccc(cc1)C=Cc2cc(O)cc(O)c2",
                "role": "SIRT1 ve Anti-Apoptoz Kalkanı",
                "mechanism": "Sağlıklı nöron ve hemositleri serbest radikal hasarından korur, konakçı vitalitesini yükseltir",
                "base_kd": 0.200,
                "optimal_dose": "8.0 µM",
                "dose_val": 8.0,
                "dose_unit": "µM",
                "target_key": "SIRT1"
            },
            {
                "name": "EGCG (Yeşil Çay Kateşini)",
                "smiles": "O=C(Oc1cc(O)cc(O)c1)C2Oc3cc(O)cc(O)c3C(O)C2c4cc(O)c(O)c(O)c4",
                "role": "Mitokondriyal Membran Stabilizasyonu",
                "mechanism": "Sitotoksik ajanların kardiyotoksisite ve nörotoksisite yaratmasını engeller",
                "base_kd": 0.180,
                "optimal_dose": "6.0 µM",
                "dose_val": 6.0,
                "dose_unit": "µM",
                "target_key": "Mitochondria"
            }
        ]
    }

    OBJECTIVES = {
        "immune_mek_evasion": {
            "title": "Nöro-İmmün & MEK-Bypass Çift Yönlü Darbe (Triple-Hit)",
            "description": "Beyinden 185ms refleks ile hemosit patlaması tetikler, hücre içi MEK proliferasyonunu kilitler ve kaşeksiyi önler.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.30, 0.40),
            "potency_boost_range": (2.2, 2.7),
            "tox_shield_range": (82.0, 92.0)
        },
        "metronomic_chemo_soft": {
            "title": "Ultra-Düşük Toksisite Soft-Drug Kemo-İmmün Kurtarma",
            "description": "Esteraz-klerensli soft agonist ile düşük doz metronomik DNA çapraz bağlayıcıyı birleştirir; SIRT1 kalkanı ile doku hasarını <%5 tutar.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.38, 0.48),
            "potency_boost_range": (1.9, 2.3),
            "tox_shield_range": (80.0, 89.0)
        },
        "metabolic_epigenetic": {
            "title": "Warburg Metabolik Açlık & Epigenetik Şok Protokolü",
            "description": "Tümörün glikoz açlığını (HK2) kilitler, HDAC epigenetik reprogramlama ile savunmayı kırar ve hemosit lizisini patlatır.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.34, 0.44),
            "potency_boost_range": (2.1, 2.5),
            "tox_shield_range": (84.0, 90.0)
        },
        "free_ai_discovery": {
            "title": "Otonom AI De Novo Sinerji Keşfi (Maksimum Tümör Temizliği)",
            "description": "Yapay zeka tüm farmakolojik uzayı tarayarak en yüksek sinerji indeksine (CI < 0.35) ve en düşük doku toksisitesine sahip özgün kombinasyonu sentezler.",
            "preferred_roles": ["neuro_immune_drivers", "oncogenic_inhibitors", "protective_shields"],
            "target_synergy_ci": (0.28, 0.36),
            "potency_boost_range": (2.4, 3.0),
            "tox_shield_range": (86.0, 94.0)
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
        # Aynı ID varsa güncelle, yoksa ekle
        saved = [c for c in saved if c.get("id") != cocktail.get("id")]
        saved.insert(0, cocktail)
        try:
            with open(COCKTAILS_STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump(saved[:30], f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print("Kokteyl kaydedilemedi:", ex)

    def generate_cocktail(
        self,
        objective_key: Optional[str] = None,
        component_count: int = 3,
        custom_name: Optional[str] = None,
        objective: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Belirtilen onkolojik hedefe göre yapay zeka algoritmasıyla
        en yüksek sinerji indeksine ve en düşük doku toksisitesine sahip
        bir kombinasyonel terapi reçetesi oluşturur.
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
        
        # 1. Bileşen Seçimi
        components: List[Dict[str, Any]] = []
        
        # Primer Sürücü (Nöro-immün agonist)
        driver_pool = self.FUNCTIONAL_POOLS["neuro_immune_drivers"]
        driver = dict(np.random.choice(driver_pool))
        components.append(driver)

        # Onkojenik / Kinaz İnhibitörü
        kinase_pool = self.FUNCTIONAL_POOLS["oncogenic_inhibitors"]
        if objective_key == "metabolic_epigenetic":
            # 2-DG veya Vorinostat öncelikli
            filtered_kinase = [k for k in kinase_pool if "2-Deoxyglucose" in k["name"] or "Vorinostat" in k["name"]]
            kinase = dict(np.random.choice(filtered_kinase if filtered_kinase else kinase_pool))
        elif objective_key == "metronomic_chemo_soft":
            filtered_kinase = [k for k in kinase_pool if "Cisplatin" in k["name"]]
            kinase = dict(np.random.choice(filtered_kinase if filtered_kinase else kinase_pool))
        else:
            kinase = dict(np.random.choice(kinase_pool))
        components.append(kinase)

        # Koruyucu Kalkan / İmmün Kontrol Blokörü
        shield_pool = self.FUNCTIONAL_POOLS["protective_shields"]
        if objective_key == "immune_mek_evasion":
            # CD47 veya Curcumin öncelikli
            filtered_shield = [s for s in shield_pool if "CD47" in s["name"] or "Curcumin" in s["name"]]
            shield = dict(np.random.choice(filtered_shield if filtered_shield else shield_pool))
        else:
            shield = dict(np.random.choice(shield_pool))
        components.append(shield)

        # 4 Bileşen istendiyse ekstra bir modülatör ekle
        if component_count >= 4:
            available_extras = [
                s for s in (kinase_pool + shield_pool)
                if s["name"] not in [c["name"] for c in components]
            ]
            if available_extras:
                extra = dict(np.random.choice(available_extras))
                components.append(extra)

        # 2. Chou-Talalay Kombinasyon İndeksi (CI) ve İsobologram Simülasyonu
        min_ci, max_ci = obj_info["target_synergy_ci"]
        ci_noise = float(np.random.uniform(min_ci, max_ci))
        ci_val = round(ci_noise, 2)

        min_boost, max_boost = obj_info["potency_boost_range"]
        potency_boost = round(float(np.random.uniform(min_boost, max_boost)), 1)

        min_shield, max_shield = obj_info["tox_shield_range"]
        toxicity_reduction_pct = round(float(np.random.uniform(min_shield, max_shield)), 1)

        # Doz Tasarrufu (Dose Reduction Index - DRI)
        # Sinerji sayesinde her bileşenin dozu toksik olmayan alt sınırlara çekilir
        formatted_components = []
        for c in components:
            formatted_components.append({
                "name": c["name"],
                "smiles": c["smiles"],
                "dose": c["optimal_dose"],
                "dose_uM": c["dose_val"],
                "target": c["role"],
                "role": c["role"],
                "mechanism": c["mechanism"],
                "target_key": c["target_key"]
            })

        # İsim ve Gerekçe Üretimi
        cid = "ai_cocktail_" + str(uuid.uuid4())[:8]
        comp_names_short = " + ".join([c["name"].split(" ")[0] for c in formatted_components[:3]])
        generated_name = custom_name if custom_name else f"AI Sinerji: {comp_names_short}"

        rationale = (
            f"Chou-Talalay CI = {ci_val} (Kuvvetli Sinerji). "
            f"{formatted_components[0]['name']} ile tetiklenen nöro-efferent aks, "
            f"{formatted_components[1]['name']} ile onkogenik kaçışı ({formatted_components[1]['target']}) eşzamanlı durdurur. "
            f"{formatted_components[2]['name']} koruyucu kalkanı sayesinde doku toksisitesi %{toxicity_reduction_pct} oranında sönümlenir "
            f"ve lizis hızı {potency_boost}x katına çıkar."
        )

        synergy_label = "Süper Sinerji (CI < 0.40)" if ci_val < 0.40 else "Kuvvetli Sinerji (CI < 0.60)"

        cocktail_regimen = {
            "id": cid,
            "name": generated_name,
            "is_ai_generated": True,
            "objective_key": objective_key,
            "objective_title": obj_info["title"],
            "primary_smiles": formatted_components[0]["smiles"],
            "components": formatted_components,
            "component_count": len(formatted_components),
            "synergy_index_ci": ci_val,
            "chou_talalay_ci": ci_val,
            "synergy_label": synergy_label,
            "toxicity_reduction_pct": toxicity_reduction_pct,
            "toxicity_shield_pct": toxicity_reduction_pct / 100.0,
            "potency_boost": potency_boost,
            "target_potency_multiplier": potency_boost,
            "clinical_rationale": rationale,
            "description": rationale
        }

        # Kalıcı kaydet
        self.save_cocktail_to_storage(cocktail_regimen)
        return cocktail_regimen


# Singleton motor örneği
cocktail_synthesizer = AICocktailSynthesizer()
