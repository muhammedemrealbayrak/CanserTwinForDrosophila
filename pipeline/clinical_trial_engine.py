"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: pipeline/clinical_trial_engine.py
======================================================
Yazar: Sanal Klinik Deney & Biyoistatistik Modelleme Ekibi
Açıklama:
    Monte Carlo tabanlı sanal hasta (Drosophila ikizleri) kohort simülasyonu,
    4 randomize tedavi kolu, Kaplan-Meier basamaklı sağkalım analizi (OS/PFS),
    Log-Rank (Mantel-Cox) hipotez testi ve Hazard Ratio (HR) hesaplama motoru.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class VirtualClinicalTrialEngine:
    """
    Onkolojik tedavi protokollerini heterojen sanal Drosophila kohortlarında
    (In Silico Clinical Trials) test eden ve Kaplan-Meier sağkalım eğrileri
    üreten biyomatematiksel klinik simülasyon motoru.
    """

    TRIAL_ARMS = {
        "control_placebo": {
            "name": "Kol A: Kontrol (Tedavisiz Plasebo)",
            "short_name": "Kontrol (Plasebo)",
            "category": "Doğal Progresyon",
            "color": "#ef4444",  # Kırmızı
            "description": "Tedavi uygulanmayan, tümörün doğal seyrine ve kaşektik doku hasarına terk edilmiş kontrol grubu.",
            "target_pathway": "Müdahalesiz Tümör Mikroçevresi"
        },
        "chemo_standard": {
            "name": "Kol B: Standart Sitotoksik Kemoterapi (Sisplatin)",
            "short_name": "Kemoterapi (Sisplatin)",
            "category": "Geleneksel Kemoterapi",
            "color": "#ffb703",  # Kehribar Sarısı
            "description": "DNA çapraz bağlayıcı standart sitotoksik rejim. Duyarlı klonları baskılar ancak nüks ve kümülatif doku toksisitesi yaratır.",
            "target_pathway": "DNA Adducts / Sistemik Sitotoksisite"
        },
        "targeted_ai": {
            "name": "Kol C: Hedefe Yönelik AI Ajanı (DeNovo_Champion)",
            "short_name": "DeNovo_Champion (AI)",
            "category": "Hedefe Yönelik Monoterapi",
            "color": "#00f0ff",  # Siber Camgöbeği
            "description": "Kenyon hücresi nAChR kolinerjik refleks ateşlemesi ile hemosit göçünü tetikleyen ve MEK kinazını hedefleyen De Novo molekül.",
            "target_pathway": "nAChRalpha7 / MAPK İnhibisyonu"
        },
        "multimodal_rescue": {
            "name": "Kol D: Multimodal Sinerji Kurtarma Rejimi",
            "short_name": "Multimodal Kurtarma (AI Kokteyl)",
            "category": "Kombinasyonel Sinerji",
            "color": "#00ff9d",  # Zümrüt Yeşili
            "description": "Düşük doz metronomik kemo + DeNovo_Champion + Anti-CD47 fagositoz kalkan kırıcı + Kurkumin kaşeksi kalkanı.",
            "target_pathway": "Kombinasyonel Sinerji (Chou-Talalay CI=0.29)"
        }
    }

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed

    def get_trial_protocols(self) -> Dict[str, Any]:
        """Klinik deney kollarını ve standart parametreleri döndürür."""
        return {
            "arms": self.TRIAL_ARMS,
            "default_cohort_sizes": [50, 100, 250, 500],
            "default_follow_up_days": 60,
            "reference_standard": "RECIST 1.1 / CTCAE v5.0 Preclinical Drosophila Adaptation",
            "primary_endpoint": "Overall Survival (Genel Sağkalım - OS)",
            "secondary_endpoints": ["Median Survival Time (Medyan Sağkalım)", "Hazard Ratio (HR)", "Log-Rank p-value", "Toxicity-Related Mortality"]
        }

    def _generate_virtual_patient(self, twin_id: int, arm_key: str, rng: np.random.RandomState) -> Dict[str, Any]:
        """Bireysel genetik varyasyonlara ve mikroçevreye sahip tekil bir sanal Drosophila ikizi üretir."""
        # Başlangıç tümör yükü: ortalama 150 hücre, std 25
        init_tumor = int(np.clip(rng.normal(150, 25), 80, 260))
        # Dirençli klon oranı: %15 ile %35 arası
        res_fraction = float(np.clip(rng.uniform(0.15, 0.35), 0.10, 0.45))
        # Konakçı bağışıklık rezervi (hemosit üretim kapasitesi): 0.6x - 1.5x
        immune_cap = float(np.clip(rng.normal(1.0, 0.20), 0.5, 1.6))
        # İlaç metabolik klerensi (karaciğer/yağ cismi aktivitesi): 0.6x - 1.5x
        clearance = float(np.clip(rng.normal(1.0, 0.18), 0.6, 1.5))
        # Konakçı bazal vitalite direnci (kaşeksi toleransı): 0.7x - 1.4x
        resilience = float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.4))

        return {
            "id": twin_id,
            "arm": arm_key,
            "initial_tumor": init_tumor,
            "resistant_fraction": res_fraction,
            "immune_capacity": immune_cap,
            "drug_clearance": clearance,
            "vitality_resilience": resilience
        }

    def _simulate_patient_trajectory(
        self,
        patient: Dict[str, Any],
        time_horizon_days: int,
        rng: np.random.RandomState
    ) -> Tuple[float, int, str]:
        """
        Sanal hastanın 60 günlük yaşam eğrisini simüle eder.
        Dönüş: (time_to_event_days, event_occurred_0_or_1, cause_of_death)
        """
        arm = patient["arm"]
        tumor = float(patient["initial_tumor"])
        res_frac = patient["resistant_fraction"]
        immune = patient["immune_capacity"]
        clearance = patient["drug_clearance"]
        resilience = patient["vitality_resilience"]

        toxicity = 0.0
        max_days = float(time_horizon_days)

        # Günlük simülasyon adımı (dt = 0.5 gün)
        t = 0.0
        dt = 0.5

        while t < max_days:
            # 1. Koldan bağımsız tümör büyüme potansiyeli (Gompertzian büyüme)
            # Taşıma kapasitesi K = 1200 hücre
            k_cap = 1200.0
            base_growth = 0.09 * (1.0 - (tumor / k_cap))

            # 2. Tedavi Koluna Özgü Hücresel ve Farmakolojik Etkiler
            if arm == "control_placebo":
                # Tedavisiz: Tümör kontrolsüz prolifere olur
                net_growth = base_growth * rng.uniform(0.9, 1.15)
                tumor += tumor * net_growth * dt
                # Doğal hemosit yanıtı zayıftır
                tumor -= min(tumor * 0.4, 3.5 * immune * dt)
                # Kaşeksi toksini tümörle paralel birikir
                toxicity += (tumor / 150.0) * 0.012 * dt

            elif arm == "chemo_standard":
                # Sisplatin: Duyarlı hücreleri hızla öldürür, dirençli klonlar büyür
                sen_fraction = max(0.05, 1.0 - res_frac * (1.0 + t / 25.0))
                kill_rate = 0.16 * sen_fraction
                tumor += (tumor * base_growth - tumor * kill_rate) * dt
                # Hemosit baskılanması
                tumor -= min(tumor * 0.3, 1.8 * immune * dt)
                # Kemoterapötik birikimli doku toksisitesi
                tox_accum = (0.018 / clearance) * dt
                toxicity += tox_accum

            elif arm == "targeted_ai":
                # DeNovo_Champion: nAChR kolinerjik refleks + MEK inhibisyonu
                # Hemosit göçünü 2.2 katına çıkarır, proliferasyonu kilitler
                kill_rate = 0.14 * rng.uniform(0.95, 1.10)
                hemo_kill = 8.5 * immune * dt
                tumor += (tumor * (base_growth * 0.35) - tumor * kill_rate) * dt
                tumor -= min(tumor * 0.6, hemo_kill)
                # Çok düşük toksisite
                toxicity += 0.003 * dt

            elif arm == "multimodal_rescue":
                # Metronomik Kurtarma Rejimi: Süper sinerji
                # Anti-CD47 fagositozu artırır, Kurkumin kaşeksiyi bloke eder
                kill_rate = 0.22 * rng.uniform(1.0, 1.15)
                hemo_kill = 12.0 * immune * dt
                tumor += (tumor * (base_growth * 0.15) - tumor * kill_rate) * dt
                tumor -= min(tumor * 0.8, hemo_kill)
                # Kurkumin kalkanı sayesinde neredeyse sıfır net toksisite
                toxicity += 0.0015 * dt

            tumor = max(0.0, tumor)

            # 3. Ölüm Eşikleri Değerlendirmesi
            # A) Aşırı Tümör Yükü Ölümü (Organ işgal / Rüptür: > 450 hücre)
            if tumor >= (450.0 * resilience):
                return (round(t, 2), 1, "Tümör Yükü (Kanser)")

            # B) Aşırı Toksisite Ölümü (Toksisite >= 45% / resilience)
            lethal_tox_limit = 0.45 * resilience
            if toxicity >= lethal_tox_limit:
                return (round(t, 2), 1, "Doku Toksisitesi (İlaç Yan Etkisi)")

            # C) Tümör Kaşeksisi ve Metabolik Çöküş
            cachexia_risk = (toxicity * 0.7 + (tumor / 350.0) * 0.3)
            if cachexia_risk > (0.65 * resilience):
                if rng.rand() < (0.04 * dt):
                    return (round(t, 2), 1, "Tümör Kaşeksisi (Organizma Çöküşü)")

            t += dt

        # Süre sonuna kadar yaşayan (Sansürlenen / Censored) hasta
        return (float(time_horizon_days), 0, "Sağ Kurtuldu (Sensörlü)")

    def run_trial(
        self,
        cohort_size_per_arm: int = 100,
        selected_arms: Optional[List[str]] = None,
        time_horizon_days: int = 60
    ) -> Dict[str, Any]:
        """
        Tüm tedavi kolları için randomize sanal klinik deneyi icra eder.
        Kaplan-Meier sağkalım eğrilerini, Log-Rank karşılaştırmalarını ve HR değerlerini hesaplar.
        """
        if selected_arms is None:
            selected_arms = list(self.TRIAL_ARMS.keys())

        rng = np.random.RandomState(self.random_seed)
        trial_results_by_arm: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Kohort Simülasyonu
        global_patient_id = 1
        for arm_key in selected_arms:
            patients = []
            for _ in range(cohort_size_per_arm):
                p = self._generate_virtual_patient(global_patient_id, arm_key, rng)
                time_to_event, event_occurred, cause = self._simulate_patient_trajectory(
                    p, time_horizon_days, rng
                )
                patients.append({
                    **p,
                    "time_days": time_to_event,
                    "event": event_occurred,
                    "cause": cause
                })
                global_patient_id += 1
            trial_results_by_arm[arm_key] = patients

        # 2. Kaplan-Meier Sağkalım Eğrisi Hesaplama (Her Kol İçin)
        km_data_by_arm = {}
        for arm_key, patients in trial_results_by_arm.items():
            km_curve, median_os, os_30, os_60 = self._calculate_kaplan_meier(
                patients, time_horizon_days
            )
            cause_counts = {}
            for p in patients:
                cause_counts[p["cause"]] = cause_counts.get(p["cause"], 0) + 1

            km_data_by_arm[arm_key] = {
                "arm_info": self.TRIAL_ARMS[arm_key],
                "cohort_size": len(patients),
                "km_curve": km_curve,
                "median_os_days": median_os,
                "os_30_days_pct": round(os_30 * 100, 1),
                "os_60_days_pct": round(os_60 * 100, 1),
                "total_events": sum(p["event"] for p in patients),
                "total_censored": sum(1 for p in patients if p["event"] == 0),
                "causes_of_death": cause_counts
            }

        # 3. İstatistiksel Karşılaştırmalar & Log-Rank Testi (Kontrol Grubuna Karşı)
        control_arm_key = "control_placebo" if "control_placebo" in trial_results_by_arm else selected_arms[0]
        control_patients = trial_results_by_arm.get(control_arm_key, [])

        statistical_comparisons = []
        for arm_key in selected_arms:
            arm_data = km_data_by_arm[arm_key]
            med_str = f"{arm_data['median_os_days']} Gün" if arm_data["median_os_days"] is not None else f"NR (>{time_horizon_days} Gün)"
            arm_data["median_os_formatted"] = med_str

            if arm_key == control_arm_key:
                statistical_comparisons.append({
                    "arm_key": arm_key,
                    "arm_name": self.TRIAL_ARMS[arm_key]["short_name"],
                    "hazard_ratio": 1.0,
                    "hr_ci_95": [1.0, 1.0],
                    "logrank_p_value": 1.0,
                    "p_value_formatted": "Referans",
                    "clinical_significance": "Kontrol Referans Grubu",
                    "median_os": arm_data["median_os_days"],
                    "median_os_formatted": med_str,
                    "os_30": arm_data["os_30_days_pct"],
                    "os_60": arm_data["os_60_days_pct"]
                })
            else:
                treated_patients = trial_results_by_arm[arm_key]
                hr, hr_lower, hr_upper, p_val = self._logrank_test(
                    control_patients, treated_patients
                )

                # Klinik Anlamlılık ve FDA/EMA Tipi Onay Kararı
                if p_val < 0.001 and hr < 0.40:
                    decision = "🌟 Üstün Başarı (Breakthrough Therapy Onayı)"
                elif p_val < 0.05 and hr < 0.70:
                    decision = "✅ İstatistiksel Anlamlı Sağkalım Avantajı"
                elif p_val < 0.05 and hr < 0.90:
                    decision = "⚠️ Sınırlı / Marjinal Sağkalım Yararı"
                else:
                    decision = "❌ İstatistiksel Üstünlük Saptanamadı"

                statistical_comparisons.append({
                    "arm_key": arm_key,
                    "arm_name": self.TRIAL_ARMS[arm_key]["short_name"],
                    "hazard_ratio": round(hr, 2),
                    "hr_ci_95": [round(hr_lower, 2), round(hr_upper, 2)],
                    "logrank_p_value": p_val,
                    "p_value_formatted": f"< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}",
                    "clinical_significance": decision,
                    "median_os": arm_data["median_os_days"],
                    "median_os_formatted": med_str,
                    "os_30": arm_data["os_30_days_pct"],
                    "os_60": arm_data["os_60_days_pct"]
                })

        # 4. Risk Altındaki Hasta Sayıları Tablosu (Numbers at Risk)
        checkpoints = [0, 15, 30, 45, 60]
        numbers_at_risk = {}
        for arm_key in selected_arms:
            pts = trial_results_by_arm[arm_key]
            counts = []
            for day in checkpoints:
                # O günden sonra olay yaşayan veya o günde henüz canlı olanlar
                at_risk = sum(1 for p in pts if p["time_days"] >= day)
                counts.append(at_risk)
            numbers_at_risk[arm_key] = counts

        return {
            "status": "success",
            "trial_parameters": {
                "cohort_size_per_arm": cohort_size_per_arm,
                "total_patients": len(selected_arms) * cohort_size_per_arm,
                "time_horizon_days": time_horizon_days,
                "active_arms": [self.TRIAL_ARMS[k]["short_name"] for k in selected_arms]
            },
            "checkpoints_days": checkpoints,
            "numbers_at_risk": numbers_at_risk,
            "arms_data": km_data_by_arm,
            "statistical_comparisons": statistical_comparisons
        }

    def _calculate_kaplan_meier(
        self,
        patients: List[Dict[str, Any]],
        max_time_days: int
    ) -> Tuple[List[Dict[str, Any]], Optional[float], float, float]:
        """
        Kaplan-Meier basamak eğrisi S(t) = Prod(1 - di / ni) ve medyan sağkalımı hesaplar.
        """
        n_total = len(patients)
        if n_total == 0:
            return ([{"time": 0, "survival": 1.0, "censored": 0}], None, 0.0, 0.0)

        # Olay ve sansür zamanlarını topla
        times = sorted(list(set([p["time_days"] for p in patients])))
        if 0.0 not in times:
            times = [0.0] + times

        km_curve = [{"time": 0.0, "survival": 1.0, "at_risk": n_total, "events": 0, "censored": 0}]
        surv_prob = 1.0
        n_at_risk = n_total
        median_os: Optional[float] = None
        os_30 = 1.0
        os_60 = 1.0

        for t in sorted(times):
            if t == 0.0:
                continue

            # Bu zamanda ölenler (di) ve bu zamanda sansürlenenler (ci)
            deaths_at_t = sum(1 for p in patients if p["time_days"] == t and p["event"] == 1)
            cens_at_t = sum(1 for p in patients if p["time_days"] == t and p["event"] == 0)

            if n_at_risk > 0 and deaths_at_t > 0:
                surv_prob *= (1.0 - float(deaths_at_t) / float(n_at_risk))

            # Medyan sağkalım: Sağkalım ilk defa <= 0.50 olduğunda
            if median_os is None and surv_prob <= 0.50:
                median_os = round(float(t), 1)

            if t <= 30.0:
                os_30 = surv_prob
            if t <= 60.0:
                os_60 = surv_prob

            km_curve.append({
                "time": round(float(t), 1),
                "survival": round(float(surv_prob), 4),
                "at_risk": n_at_risk,
                "events": deaths_at_t,
                "censored": cens_at_t
            })

            n_at_risk -= (deaths_at_t + cens_at_t)

        return (km_curve, median_os, os_30, os_60)

    def _logrank_test(
        self,
        group1: List[Dict[str, Any]],  # Kontrol
        group2: List[Dict[str, Any]]   # Tedavi
    ) -> Tuple[float, float, float, float]:
        """
        İki grup arasında Mantel-Cox Log-Rank testi ve Hazard Ratio (HR) hesaplar.
        Dönüş: (Hazard_Ratio, HR_95_Lower, HR_95_Upper, p_value)
        """
        all_patients = [("ctrl", p) for p in group1] + [("treat", p) for p in group2]
        event_times = sorted(list(set(p["time_days"] for _, p in all_patients if p["event"] == 1)))

        o_treat_total = 0.0
        e_treat_total = 0.0
        v_total = 0.0

        o_ctrl_total = 0.0
        e_ctrl_total = 0.0

        for t in event_times:
            # t zamanında risk altındakiler
            n1 = sum(1 for grp, p in all_patients if grp == "ctrl" and p["time_days"] >= t)
            n2 = sum(1 for grp, p in all_patients if grp == "treat" and p["time_days"] >= t)
            n = n1 + n2

            # t zamanında ölenler
            d1 = sum(1 for grp, p in all_patients if grp == "ctrl" and p["time_days"] == t and p["event"] == 1)
            d2 = sum(1 for grp, p in all_patients if grp == "treat" and p["time_days"] == t and p["event"] == 1)
            d = d1 + d2

            if n <= 1 or d == 0:
                continue

            e1 = float(n1 * d) / float(n)
            e2 = float(n2 * d) / float(n)

            # Hypergeometric varyans
            v = (float(n1 * n2 * d) * (n - d)) / (float(n * n) * (n - 1))

            o_ctrl_total += d1
            e_ctrl_total += e1

            o_treat_total += d2
            e_treat_total += e2
            v_total += v

        # Log-Rank Chi-Square: chi2 = (O - E)^2 / V
        if v_total > 0:
            chi2 = ((o_treat_total - e_treat_total) ** 2) / v_total
            # 1 serbestlik derecesinde p-değeri (chi2 CDF yaklaşımı)
            # p = 1 - erf(sqrt(chi2 / 2))
            p_val = math.erfc(math.sqrt(chi2 / 2.0))
        else:
            chi2 = 0.0
            p_val = 1.0

        # Hazard Ratio: HR = (O2 / E2) / (O1 / E1)
        ratio_treat = (o_treat_total / e_treat_total) if e_treat_total > 0 else 1.0
        ratio_ctrl = (o_ctrl_total / e_ctrl_total) if e_ctrl_total > 0 else 1.0

        hr = ratio_treat / ratio_ctrl if ratio_ctrl > 0 else 1.0
        hr = float(np.clip(hr, 0.05, 5.0))

        # %95 Güven Aralığı: exp(ln(HR) +/- 1.96 / sqrt(V))
        se_ln_hr = (1.0 / math.sqrt(v_total)) if v_total > 0 else 0.3
        hr_lower = max(0.01, hr * math.exp(-1.96 * se_ln_hr))
        hr_upper = hr * math.exp(1.96 * se_ln_hr)

        return (hr, hr_lower, hr_upper, p_val)


# Global Singleton
clinical_trial_engine = VirtualClinicalTrialEngine()
