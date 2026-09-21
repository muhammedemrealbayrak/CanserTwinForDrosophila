"""
Drosophila In Silico Neuro-Immune Digital Twin
Test & De Novo Karşılaştırma Betiği: run_platform.py
======================================================
Yazar: Baş Sistem Biyoloğu & Yapay Zeka Baş Mimarı
Açıklama:
    Madde A (Nikotinik Hızlı Ajan), Madde B (Polifenol Yavaş Ajan) ve
    Yapay Zekanın sıfırdan türettiği De Novo Sentetik İlaç Adaylarını
    zamana karşı simülasyonda yarıştırır ve detaylı telemetri tablosu üretir.
"""

import sys
import os
import time

# Python yolunu ayarla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platform_engine import DrosophilaInSilicoPlatform
from denovo_ai.molecule_generator import DeNovoMoleculeGenerator


def print_banner(title: str):
    print("=" * 85)
    print(f"  {title.upper()}")
    print("=" * 85)


def format_table_row(cols, widths):
    row_str = " | ".join(f"{str(col):<{w}}" for col, w in zip(cols, widths))
    return f"| {row_str} |"


from data_manager import DrosophilaDataManager

data_mgr = DrosophilaDataManager()


def benchmark_single_candidate(name: str, smiles_or_key: str, sim_duration_s: float = 1200.0):
    engine = DrosophilaInSilicoPlatform(
        active_compound_smiles_or_name=smiles_or_key,
        initial_tumor_burden=160,
        domain_size_um=500.0,
        grid_resolution=16
    )
    result = engine.run_benchmark(duration_seconds=sim_duration_s)
    final_snap = engine.history[-1]
    consumed = engine.fuel_engine.cumulative_spent

    # Veritabanına kalıcı olarak kaydet
    data_mgr.record_simulation_run(result)

    return {
        "result": result,
        "final_snap": final_snap,
        "consumed": consumed,
        "profile": engine.active_drug
    }


def main():
    print_banner("IN SILICO DROSOPHILA DİJİTAL İKİZ - ZAMANA KARŞI İLAÇ YARIŞI")
    print("Entegrasyon: PubChem (SMILES) + FlyWire (3D Konektom) + Fly Cell Atlas (scRNA-seq)")
    print("Metabolizma: 4-Aşamalı Kemik İliği/Lenf Bezi Yakıt Hiyerarşisi\n")

    # Aday Listesi: Referans Maddeler + De Novo AI Adayları
    candidates = [
        ("Madde A (Hızlı/Toksik)", "Nicotine"),
        ("Madde B (Yavaş/Düşük Toksik)", "Curcumin"),
        ("De Novo Gen 1 (Sentetik Hibrit)", "CN1CCC[C@H]1c2cccnc2F"),
        ("De Novo Şampiyon (Optimize Formül)", "CC1=NC=C(C=C1)CCN(C)C(=O)CF")
    ]

    benchmark_records = []
    print("Simülasyon motoru başlatılıyor, adaylar zamana karşı test ediliyor...\n")

    for name, smiles in candidates:
        t0 = time.time()
        data = benchmark_single_candidate(name, smiles, sim_duration_s=1200.0) # 20 dakika simülasyon
        elapsed_compute = time.time() - t0
        benchmark_records.append((name, data))
        print(f"  [OK] {name:<35} -> Simüle edildi ({elapsed_compute:.1f}s)")

    # 1. PERFORMANS VE ZAMAN TABLOSU
    print("\n" + "=" * 85)
    print("  MOLEKÜLER PERFORMANS & ZAMAN SKOR TABLOSU (20 Dakikalık Eşzamanlı Simülasyon)")
    print("=" * 85)

    headers = [
        "Aday Molekül Adı", "Tetikleme(s)", "Hemosit", "Tümör(%)",
        "Toksisite", "LogP", "Kd (uM)", "Skor (Fitness)"
    ]
    widths = [32, 12, 8, 9, 10, 6, 8, 14]
    print(format_table_row(headers, widths))
    print("|" + "|".join(["-" * (w + 2) for w in widths]) + "|")

    for name, data in benchmark_records:
        res = data["result"]
        prof = data["profile"]
        row = [
            name[:32],
            f"{res.time_to_trigger_s:5.1f} s",
            res.total_hemocytes_produced,
            f"%{res.tumor_clearance_pct:4.1f}",
            f"%{res.systemic_toxicity_pct:4.1f}",
            f"{prof.logP:4.2f}",
            f"{prof.kd_micromolar:5.3f}",
            f"{res.multiobjective_fitness_score:5.1f} / 100"
        ]
        print(format_table_row(row, widths))

    # 2. ŞAMPİYON MOLEKÜL VE BİYOKİMYASAL TÜKETİM DETAYI
    best_candidate_name, best_data = max(benchmark_records, key=lambda x: x[1]["result"].multiobjective_fitness_score)
    best_res = best_data["result"]
    best_cons = best_data["consumed"]
    best_prof = best_data["profile"]

    print("\n" + "=" * 85)
    print(f"  [SAMPIYON MOLEKUL]: {best_candidate_name.upper()}")
    print("=" * 85)
    print(f"  * Moleküler Formül (SMILES) : {best_prof.smiles}")
    print(f"  * Molekül Ağırlığı          : {best_prof.molecular_weight:.2f} g/mol")
    print(f"  * Doku Penetrasyonu (LogP)  : {best_prof.logP:.2f} (Membran geçirgenliği optimize)")
    print(f"  * Reseptör Afinitesi (Kd)   : {best_prof.kd_micromolar:.4f} uM (Yüksek Duyarlılık)")
    print(f"  * Bağışıklığı Tetikleme Hızı: Sadece {best_res.time_to_trigger_s:.1f} saniye içinde eferent akı başlattı!")
    print(f"  * Üretilen Savunma Hücresi  : {best_res.total_hemocytes_produced} Hemosit (Plazmatosit + Lamellosit)")
    print(f"  * Sistemik Güvenlik / Doku  : Sadece %{best_res.systemic_toxicity_pct:.1f} toksisite (Minimum Yan Etki)")
    print("-" * 85)
    print("  [KEMİK İLİĞİ / LENF BEZİ 4-AŞAMALI YAKIT HARCAMA PROFİLİ]")
    print(f"    - Aşama 1 (Glukoz / ATP)          : {best_cons['glucose']:.4f} mM / {best_cons['atp']:.4f} mM harcandı")
    print(f"    - Aşama 2 (Glutamin / BCAA)       : {best_cons['glutamine']:.4f} mM / {best_cons['bcaa']:.4f} mM harcandı")
    print(f"    - Aşama 3 (Nükleotidler & Zn/Fe)  : {best_cons['nucleotides']:.4f} mM harcandı")
    print(f"    - Aşama 4 (Lipitler & Yağ Asidi)  : {best_cons['lipids']:.4f} mM (Dış zar inşası tamamlandı)")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
