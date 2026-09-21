import sys
sys.stdout.reconfigure(encoding='utf-8')
from platform_engine import DrosophilaInSilicoPlatform

print("==================================================================")
print("  DROSOPHILA DIGITAL TWIN: BIOLOGICAL REALISM & FAILURE MODES")
print("==================================================================")

# 1. Aşırı Doz & Öldürücü Toksisite (Host Lethality)
print("\n[TEST 1] Yüksek Doz Nicotine (5.0 uM) -> Toksik Şok / Konakçı Ölümü:")
p1 = DrosophilaInSilicoPlatform(active_compound_smiles_or_name='Nicotine')
p1.drug_dose_uM = 5.0
for i in range(100):
    s1 = p1.step(1.0)
    if not s1['host_alive']:
        print(f"  -> {i+1}. saniyede konakçı canlı EX oldu! Toksisite: %{s1['toxicity_pct']:.1f}, Vitalite: %{s1['host_vitality_pct']:.1f}")
        print(f"  -> Durum: {s1['clinical_status_text']} ({s1['clinical_outcome']})")
        break
assert s1['clinical_outcome'] == "HOST_LETHALITY_OVERDOSE", "Test 1 başarısız: Konakçı ölmedi!"

# 2. Düşük Doz / Zayıf İlaç (Tumor Progression / Escape)
print("\n[TEST 2] Yetersiz Doz (0.1 uM Dopamine) -> Tümör İstilası / Tedavi Başarısızlığı:")
p2 = DrosophilaInSilicoPlatform(active_compound_smiles_or_name='Dopamine')
p2.drug_dose_uM = 0.1
for i in range(160):
    s2 = p2.step(1.0)
print(f"  -> 160 saniye sonra tümör: {s2['cancer_cells']} hücreye fırladı (Başlangıç: 150).")
print(f"  -> Durum: {s2['clinical_status_text']} ({s2['clinical_outcome']})")
assert s2['clinical_outcome'] == "TUMOR_PROGRESSION_ESCAPE", "Test 2 başarısız: Tümör çoğalamadı!"

# 3. Monoterapi Dirençli Klon Patlaması (Tumor Relapse)
print("\n[TEST 3] Monoterapi (1.0 uM) -> Dirençli Klon Seçilimi & Relaps:")
p3 = DrosophilaInSilicoPlatform(active_compound_smiles_or_name='Nicotine')
p3.drug_dose_uM = 1.0
for i in range(180):
    s3 = p3.step(1.0)
print(f"  -> 180 saniye sonra Dirençli Klon Sayısı: {s3['resistant_cancer_cells']} / Toplam: {s3['cancer_cells']} (%{s3['resistant_cancer_cells']/s3['cancer_cells']*100:.1f})")
print(f"  -> Durum: {s3['clinical_status_text']} ({s3['clinical_outcome']})")
assert s3['resistant_cancer_cells'] > 70, "Test 3 başarısız: Dirençli klonlar seçilemedi!"

# 4. Sinerjik Kokteyl (Immuno-MEK Synergy)
print("\n[TEST 4] Sinerjik Kokteyl (Immuno-MEK Synergy) -> Düşük Toksisite & Doku Koruması:")
p4 = DrosophilaInSilicoPlatform()
p4.set_cocktail('immuno_mek_synergy')
for i in range(150):
    s4 = p4.step(1.0)
print(f"  -> 150 saniye sonra Toksisite: %{s4['toxicity_pct']:.1f}, Vitalite: %{s4['host_vitality_pct']:.1f}")
print(f"  -> Durum: {s4['clinical_status_text']} ({s4['clinical_outcome']})")
assert s4['host_alive'] is True and s4['toxicity_pct'] < 15.0, "Test 4 başarısız: Kokteyl güvenli değil!"

print("\n>>> TÜM BİYOLOJİK GERÇEKÇİLİK VE BAŞARISIZLIK MODLARI BAŞARIYLA DOĞRULANDI! <<<")
