import sys
sys.stdout.reconfigure(encoding='utf-8')
from platform_engine import DrosophilaInSilicoPlatform

print("==================================================================")
print("  TESTING ONCOLOGICAL TREATMENT MODALITIES & RESCUE PROTOCOLS")
print("==================================================================")

p = DrosophilaInSilicoPlatform()

# 1. Test Stereotactic Radiotherapy Pulse
print("\n[MODALITY 1] Stereotaktik Radyoterapi (SABR / IMRT) Darbesi:")
initial_cancer = len([c for c in p.cancer_cells if c.state.value not in ["apoptotic", "lysed"]])
print(f"  -> Işınlama öncesi canlı kanser: {initial_cancer}")
rad_res = p.apply_radiation_pulse(8.0)
print(f"  -> 8.0 Gy Radyasyon uygulandı! Hasar gören hücre: {rad_res['damaged_cells']}, Yok edilen: {rad_res['destroyed_cells']}")
assert rad_res["status"] == "radiation_applied", "Radyoterapi başarısız!"

# 2. Test Cytotoxic Chemotherapy
print("\n[MODALITY 2] Sitotoksik Kemoterapi (Sisplatin + Paklitaksel):")
chemo_mod = p.set_treatment_modality("cytotoxic_chemotherapy")
print(f"  -> Rejim Aktif: {chemo_mod['name']} ({chemo_mod['category']})")
for _ in range(30):
    s = p.step(1.0)
print(f"  -> 30s Kemoterapi sonrası kanser: {s['cancer_cells']}, Toksisite: %{s['toxicity_pct']:.1f}")
assert s["active_modality"] == "cytotoxic_chemotherapy", "Kemoterapi rejimi aktifleşmedi!"

# 3. Test Checkpoint Immunotherapy Anti-CD47
print("\n[MODALITY 3] Kontrol Noktası İmmünoterapisi (Anti-CD47):")
cd47_mod = p.set_treatment_modality("immunotherapy_cd47")
print(f"  -> Rejim Aktif: {cd47_mod['name']}")
for _ in range(30):
    s = p.step(1.0)
print(f"  -> 30s Anti-CD47 sonrası kanser: {s['cancer_cells']} (Dirençli: {s['resistant_cancer_cells']}), Toksisite: %{s['toxicity_pct']:.1f}")
assert s["active_modality"] == "immunotherapy_cd47", "Anti-CD47 rejimi aktifleşmedi!"

# 4. Test Metabolic Starvation (2-DG)
print("\n[MODALITY 4] Metabolik Warburg Açlık Terapisi (2-DG):")
metab_mod = p.set_treatment_modality("metabolic_starvation")
print(f"  -> Rejim Aktif: {metab_mod['name']}")
initial_c = s["cancer_cells"]
for _ in range(30):
    s = p.step(1.0)
print(f"  -> 30s 2-DG sonrası kanser: {s['cancer_cells']} (Başlangıç: {initial_c}), Mitoz blokajı başarılı.")
assert s["cancer_cells"] <= initial_c, "Metabolik açlık tümör büyümesini durduramadı!"

# 5. Test Metronomic Rescue
print("\n[MODALITY 5] Metronomik Multimodal Kurtarma Protokolü (AI Şampiyon):")
metro_mod = p.set_treatment_modality("metronomic_rescue")
print(f"  -> Rejim Aktif: {metro_mod['name']}")
for _ in range(50):
    s = p.step(1.0)
print(f"  -> 50s Metronomik rejim sonrası kanser: {s['cancer_cells']}, Toksisite: %{s['toxicity_pct']:.1f}, Vitalite: %{s['host_vitality_pct']:.1f}")
assert s["host_alive"] is True and s["toxicity_pct"] < 25.0, "Metronomik rejim başarısız!"

print("\n>>> TÜM ONKOLOJİK TEDAVİ VE KURTARMA PROTOKOLLERİ 100% BAŞARIYLA GEÇTİ! <<<")
