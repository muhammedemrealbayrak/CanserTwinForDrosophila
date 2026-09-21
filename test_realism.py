import sys
sys.stdout.reconfigure(encoding='utf-8')
from platform_engine import DrosophilaInSilicoPlatform

print('=== TEST 5: Optimized De Novo Champion ===')
p = DrosophilaInSilicoPlatform(active_compound_smiles_or_name='CC1=NC=C(C=C1)CCN(C)C(=O)CF')
p.drug_dose_uM = 2.5
for i in range(160):
    s = p.step(1.0)
    if i in (10, 50, 100, 159):
        print(f"Step {i+1}s: cancer={s['cancer_cells']} (res={s['resistant_cancer_cells']}, sens={s['sensitive_cancer_cells']}), tox={s['toxicity_pct']:.1f}%, vitality={s['host_vitality_pct']}%, status={s['clinical_status_text']}")
