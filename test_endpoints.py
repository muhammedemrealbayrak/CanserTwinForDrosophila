import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8060"

def safe_str(s):
    return str(s).encode('ascii', 'replace').decode('ascii')

def test():
    print("Testing Drosophila Digital Twin Web Endpoints & AI Dose Engine...")

    # 1. GET /
    print("\n1. Testing GET / (index.html)...")
    res = urllib.request.urlopen(f"{BASE_URL}/")
    assert res.status == 200
    html = res.read().decode('utf-8')
    assert len(html) > 150000
    assert "multidrug-candidates-grid" in html
    assert "runMultiDrugPrediction" in html
    assert "calculateAIOptimalDoseForActive" in html
    assert "overflow-y: auto" in html
    print(f"   [OK] index.html served successfully ({len(html)} bytes)")

    # 2. GET /api/compounds
    print("\n2. Testing GET /api/compounds...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/compounds")
    assert res.status == 200
    compounds = json.loads(res.read().decode('utf-8'))
    assert len(compounds) >= 50
    print(f"   [OK] {len(compounds)} compounds loaded")

    # 3. GET /api/benchmarks
    print("\n3. Testing GET /api/benchmarks...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/benchmarks")
    assert res.status == 200
    benchmarks = json.loads(res.read().decode('utf-8'))
    assert len(benchmarks) >= 20
    top = benchmarks[0]
    print(f"   [OK] {len(benchmarks)} benchmark entries loaded (Top: {top['molecule_name']} with fitness {top['fitness_score']})")

    # 4. GET /api/molecule_svg
    print("\n4. Testing GET /api/molecule_svg...")
    test_smiles = urllib.request.quote("Cc1ccc(CCN(C)C(=O)CF)cn1")
    res = urllib.request.urlopen(f"{BASE_URL}/api/molecule_svg?smiles={test_smiles}")
    assert res.status == 200
    svg_data = res.read().decode('utf-8')
    assert "<svg" in svg_data and "</svg>" in svg_data
    print(f"   [OK] RDKit 2D SVG generated ({len(svg_data)} bytes)")

    # 5. POST /api/denovo/generate
    print("\n5. Testing POST /api/denovo/generate...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/denovo/generate",
        data=json.dumps({"parent": "Nicotine", "strategy": "balanced", "count": 4}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    denovo = json.loads(res.read().decode('utf-8'))
    assert denovo["status"] == "success"
    assert len(denovo["candidates"]) == 4
    c0 = denovo["candidates"][0]
    print(f"   [OK] De Novo candidate: {c0['candidate_name']} (Tox: {c0['toxicity_pct']}%, Ro5: {safe_str(c0['lipinski']['rule_label'])})")

    # 6. POST /api/predict_qsar
    print("\n6. Testing POST /api/predict_qsar...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/predict_qsar",
        data=json.dumps({"smiles": "CC1=NC=C(C=C1)CCN(C)C(=O)CF"}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    qsar = json.loads(res.read().decode('utf-8'))
    assert qsar["status"] == "success"
    assert "lipinski" in qsar
    print(f"   [OK] QSAR profile: MW={qsar['profile']['molecular_weight']}, Ro5={safe_str(qsar['lipinski']['rule_label'])}")

    # 7. POST /api/ai/predict_multidrug
    print("\n7. Testing POST /api/ai/predict_multidrug...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/ai/predict_multidrug",
        data=json.dumps({"pathway": "ras_mek", "count": 4}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    multidrug = json.loads(res.read().decode('utf-8'))
    assert multidrug["status"] == "success"
    assert len(multidrug["candidates"]) == 4
    for i, mc in enumerate(multidrug["candidates"]):
        print(f"   Candidate #{i+1}: {mc['candidate_name']}")
        print(f"      Optimal Dose: {mc['optimal_dose_uM']} uM | Clr: %{mc['predicted_tumor_clearance']} | Tox: %{mc['predicted_tissue_toxicity']} | TI: {mc['therapeutic_index']}x")
        assert mc["optimal_dose_uM"] > 0.0
        assert mc["predicted_tumor_clearance"] > 70.0
    print("   [OK] Multi-drug AI candidates and autonomous doses generated successfully!")

    # 8. POST /api/ai/optimize_dose
    print("\n8. Testing POST /api/ai/optimize_dose...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/ai/optimize_dose",
        data=json.dumps({"molecule": "Trametinib"}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    dose_data = json.loads(res.read().decode('utf-8'))
    assert dose_data["status"] == "success"
    assert "optimal_dose_uM" in dose_data
    assert "dose_curve" in dose_data
    print(f"   [OK] Trametinib Optimal Dose: {dose_data['optimal_dose_uM']} uM (TI: {dose_data['therapeutic_index']}x, ED50: {dose_data['ed50_uM']} uM)")

    print("\n=======================================================")
    print("ALL 8 ENDPOINTS & AI MULTI-DRUG DOSE ENGINE PASSED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    test()
