"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: data_manager.py
======================================================
Yazar: Biyoinformatik Veri Yönetimi & Makine Öğrenmesi Ekibi
Açıklama:
    Tüm veri setini (20+ literatür bileşiği + De Novo AI adayları) derler,
    RDKit ile kimyasal deskriptörleri çıkarır, Scikit-Learn ile farmakolojik
    ve biyolojik potans sınıflandırması yapar ve sonuçları SQLite veritabanında
    ve JSON/CSV dosyalarında kalıcı olarak saklar.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


class DrosophilaDataManager:
    """
    Veri derleme, RDKit deskriptör üretimi, ML sınıflandırma ve
    SQLite kalıcılık yöneticisi.
    """

    DB_PATH = os.path.join(os.path.dirname(__file__), "data", "in_silico_drosophila.db")
    CSV_INPUT = os.path.join(os.path.dirname(__file__), "data", "drosophila_anticancer_dataset.csv")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

    # De Novo Yapay Zeka Tarafından Üretilen Ek Adaylar
    DE_NOVO_ADDITIONS = [
        {
            "Molecule_Name": "DeNovo_Champion",
            "Database_ID": "AI-GEN: 2026-CHAMPION-01",
            "Canonical_SMILES": "CC1=NC=C(C=C1)CCN(C)C(=O)CF",
            "Target_Pathway_or_Gene": "nAChRalpha7 / Cholinergic Anti-Inflammatory & Egress",
            "Assay_Type_and_System": "In silico 3D Connectome & Lymph Gland Hematopoiesis",
            "Biological_Activity": "EC50 = 45 nM; 1.0s response latency, 0% toxicity",
            "Academic_Reference": "In Silico De Novo Design (Gemini AI Digital Twin Engine, 2026)"
        },
        {
            "Molecule_Name": "DeNovo_Alpha_1",
            "Database_ID": "AI-GEN: 2026-ALPHA-01",
            "Canonical_SMILES": "CN1CCC[C@H]1c2cccnc2F",
            "Target_Pathway_or_Gene": "nAChR / DopR Neuro-Immune Cross-talk",
            "Assay_Type_and_System": "In silico 3D Larval Connectome Model",
            "Biological_Activity": "EC50 = 150 nM; 1.0s response latency, 8% toxicity",
            "Academic_Reference": "In Silico De Novo Design (Gemini AI Digital Twin Engine, 2026)"
        }
    ]

    def __init__(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        self.df_compounds: Optional[pd.DataFrame] = None
        self.ml_summary: Dict[str, Any] = {}

    def compile_and_classify_dataset(self) -> pd.DataFrame:
        """
        1. CSV'den literatür moleküllerini okur ve De Novo adayları ekler.
        2. RDKit ile 10 adet fizikokimyasal deskriptör hesaplar.
        3. Molekülleri 4 farmakolojik kategoriye ve 3 biyolojik potans sınıfına ayırır.
        4. Random Forest ile sınıflandırma modeli eğitir ve özellik önemlerini belirler.
        5. SQLite veritabanına ve sonuç dosyalarına kaydeder.
        """
        print("[1/4] Veri seti okunuyor ve RDKit deskriptörleri çıkarılıyor...")
        raw_df = pd.read_csv(self.CSV_INPUT)
        combined_records = raw_df.to_dict(orient="records") + self.DE_NOVO_ADDITIONS

        enriched_rows = []
        for row in combined_records:
            smiles = row["Canonical_SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # RDKit Fizikokimyasal Deskriptörleri
            mw = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            hbd = int(rdMolDescriptors.CalcNumHBD(mol))
            hba = int(rdMolDescriptors.CalcNumHBA(mol))
            rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
            arom_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            f_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            qed_score = float(QED.qed(mol))

            # Farmakolojik Kategori Sınıflandırması
            name = row["Molecule_Name"]
            pathway = row["Target_Pathway_or_Gene"]
            if "DeNovo" in name:
                category = "De Novo Sentetik AI"
            elif any(chem in name for chem in ["Carboplatin", "Cisplatin", "Gemcitabine", "Methotrexate", "Doxorubicin", "5-Fluorouracil", "Paclitaxel"]):
                category = "Geleneksel Kemoterapötik"
            elif any(nat in name for nat in ["Curcumin", "Resveratrol", "Toyocamycin", "Withaferin A", "Epigallocatechin Gallate"]):
                category = "Doğal / Fitokimyasal Ajan"
            else:
                category = "Hedefe Yönelik İnhibitör"

            # Biyolojik Potans Sınıflandırması (Drosophila Test Çıktısına Göre)
            activity_str = str(row["Biological_Activity"]).lower()
            if "nm" in activity_str or "0." in activity_str or "1.0 u" in activity_str or "2.5 u" in activity_str:
                potency_class = "Yüksek Potans (Sub-mikromolar / <1-5 uM)"
                potency_code = 2
            elif "diet" in activity_str or "50 u" in activity_str or "100 u" in activity_str:
                potency_class = "Düşük Potans / Diyet Düzeyi (>25 uM)"
                potency_code = 0
            else:
                potency_class = "Orta Potans (5-25 uM)"
                potency_code = 1

            enriched_rows.append({
                **row,
                "Canonical_SMILES": Chem.MolToSmiles(mol),
                "Molecular_Weight": round(mw, 2),
                "LogP": round(logp, 2),
                "TPSA": round(tpsa, 2),
                "HBD": hbd,
                "HBA": hba,
                "Rotatable_Bonds": rotb,
                "Aromatic_Rings": arom_rings,
                "Fraction_CSP3": round(f_csp3, 3),
                "Heavy_Atoms": heavy_atoms,
                "QED_Drug_Likeness": round(qed_score, 3),
                "Pharmacological_Category": category,
                "Predicted_Potency_Class": potency_class,
                "Potency_Code": potency_code
            })

        self.df_compounds = pd.DataFrame(enriched_rows)

        # [2/4] Makine Öğrenmesi Sınıflandırma Analitiği
        print("[2/4] Makine öğrenmesi modeli eğitiliyor (Random Forest Classifier)...")
        feature_cols = [
            "Molecular_Weight", "LogP", "TPSA", "HBD", "HBA",
            "Rotatable_Bonds", "Aromatic_Rings", "Fraction_CSP3", "QED_Drug_Likeness"
        ]
        X = self.df_compounds[feature_cols].values
        y = self.df_compounds["Potency_Code"].values

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = dict(zip(feature_cols, [round(float(imp), 4) for imp in rf.feature_importances_]))
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        self.ml_summary = {
            "total_compounds": len(self.df_compounds),
            "categories_distribution": self.df_compounds["Pharmacological_Category"].value_counts().to_dict(),
            "potency_distribution": self.df_compounds["Predicted_Potency_Class"].value_counts().to_dict(),
            "feature_importances": sorted_imp,
            "average_descriptors": {
                "avg_mw": round(float(self.df_compounds["Molecular_Weight"].mean()), 1),
                "avg_logP": round(float(self.df_compounds["LogP"].mean()), 2),
                "avg_tpsa": round(float(self.df_compounds["TPSA"].mean()), 1),
                "avg_qed": round(float(self.df_compounds["QED_Drug_Likeness"].mean()), 2)
            }
        }

        # [3/4] SQLite Veritabanı Kaydı
        print(f"[3/4] Sonuçlar SQLite veritabanına kaydediliyor: {self.DB_PATH}...")
        self._persist_to_sqlite()

        # [4/4] Çıktı Dosyaları
        csv_out = os.path.join(self.RESULTS_DIR, "classified_compounds.csv")
        json_out = os.path.join(self.RESULTS_DIR, "classification_summary.json")
        self.df_compounds.to_csv(csv_out, index=False)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(self.ml_summary, f, indent=2, ensure_ascii=False)

        print(f"[Tamamlandı] {len(self.df_compounds)} bileşik başarıyla sınıflandırıldı ve kaydedildi.")
        return self.df_compounds

    def _persist_to_sqlite(self):
        """SQLite veritabanı şemasını kurar ve tabloları doldurur."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Tablo 1: compounds (Moleküller ve Deskriptörler)
        cursor.execute("DROP TABLE IF EXISTS compounds")
        self.df_compounds.to_sql("compounds", conn, if_exists="replace", index=False)

        # Tablo 2: ml_analytics (Model Metrikleri ve Özellik Önemleri)
        cursor.execute("DROP TABLE IF EXISTS ml_analytics")
        cursor.execute("""
            CREATE TABLE ml_analytics (
                metric_key TEXT PRIMARY KEY,
                metric_json TEXT
            )
        """)
        cursor.execute("INSERT INTO ml_analytics VALUES (?, ?)", ("feature_importances", json.dumps(self.ml_summary["feature_importances"])))
        cursor.execute("INSERT INTO ml_analytics VALUES (?, ?)", ("categories_distribution", json.dumps(self.ml_summary["categories_distribution"])))
        cursor.execute("INSERT INTO ml_analytics VALUES (?, ?)", ("average_descriptors", json.dumps(self.ml_summary["average_descriptors"])))

        # Tablo 3: benchmark_history (Simülasyon Koşu Geçmişi)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                molecule_name TEXT,
                time_to_trigger_s REAL,
                hemocytes_produced INTEGER,
                tumor_clearance_pct REAL,
                toxicity_pct REAL,
                fitness_score REAL
            )
        """)

        conn.commit()
        conn.close()

    def record_simulation_run(self, candidate_result: Any):
        """Her bir simülasyon koşusunu veritabanında saklar."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO benchmark_history (
                molecule_name, time_to_trigger_s, hemocytes_produced,
                tumor_clearance_pct, toxicity_pct, fitness_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            candidate_result.candidate_name,
            candidate_result.time_to_trigger_s,
            candidate_result.total_hemocytes_produced,
            candidate_result.tumor_clearance_pct,
            candidate_result.systemic_toxicity_pct,
            candidate_result.multiobjective_fitness_score
        ))
        conn.commit()
        conn.close()

    def get_all_compounds(self) -> List[Dict[str, Any]]:
        """Veritabanındaki tüm bileşikleri döndürür."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM compounds ORDER BY Pharmacological_Category, Molecule_Name")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


if __name__ == "__main__":
    manager = DrosophilaDataManager()
    df = manager.compile_and_classify_dataset()
    print("\n" + "=" * 80)
    print("  DROSOPHILA ANTİKANSER VERİ DERLEME VE SINIFLANDIRMA ÖZETİ")
    print("=" * 80)
    print("  Kategori Dağılımı:")
    for cat, count in manager.ml_summary["categories_distribution"].items():
        print(f"    * {cat:<32} : {count} bileşik")
    print("\n  En Belirleyici Moleküler Özellikler (Feature Importance):")
    for feat, imp in list(manager.ml_summary["feature_importances"].items())[:5]:
        print(f"    * {feat:<25} : %{imp * 100:.1f}")
    print("=" * 80 + "\n")
