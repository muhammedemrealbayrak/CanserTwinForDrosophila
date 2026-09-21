# 🧠 Drosophila In Silico Neuro-Immune Digital Twin
### 3D Whole-Brain Connectome (FlyWire FAFB v783) & Multimodal Oncological Synergy Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/cheminformatics-RDKit%202024-green.svg)](https://www.rdkit.org/)
[![Three.js](https://img.shields.io/badge/visualization-Three.js%20WebGL-black.svg)](https://threejs.org/)
[![Connectome](https://img.shields.io/badge/dataset-FlyWire%20FAFB%20v783%20(Nature%202024)-orange.svg)](https://flywire.ai/)
[![Algorithm](https://img.shields.io/badge/synergy-Chou--Talalay%20Median--Effect-purple.svg)](https://pubmed.ncbi.nlm.nih.gov/16564633/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Proje Özeti & Bilimsel Arka Plan (Scientific Abstract)

**Drosophila In Silico Neuro-Immune Digital Twin**, model organizma *Drosophila melanogaster*'ın tam beyin elektron mikroskobu konektomu (**Princeton FlyWire FAFB v783 - Nature 2024**, 139.248 nöron ve 54.5 milyon sinaps) ile 3D invaziv tümör mikroçevresi arasındaki **185 ms nöro-immün eferent refleks yayını** simüle eden ileri düzey bir biyo-hesaplamalı dijital ikiz ve onkolojik ilaç keşif platformudur.

Platform; mantar cisimciği (**Kenyon Hücresi KCg-m**) ve optik lob kolinerjik ateşlemesiyle tetiklenen lenf bezi hemosit salınımını (**plazmatositler, lamellositler, kristal hücreler**), klonal heterojeniteye sahip 3D mikrovasküler tümör dokusundaki hücre lizisini, tümörün salgıladığı kaşektik toksin birikimini ve metabolik yakıt tüketimini (**ATP, Glikoliz, BCAA, Lipidler**) gerçek zamanlı diferansiyel denklemlerle modellemektedir.

---

## 🔬 Temel Mimari Bileşenler ve Özellikler

### 1. 🧫 3D Hücresel Simülasyon Motoru (`platform_engine.py`)
* **Klonal Heterojenite & İlaç Direnci:** 150 başlangıç kanser hücresi içerisinde duyarlı klonlar ile hedefe yönelik tedavilere (MEK/Ras hiper-proliferasyonu) dirençli mutant klonların eşzamanlı evrimi.
* **Gerçekçi Toksisite ve Konakçı Canlılığı (CTCAE / RECIST 1.1):** Birikimli kaşektik doku yıkımı ve aşırı dozajda konakçı vitalitesi hesaplanır. Doku toksisitesi $\ge \%45.0$ olduğunda toksik organizma ölümü gerçekleşir.
* **Metabolik Yakıt Dinamiği (`biology/fuel_metabolism.py`):** Kanser dokusunun Warburg etkisiyle glikoliz, ATP, BCAA ve serbest yağ asitlerini tüketme hızını simüle eder.
* **Lenf Bezi Egress Kinetiği (`biology/lymph_gland.py`):** Nöral asetilkolin (ACh) ve sNPF dalgasıyla lenf bezinden hemositlerin devriye ve fagositoz için dokuya göçünü yürütür.

### 2. 🤖 Otonom AI Sinerjik Kokteyl Sentezleyici (`denovo_ai/cocktail_generator.py`)
* **Chou-Talalay Medyan Etki Eşitliği & Kombinasyon İndeksi ($CI$):**
  $$CI = \sum_{i=1}^{n} \frac{D_i}{(D_x)_i} = \frac{D_1}{(D_x)_1} + \frac{D_2}{(D_x)_2} + \dots + \frac{D_n}{(D_x)_n}$$
  * $CI < 0.35$: **🌟 Süper Sinerji** (Örn: $CI = 0.28 - 0.32$)
  * $0.35 \le CI < 0.70$: **⚡ Kuvvetli Sinerji**
  * $CI \ge 1.0$: Antagonist veya additif kombinasyonlar elenir.
* **Doz Azaltım İndeksi (DRI):** Toksisiteyi %85–90 azaltarak ajanların mikro-dozaj seviyelerini ($\mu\text{M}$) belirler.
* **Hedefli Stratejiler:** MEK-Bypass Triple-Hit, Metronomik Kemo-İmmün Kurtarma, Metabolik Warburg Açlığı ve Serbest AI Keşfi.
* **Kalıcı Depolama:** Sentezlenen tüm kokteyller `data/ai_generated_cocktails.json` içerisinde saklanır ve simülatöre 1 tıkla aktarılır.

### 3. 🧠 FlyWire FAFB v783 Tam Beyin Konektomu (`data/`)
* **139.248 Nöron Koordinat Haritası:** Elektron mikroskobu ile taranmış tüm beyin nöronlarının 3D WebGL nokta bulutu.
* **Nörotransmitter ve Bölge Filtreleri:** Asetilkolin (86.193), Glutamat (24.875), GABA (19.171), Dopamin (5.909), Serotonin (2.282), Oktopamin (216) ve beyin nöropilleri.
* **4'lü Sinaptik Devre:** KCg-m + DAN-PPL1 + MBON-$\gamma$1 + PN-AL gerçek sinaptik bağlantı modeli.

### 4. 📚 Molekül Veri Bankası & Canlı NCBI PubChem Entegrasyonu
* **56+ Onaylı Antikanser Ajanı:** Hedefe yönelik kinaz inhibitörleri (Trametinib, Cobimetinib), klasik kemoterapötikler (Sisplatin, Paklitaksel, Doksorubisin), fitokimyasallar (Kurkumin, Resveratrol, EGCG, Kersetin) ve De Novo AI analogları.
* **Canlı PubChem PUG REST API:** 115+ milyon kimyasal bileşiği anlık sorgulama, canonical SMILES ve 2D Lewis vektör çizimi.
* **RDKit QSAR & Lipinski Ro5 Analizi:** Moleküler ağırlık, LogP, TPSA, HBD/HBA ve oral biyoyararlanım tahmini.

### 5. 🏥 Multimodal Onkolojik Tedavi Protokolleri & Radyoterapi
* **Sitotoksik Kemoterapi:** DNA çapraz bağlayıcı ve mikrotübül dondurucu rejim.
* **Stereotaktik SABR Radyoterapisi:** Tümör nodülüne odaklanmış 8.0 Gy iyonlaştırıcı radyasyon darbesi ile doğrudan çift zincir DNA kırığı (DSB).
* **Anti-CD47 İmmünoterapi:** Kanser hücrelerinin "beni yeme" kalkanını kırarak fagositozu 2.4 katına çıkarma.
* **2-Deoksiglukoz (2-DG) Metabolik Açlık:** Hekzokinaz-II blokajı ile Warburg glikolizini kesme.
* **Metronomik Multimodal Kurtarma:** Düşük doz sürekli kemoterapi + nöral kolinerjik ateşleme + immün kontrol noktası kalkanı.

---

## 🖥️ Kullanıcı Arayüzü Sekmeleri (Laboratuvar Konsolu)

| Sekme | Başlık | Açıklama |
| :---: | :--- | :--- |
| **1** | 🧬 3D Canlı Simülatör & Konektom | 3D tümör mikroçevresi, hemosit lizisi, canlı Vm ve Kalsiyum osiloskopları. |
| **2** | 🖥️ İkili Senkron Görünüm (Dual View) | Sol ekranda FlyWire beyin konektomu, sağ ekranda 3D tümör ve refleks köprüsü. |
| **3** | 📚 Molekül Veri Bankası | 56+ bileşiklik veri seti, CSV/JSON indirme ve canlı filtreleme. |
| **4** | 🧪 Sinerjik Kokteyller & Reçeteler | Chou-Talalay sinerji kokteylleri ve **🤖 Otonom AI Kokteyl Sentezleyici**. |
| **5** | 🌐 Çevrimiçi Veri & Canlı PubChem | Gerçek zamanlı PubChem sorgusu ve RDKit QSAR Lewis çizimi. |
| **6** | 🏆 Liderlik Tablosu & Benchmark | Çok kriterli fitness formülasyonu ile 20 dakikalık zamana karşı ilaç yarışı. |
| **7** | 🤖 Makine Öğrenmesi & QSAR | Random Forest öznitelik önemi, De Novo Studio ve otonom Hill-Langmuir dozaj optimizasyonu. |
| **8** | 🏥 Alternatif Onkoloji Protokolleri | Kemoterapi, SABR radyasyon darbeleri, Anti-CD47 immünoterapi rehberi. |

---

## 🚀 Kurulum ve Çalıştırma (Getting Started)

### Gereksinimler
* Python 3.10, 3.11 veya 3.12
* Modern bir web tarayıcısı (Chrome, Edge, Firefox - WebGL destekli)

### Adım 1: Depoyu Klonlayın
```bash
git clone https://github.com/muhammedemrealbayrak/CanserTwinForDrosophila.git
cd CanserTwinForDrosophila
```

### Adım 2: Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```
*(Eğer `rdkit` yüklü değilse: `pip install rdkit numpy`)*

### Adım 3: Laboratuvar Konsolunu Başlatın
```bash
python web_dashboard/app_server.py
```
Konsol başlatıldığında web tarayıcınız otomatik olarak açılacaktır:
👉 **http://127.0.0.1:8060**

---

## 📡 REST API Uç Noktaları (API Catalog)

| Metot | Uç Nokta | Açıklama |
| :--- | :--- | :--- |
| `GET` | `/` | Web Laboratuvar Konsolu (HTML, CSS, Three.js) |
| `GET` | `/api/compounds` | Tüm 56+ biyoaktif molekül veri bankası |
| `GET` | `/api/cocktails` | Temel ve yapay zeka tarafından sentezlenmiş sinerjik kokteyller |
| `POST` | `/api/cocktails/generate` | Hedefe göre otonom AI Chou-Talalay sinerjik kokteyl tahmini |
| `POST` | `/api/set_cocktail` | Seçilen kokteyli 3D simülatöre yükleme ve dokuyu hazırlama |
| `POST` | `/api/step` | Simülasyonu `dt` saniye ilerletir; hücre koordinatları ve telemetri döner |
| `POST` | `/api/reset` | 150 kanser hücresi ve 30 hemositle simülatörü sıfırlar |
| `POST` | `/api/apply_radiation` | Tümör nodülüne odaklı 8.0 Gy stereotaktik radyasyon darbesi uygular |
| `POST` | `/api/set_treatment_modality`| Tedavi modalitesini değiştirir (Kemoterapi, İmmünoterapi, vb.) |
| `POST` | `/api/online/fetch_compound` | PubChem REST API üzerinden anlık molekül çeker ve veritabanına ekler |
| `POST` | `/api/predict_qsar` | SMILES diziliminin RDKit farmakokinetik profilini çıkarır |
| `POST` | `/api/denovo/generate` | De Novo yapay zeka mutasyon stüdyosu analogları türetir |
| `POST` | `/api/ai/optimize_dose` | Hill-Langmuir doygunluğuna göre optimum dozu (µM) hesaplar |
| `GET` | `/api/connectome/whole_brain`| 139.248 nöronluk FlyWire FAFB v783 3D koordinatları |
| `GET` | `/api/connectome/multi_neuron`| 4'lü sinaptik devre koordinatları ve bağlantıları |
| `GET` | `/api/export_dataset?format=csv`| Veri setini CSV veya JSON olarak dışa aktarır |

---

## 📂 Dizin Yapısı (Project Structure)

```
in_silico_digital_twin_platform/
├── biology/                          # Biyolojik motorlar ve kinetik modeller
│   ├── fuel_metabolism.py            # Kanser hücre metabolizması ve kaşeksi
│   └── lymph_gland.py                # Drosophila lenf bezi hemosit salınımı
├── data/                             # Veritabanı ve konektom veri dosyaları
│   ├── ai_generated_cocktails.json   # AI tarafından sentezlenen kokteyller
│   ├── drosophila_anticancer_dataset.csv # 56+ bileşik veri seti
│   ├── flywire_whole_brain_140k.json # FlyWire 140k nöron koordinatları
│   └── in_silico_drosophila.db       # SQLite3 molekül veritabanı
├── denovo_ai/                        # Yapay Zeka Tasarım & Dozaj Motorları
│   ├── cocktail_generator.py         # Chou-Talalay AI Kokteyl Sentezleyici
│   ├── dose_optimizer.py             # Hill kinetiği otonom doz belirleyici
│   └── molecule_generator.py         # De Novo RDKit molekül mutasyon motoru
├── pipeline/                         # Veri boru hatları ve konektörler
│   ├── benchmark_engine.py           # Zaman yarışı ve fitness liderlik tablosu
│   ├── online_data_fetcher.py        # NCBI PubChem PUG REST API çekici
│   └── pubchem_connector.py          # SMILES ve QSAR profilleme motoru
├── web_dashboard/                    # Laboratuvar Web Arayüzü & REST API
│   ├── app_server.py                 # HTTP REST API sunucusu (Port: 8060)
│   └── static/
│       └── index.html                # 8 sekmeli Three.js WebGL arayüzü
├── platform_engine.py                # 3D Hücresel ve Nöro-İmmün Ana Motor
├── run_platform.py                   # Konsol simülasyon başlatıcısı
├── requirements.txt                  # Python bağımlılıkları
├── .gitignore                        # Git hariç tutma kuralları
└── README.md                         # Proje teknik dokümantasyonu
```

---

## 📑 Bilimsel Kaynakça & Referanslar (Academic References)

1. **FlyWire Whole-Brain Connectome:** Dorkenwald, S., et al. (2024). *Neuronal wiring diagram of an adult brain.* **Nature**, 634, 124–138.
2. **Chou-Talalay Sinerji Yöntemi:** Chou, T. C. (2006). *Theoretical basis, experimental design, and computerized simulation of synergism and antagonism in drug combination studies.* **Pharmacological Reviews**, 58(3), 621–681.
3. **Drosophila İmmün-Tümör Etkileşimi:** Pastor-Pareja, J. C., et al. (2008). *An innate immune response of Drosophila to malignant tumors.* **Disease Models & Mechanisms**, 1(2-3), 144–154.
4. **Warburg Etkisi & Metabolik Açlık:** Vander Heiden, M. G., et al. (2009). *Understanding the Warburg effect: the metabolic requirements of cell proliferation.* **Science**, 324(5930), 1029–1033.

---

## 📜 Lisans (License)
Bu proje **MIT Lisansı** ile lisanslanmıştır. Akademik ve klinik araştırmalarda özgürce kullanılabilir ve geliştirilebilir.
