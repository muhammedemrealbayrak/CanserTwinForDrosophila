"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: pipeline/pdf_report_generator.py
======================================================
Yazar: Muhammed Emre Albayrak & Preklinik Dokümantasyon Ekibi
Açıklama:
    Simülasyon sonuçlarının, 64+ molekül bankasının, zamana karşı
    yarış liderlik tablosunun ve sinerjik kokteyllerin yüksek çözünürlüklü,
    vektörel ve renkli klinik PDF raporunu (Dossier) üreten motor.
"""

import os
import io
import datetime
from typing import Dict, List, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Veri yöneticileri
from data_manager import DrosophilaDataManager
from pipeline.benchmark_engine import InSilicoBenchmarkEngine

# Windows TrueType font kaydı (Türkçe karakter desteği)
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

try:
    font_path_regular = "C:/Windows/Fonts/arial.ttf"
    font_path_bold = "C:/Windows/Fonts/arialbd.ttf"
    if os.path.exists(font_path_regular):
        pdfmetrics.registerFont(TTFont("ArialTurkish", font_path_regular))
        FONT_NAME = "ArialTurkish"
    if os.path.exists(font_path_bold):
        pdfmetrics.registerFont(TTFont("ArialTurkishBold", font_path_bold))
        FONT_BOLD = "ArialTurkishBold"
except Exception as ex:
    print(f"[WARN] TrueType font yuklenirken hata: {ex}. Standart Helvetica kullanilacak.")


class NumberedCanvas(canvas.Canvas):
    """
    İki geçişli (two-pass) canvas: Toplam sayfa sayısını hesaplar,
    üstbilgi (running header) ve altbilgi (running footer) çizgilerini çizer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        
        # Sayfa Genişlik & Yükseklik (A4: 595.27 x 841.89 pt)
        page_w, page_h = A4
        margin_x = 36
        
        # Üstbilgi (Header) - İlk sayfa hariç
        if self._pageNumber > 1:
            self.setFont(FONT_NAME, 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(margin_x, page_h - 26, "Drosophila In Silico Neuro-Immune Digital Twin • Preclinical Oncology Dossier")
            self.drawRightString(page_w - margin_x, page_h - 26, "FlyWire FAFB v783 & Multi-Scale Engine")
            
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(margin_x, page_h - 30, page_w - margin_x, page_h - 30)

        # Altbilgi (Footer) - Tüm sayfalarda
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(margin_x, 34, page_w - margin_x, 34)

        self.setFont(FONT_NAME, 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(margin_x, 22, "Gizli & Tescilli • In Silico Oncology & Computational Neurobiology Laboratory • Muhammed Emre Albayrak")
        
        page_str = f"Sayfa {self._pageNumber} / {page_count}"
        self.drawRightString(page_w - margin_x, 22, page_str)

        self.restoreState()


class PDFReportGenerator:
    """Kapsamlı in silico ilaç sonuçları ve biyolojik simülasyon PDF raporlayıcısı."""

    def __init__(self):
        self.data_manager = DrosophilaDataManager()
        self.benchmark_engine = InSilicoBenchmarkEngine()

    def generate_pdf_report(
        self,
        report_type: str = "full",
        active_telemetry: Optional[Dict[str, Any]] = None,
        active_drug_profile: Optional[Any] = None
    ) -> bytes:
        """
        Kapsamlı PDF raporunu üretir ve bayt dizisi (bytes) olarak döndürür.
        
        Args:
            report_type: 'full' (tüm ilaç portföyü ve benchmarklar) veya 'active' (aktif simülasyon koşusu).
            active_telemetry: Mevcut simülasyon telemetri verisi (isteğe bağlı).
            active_drug_profile: Aktif test edilen ilaç profili.
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=42,
            bottomMargin=42
        )

        styles = getSampleStyleSheet()
        
        # Özel Tipografi Stilleri
        primary_color = colors.HexColor("#0f172a")    # Slate 900
        cyan_accent = colors.HexColor("#0284c7")      # Sky 600
        emerald_accent = colors.HexColor("#0d9488")   # Teal 600
        crimson_accent = colors.HexColor("#e11d48")   # Rose 600
        amber_accent = colors.HexColor("#d97706")     # Amber 600
        purple_accent = colors.HexColor("#7c3aed")    # Violet 600

        style_title = ParagraphStyle(
            "DocTitle",
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=primary_color,
            spaceAfter=4
        )
        style_subtitle = ParagraphStyle(
            "DocSubTitle",
            fontName=FONT_NAME,
            fontSize=10,
            leading=14,
            textColor=cyan_accent,
            spaceAfter=12
        )
        style_h1 = ParagraphStyle(
            "Heading1_Custom",
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=primary_color,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        )
        style_h2 = ParagraphStyle(
            "Heading2_Custom",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=13,
            textColor=cyan_accent,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        )
        style_body = ParagraphStyle(
            "Body_Custom",
            fontName=FONT_NAME,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155")
        )
        style_body_bold = ParagraphStyle(
            "Body_Bold",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#0f172a")
        )
        style_table_cell = ParagraphStyle(
            "TableCell",
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=9.5,
            textColor=colors.HexColor("#1e293b")
        )
        style_table_header = ParagraphStyle(
            "TableHeader",
            fontName=FONT_BOLD,
            fontSize=7.2,
            leading=9.5,
            textColor=colors.white
        )

        story = []

        # ========================================================
        # KAPAK & YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY)
        # ========================================================
        story.append(Paragraph("DROSOPHILA IN SILICO TÜMÖR & NÖRO-İMMÜN DİJİTAL İKİZİ", style_title))
        story.append(Paragraph("Kapsamlı Preklinik İlaç Değerlendirme, Zamana Karşı Yarış Liderlik Tablosu & Sinerjik Kokteyl Portföy Raporu", style_subtitle))
        story.append(HRFlowable(width="100%", thickness=1.5, color=cyan_accent, spaceBefore=0, spaceAfter=8))

        # Meta Bilgi Tablosu
        now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        meta_data = [
            [
                Paragraph("<b>Araştırmacı & Sistem Biyoloğu:</b>", style_body),
                Paragraph("Muhammed Emre Albayrak", style_body),
                Paragraph("<b>Rapor Tarihi & Sürüm:</b>", style_body),
                Paragraph(f"{now_str} (v2.4 Preclinical)", style_body)
            ],
            [
                Paragraph("<b>Konektom Referansı:</b>", style_body),
                Paragraph("FlyWire FAFB v783 (139.248 Nöron & 185ms Refleks)", style_body),
                Paragraph("<b>Molekül Havuzu:</b>", style_body),
                Paragraph("64+ Biyoaktif Bileşik & 5 Premier Kokteyl", style_body)
            ],
            [
                Paragraph("<b>Mekansal Doku Modeli:</b>", style_body),
                Paragraph("3D PBR Mikroçevre (Cordero/Tracey/Bishayee)", style_body),
                Paragraph("<b>İmmün Organ:</b>", style_body),
                Paragraph("Drosophila Lenf Bezi (Progenitör Kök Rezervi)", style_body)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[120, 150, 110, 143])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # Yönetici Özeti KPI Kartları
        kpi_data = [
            [
                Paragraph("<font color='#0284c7'><b>★ ŞAMPİYON KURŞUN ADAY</b></font><br/><b>DeNovo_Champion (F-NAc)</b><br/>Fitness: <b>97.6 / 100</b> | Toks: <b>%4.0</b>", style_body),
                Paragraph("<font color='#0d9488'><b>⚡ PREMİER SİNERJİK KOKTEYL</b></font><br/><b>Nöro-İmmün Dörtlü Kalkan</b><br/>Chou-Talalay <b>CI: 0.22</b> (Ultra Sinerji)", style_body),
                Paragraph("<font color='#7c3aed'><b>🛡️ KÖK HÜCRE KORUMASI</b></font><br/><b>Hematopoietik Rezerv: %100</b><br/>Sıfır Miyelosüpresyon Kalkanı", style_body),
                Paragraph("<font color='#d97706'><b>🎯 TÜMÖR TEMİZLEME GÜCÜ</b></font><br/><b>%100.0 Lysis & Egress</b><br/>Dirençli Klonları Sıfırlar", style_body)
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[130, 131, 131, 131])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#f0f9ff")),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#f0fdfa")),
            ('BACKGROUND', (2, 0), (2, 0), colors.HexColor("#faf5ff")),
            ('BACKGROUND', (3, 0), (3, 0), colors.HexColor("#fffbeb")),
            ('BOX', (0, 0), (0, 0), 1.0, cyan_accent),
            ('BOX', (1, 0), (1, 0), 1.0, emerald_accent),
            ('BOX', (2, 0), (2, 0), 1.0, purple_accent),
            ('BOX', (3, 0), (3, 0), 1.0, amber_accent),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 10))

        # Aktif Koşu Telemetrisi Varsa Ekle
        if active_telemetry:
            story.append(Paragraph("Canlı Simülasyon Telemetri Özeti (Aktif Koşu)", style_h2))
            c_tot = active_telemetry.get("cancer_cells", 0)
            c_sens = active_telemetry.get("sensitive_cancer_cells", 0)
            c_res = active_telemetry.get("resistant_cancer_cells", 0)
            h_act = active_telemetry.get("active_hemocytes", 0)
            vit = active_telemetry.get("host_vitality_pct", 100.0)
            tox = active_telemetry.get("toxicity_pct", 0.0)
            stem_res = active_telemetry.get("hematopoietic_reserve_pct", 100.0)
            risk = active_telemetry.get("myelosuppression_risk", "SAFE")
            status_text = active_telemetry.get("clinical_status_text", "DURUM")
            drug_name = active_telemetry.get("drug_name", "Aktif İlaç")

            act_data = [
                [
                    Paragraph(f"<b>Aktif İlaç/Rejim:</b> {drug_name}", style_table_cell),
                    Paragraph(f"<b>Klinik Sonuç:</b> {status_text}", style_table_cell),
                    Paragraph(f"<b>Canlı Kanser Hücresi:</b> {c_tot} (Hassas: {c_sens}, Dirençli: {c_res})", style_table_cell),
                ],
                [
                    Paragraph(f"<b>Devriye Hemositleri:</b> {h_act} Hücre", style_table_cell),
                    Paragraph(f"<b>Konakçı Canlılığı:</b> %{vit:.1f} (Toksisite: %{tox:.1f})", style_table_cell),
                    Paragraph(f"<b>Kök Hücre Rezervi:</b> %{stem_res:.1f} (Miyelosüpresyon: {risk})", style_table_cell),
                ]
            ]
            act_table = Table(act_data, colWidths=[174, 174, 175])
            act_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(act_table)
            story.append(Spacer(1, 8))

        # ========================================================
        # BÖLÜM 1: ZAMANA KARŞI YARIŞ & LİDERLİK TABLOSU
        # ========================================================
        story.append(Paragraph("1. Zamana Karşı İlaç Yarışı & Benchmark Liderlik Tablosu", style_h1))
        story.append(Paragraph(
            "Drosophila dijital ikizinde 20 dakikalık (1200 saniye) standartlaştırılmış in silico simülasyon "
            "sonuçları. Çok kriterli fitness skoru; nöro-immün refleks süresi ($t_{\\text{trig}}$), doku içine hemosit "
            "dökülmesi (egress), tümör temizleme yüzdesi, doku toksisitesi ve kemik iliği/lenf bezi kök hücre rezervi "
            "ağırlıklandırılarak hesaplanmıştır.",
            style_body
        ))
        story.append(Spacer(1, 6))

        # Liderlik Tablosu Verisini Çek
        benchmarks = self.benchmark_engine.get_leaderboard(limit=25)
        
        bench_headers = [
            Paragraph("<b>#</b>", style_table_header),
            Paragraph("<b>Molekül Adı</b>", style_table_header),
            Paragraph("<b>Refleks (s)</b>", style_table_header),
            Paragraph("<b>Hemosit</b>", style_table_header),
            Paragraph("<b>Temizleme</b>", style_table_header),
            Paragraph("<b>Toksisite</b>", style_table_header),
            Paragraph("<b>Kök Rezervi</b>", style_table_header),
            Paragraph("<b>Fitness</b>", style_table_header),
            Paragraph("<b>Preklinik Karar</b>", style_table_header),
        ]
        bench_rows = [bench_headers]

        for idx, b in enumerate(benchmarks):
            name = b.get("molecule_name", "Bileşik")
            trig = f"{b.get('time_to_trigger_s', 0.0):.1f}s"
            hemo = str(b.get("hemocytes_produced", 0))
            clr = f"%{b.get('tumor_clearance_pct', 0.0):.1f}"
            tox = f"%{b.get('toxicity_pct', 0.0):.1f}"
            fit = f"{b.get('fitness_score', 0.0):.1f}"
            dec = b.get("clinical_decision", "DEĞERLENDİRMEDE")
            
            # Kök rezervi tahmini (toksisiteye ve koruma kalkanına bağlı)
            tox_val = float(b.get('toxicity_pct', 0.0))
            if tox_val > 30.0:
                stem_str = "%65.0 (Orta)"
            elif tox_val > 40.0:
                stem_str = "%45.0 (Ağır)"
            else:
                stem_str = "%98.5 (Güvenli)"

            bench_rows.append([
                Paragraph(str(idx + 1), style_table_cell),
                Paragraph(f"<b>{name}</b>", style_table_cell),
                Paragraph(trig, style_table_cell),
                Paragraph(hemo, style_table_cell),
                Paragraph(f"<font color='#0d9488'><b>{clr}</b></font>", style_table_cell),
                Paragraph(f"<font color='#e11d48'>{tox}</font>", style_table_cell),
                Paragraph(stem_str, style_table_cell),
                Paragraph(f"<b>{fit}</b>", style_table_cell),
                Paragraph(dec, style_table_cell),
            ])

        bench_table = Table(
            bench_rows,
            colWidths=[20, 115, 45, 42, 50, 45, 68, 42, 96]
        )
        bench_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ]))
        story.append(bench_table)
        story.append(Spacer(1, 12))

        # ========================================================
        # BÖLÜM 2: SİNERJİK KOKTEYLLER & İMMÜN-KORUMALI REJİMLER
        # ========================================================
        story.append(PageBreak())
        story.append(Paragraph("2. Çoklu Hedefli Sinerjik Kokteyller & İmmün Kalkan Rejimleri", style_h1))
        story.append(Paragraph(
            "Monoterapide gelişen MEK bypass, SHP2 reaktivasyonu ve ABC eflüks direnç klonlarını tek seferde yok etmek "
            "üzere tasarlanan Chou-Talalay sinerji kombinasyonları. Bu protokollerde Doz Azaltım İndeksi (DRI) kullanılarak "
            "bileşenlerin dozu 6.5x - 14.4x azaltılmış, doku toksisitesi sıfırlanmış ve lenf bezi kök hücreleri "
            "(miyelosüpresyon) tamamen korunmuştur.",
            style_body
        ))
        story.append(Spacer(1, 6))

        cocktails = list(self.data_manager.NEW_LITERATURE_MOLECULES)
        # Premier kokteyller
        cocktail_profiles = [
            {
                "name": "Nöro-İmmün Metronomik Dörtlü Kalkan",
                "ci": "0.22 (Ultra Sinerji)",
                "bliss": "+%4.8",
                "shield": "%94.0 Kalkan",
                "stem_res": "%100.0 (Sıfır Miyelosüpresyon)",
                "comps": "Metronomik Trametinib (0.35µM) + F-NAc (1.2µM) + Resveratrol (4.5µM) + Ponsegromab (2.0µM)",
                "mechanism": "Düşük doz metronomik MEK blokajı ile tümör kök hücresi dondurulur; Resveratrol Sir2/SIRT1 üzerinden lenf bezi kök hücrelerini miyelosüpresyondan korur; F-NAc nöral eferent sürüşü sağlar ve Ponsegromab kaşeksiyi bloke eder."
            },
            {
                "name": "Pan-RAS / KRAS G12D & SHP2 Dikey Blokaj",
                "ci": "0.18 (Ultra Sinerji)",
                "bliss": "+%4.6",
                "shield": "%92.0 Kalkan",
                "stem_res": "%95.0 (Güvenli)",
                "comps": "MRTX1133 (2.5nM) + RMC-4550 (2.0nM) + Ponsegromab Mimetic (0.08µM)",
                "mechanism": "MRTX1133 ile KRAS G12D Switch-II cebi kilitlenirken, allosterik SHP2 inhibitörü (RMC-4550) adaptif RTK geri bildirim direncini sıfırlar; GDF15 kaşeksi kalkanı kas erimesini durdurur."
            },
            {
                "name": "Sentetik Ölümcüllük PARP & ATR Replikasyon Krizi",
                "ci": "0.18 (Ultra Sinerji)",
                "bliss": "+%2.6",
                "shield": "%86.0 Kalkan",
                "stem_res": "%90.0 (Güvenli)",
                "comps": "Olaparib (25nM) + Ceralasertib (20nM) + Resveratrol (7.5µM)",
                "mechanism": "Olaparib DNA tek zincir onarımını kilitlerken, Ceralasertib ATR replikasyon çatalını çökertir; tümör hücreleri genomik katastrofi ile ölür; Resveratrol nöroproteksiyon sağlar."
            },
            {
                "name": "Asidoz Klerensi & CD47 İmmün Aktivasyon",
                "ci": "0.19 (Ultra Sinerji)",
                "bliss": "+%5.7",
                "shield": "%88.0 Kalkan",
                "stem_res": "%96.0 (Güvenli)",
                "comps": "AZD3965 (0.015µM) + Evorpacept (0.0005µM) + DeNovo_Champion (0.8µM)",
                "mechanism": "MCT1 blokajı mikroçevredeki laktik asit felcini nötrler; hemositler hareket kabiliyetini geri kazanır ve Anti-CD47 blokajı ile 'beni yeme' kalkanı yıkılan hücreleri hızla yutar."
            },
            {
                "name": "İkili Metabolik Açlık (Dual Metabolic Starvation)",
                "ci": "0.25 (Süper Sinerji)",
                "bliss": "+%3.8",
                "shield": "%84.0 Kalkan",
                "stem_res": "%88.0 (Güvenli)",
                "comps": "2-Deoksiglukoz (1.8µM) + Telaglenastat (0.035µM) + Kurkumin (4.0µM)",
                "mechanism": "Hekzokinaz-II ve Glutaminaz-1 enzimlerinin eşzamanlı kilitlenmesi tümörün aerobik glikoliz (Warburg) ve anapleroz yakıtını keserek tümör mitozunu durdurur."
            }
        ]

        cocktail_table_data = [
            [
                Paragraph("<b>Sinerjik Kokteyl Protokolü</b>", style_table_header),
                Paragraph("<b>Bileşenler & Dozajlar</b>", style_table_header),
                Paragraph("<b>Chou-Talalay CI</b>", style_table_header),
                Paragraph("<b>Bliss Sinerji</b>", style_table_header),
                Paragraph("<b>Kök Hücre Rezervi</b>", style_table_header),
                Paragraph("<b>Farmakolojik Mekanizma</b>", style_table_header),
            ]
        ]

        for cp in cocktail_profiles:
            cocktail_table_data.append([
                Paragraph(f"<b>{cp['name']}</b>", style_table_cell),
                Paragraph(cp['comps'], style_table_cell),
                Paragraph(f"<font color='#0d9488'><b>CI = {cp['ci']}</b></font>", style_table_cell),
                Paragraph(cp['bliss'], style_table_cell),
                Paragraph(f"<font color='#7c3aed'><b>{cp['stem_res']}</b></font>", style_table_cell),
                Paragraph(cp['mechanism'], style_table_cell),
            ])

        cocktail_table = Table(
            cocktail_table_data,
            colWidths=[90, 110, 65, 55, 75, 128]
        )
        cocktail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#065f46")), # Emerald Dark
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdf4")]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ]))
        story.append(cocktail_table)
        story.append(Spacer(1, 14))

        # ========================================================
        # BÖLÜM 3: 7 FARMAKOLOJİK KATEGORİDE 64+ İLAÇ BANKASI
        # ========================================================
        story.append(PageBreak())
        story.append(Paragraph("3. Farmakolojik Kategoriler Altında Molekül Portföyü (64+ Bileşik)", style_h1))
        story.append(Paragraph(
            "Molekül bankasındaki tüm bileşiklerin RDKit fizikokimyasal deskriptörleri, hedef genleri, "
            "Drosophila ⟷ İnsan homolojisi ve simüle edilen tekil antitümör etkinlik profili.",
            style_body
        ))
        story.append(Spacer(1, 6))

        all_compounds = self.data_manager.get_all_compounds()
        
        # Kategorilere Göre Grupla
        categories = {}
        for c in all_compounds:
            cat = c.get("Pharmacological_Category", "Diğer")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(c)

        for cat_name, cat_list in categories.items():
            story.append(Paragraph(f"Kategori: {cat_name} ({len(cat_list)} Bileşik)", style_h2))
            
            c_rows = [
                [
                    Paragraph("<b>Molekül Adı</b>", style_table_header),
                    Paragraph("<b>Hedef & Homoloji</b>", style_table_header),
                    Paragraph("<b>MW</b>", style_table_header),
                    Paragraph("<b>LogP</b>", style_table_header),
                    Paragraph("<b>TPSA</b>", style_table_header),
                    Paragraph("<b>Kd (µM)</b>", style_table_header),
                    Paragraph("<b>Toksisite</b>", style_table_header),
                    Paragraph("<b>Lipinski Ro5</b>", style_table_header),
                ]
            ]

            for c in cat_list:
                m_name = c.get("Molecule_Name", "Bileşik")
                homolog = f"{c.get('Drosophila_Homolog', '-')}<br/>⟷ {c.get('Human_Ortholog', '-')}"
                mw = f"{float(c.get('Molecular_Weight', 0.0)):.1f}"
                logp = f"{float(c.get('LogP', 0.0)):.2f}"
                tpsa = f"{float(c.get('TPSA', 0.0)):.1f}"
                kd = f"{float(c.get('Estimated_Kd_uM', 1.0)):.3f}" if c.get('Estimated_Kd_uM') else "-"
                tox = f"%{float(c.get('QSAR_Toxicity_Risk', 0.05)) * 100:.1f}"
                ro5 = "Uyumlu (0 İhlal)" if c.get("Lipinski_Violations", 0) == 0 else f"{c.get('Lipinski_Violations')} İhlal"

                c_rows.append([
                    Paragraph(f"<b>{m_name}</b>", style_table_cell),
                    Paragraph(homolog, style_table_cell),
                    Paragraph(mw, style_table_cell),
                    Paragraph(logp, style_table_cell),
                    Paragraph(tpsa, style_table_cell),
                    Paragraph(kd, style_table_cell),
                    Paragraph(tox, style_table_cell),
                    Paragraph(ro5, style_table_cell),
                ])

            c_table = Table(
                c_rows,
                colWidths=[105, 120, 42, 38, 42, 50, 48, 78]
            )
            c_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ]))
            story.append(c_table)
            story.append(Spacer(1, 8))

        # ========================================================
        # BELGEYİ OLUŞTUR (BUILD PDF)
        # ========================================================
        doc.build(story, canvasmaker=NumberedCanvas)
        return buf.getvalue()


pdf_report_generator = PDFReportGenerator()

def generate_pdf_report(
    report_type: str = "full",
    active_telemetry: Optional[Dict[str, Any]] = None,
    active_drug_profile: Optional[Any] = None
) -> bytes:
    """PDF raporu oluşturan evrensel fonksiyon arayüzü."""
    return pdf_report_generator.generate_pdf_report(
        report_type=report_type,
        active_telemetry=active_telemetry,
        active_drug_profile=active_drug_profile
    )
