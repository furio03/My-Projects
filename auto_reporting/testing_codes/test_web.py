from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from pathlib import Path
import matplotlib.pyplot as plt

# 1. FUNZIONE DI PRODUZIONE PDF (La versione che abbiamo concordato)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from pathlib import Path

def create_pdf_report(report_text, histogram_path=None, domain=None):
    file_path = "4μ_Executive_Report.pdf"
    # Margini leggermente più ampi per un look più "executive"
    doc = SimpleDocTemplate(file_path, pagesize=letter, topMargin=50, bottomMargin=50, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    
   # --- Palette Colori ---
    primary_blue = HexColor('#0D47A1')

    # --- Stili ---
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['h1'],
        fontSize=26,
        textColor=primary_blue,
        alignment=1,
        fontName='Helvetica-Bold',
        spaceAfter=5
    )
    
    domain_subtitle_style = ParagraphStyle(
        'DomainSubtitle',
        parent=styles['h2'],
        fontSize=12,
        textColor=primary_blue,
        alignment=1,
        fontName='Helvetica-BoldOblique',
        letterSpacing=1.5, # Un po' di aria tra le lettere per eleganza
        spaceAfter=15
    )

    body_style = ParagraphStyle(
        'ModernBody', parent=styles['Normal'], fontSize=10.5, leading=14,
        alignment=4, spaceAfter=12  # 4 = Giustificato
    )

    story = []

    # --- Header Pulito ---
    domain_display = domain.replace('_', ' ').upper() if domain else "GENERAL"
    
    # Titolo principale con la mu corposa
    story.append(Paragraph(f"4&mu; Executive Report", title_style))
    
    # Sottotitolo del Dominio (es. BUSINESS ECONOMICS)
    story.append(Paragraph(domain_display, domain_subtitle_style))
    
    # Linea singola, netta e professionale
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, hAlign='CENTER'))
    story.append(Spacer(1, 25))

    # --- Analisi del Testo ---
    paragraphs = report_text.split('\n\n')
    chart_inserted = False

    for p_text in paragraphs:
        p_text = p_text.strip()
        if p_text:
            story.append(Paragraph(p_text, body_style))
            
            # --- Integrazione Grafico ---
            # Inseriamo il grafico nel contesto statistico
            if not chart_inserted and histogram_path and any(word in p_text.lower() for word in ["statistical", "variables", "correlation"]):
                if Path(histogram_path).exists():
                    story.append(Spacer(1, 15))
                    img = RLImage(histogram_path, width=5.2*inch, height=3.5*inch)
                    img.hAlign = 'CENTER'
                    story.append(img)
                    # Didascalia in blu coordinato
                    caption_style = ParagraphStyle('Caption', parent= domain_subtitle_style, fontSize=9, spaceBefore=5)
                    story.append(Paragraph(f"Figure 1: {domain_display} Distribution Analysis", caption_style))
                    story.append(Spacer(1, 25))
                    chart_inserted = True

    # --- Footer Line ---
    story.append(Spacer(1, 30))
    # Rimuovi 'alignment=1' e usa 'hAlign' oppure ometti se la larghezza è 100%
    story.append(HRFlowable(width="100%", thickness=0.5, color=primary_blue, hAlign='CENTER'))
    
    doc.build(story)
    return file_path

# 2. GENERAZIONE DI UN GRAFICO DI TEST (Per simulare l'istogramma del tuo sito)
test_img_path = "temp_test_plot.png"
plt.figure(figsize=(6, 4))
plt.bar(['A', 'B', 'C'], [10, 20, 15], color='green', alpha=0.7)
plt.title("Test Distribution Chart")
plt.savefig(test_img_path)
plt.close()

# 3. TESTO DI PROVA (Simula l'output dell'IA basato sul tuo report precedente)
mock_report = """ As we delve into the dataset, our primary objective is to identify strategies to reduce the unit variable cost. The statistical results provide a comprehensive overview of the relationships between various variables, including price, quantity demanded, quantity supplied, and unit variable cost. Notably, the correlation matrix reveals a strong positive correlation between price and unit variable cost, indicating that as the price increases, the unit variable cost also tends to rise. This relationship is crucial in understanding how pricing strategies can impact production costs.
Furthermore, the correlation matrix shows a strong negative correlation between quantity demanded and price, suggesting that as the price increases, the quantity demanded decreases. This is a fundamental principle of economics, and it highlights the importance of balancing pricing strategies with demand. The correlation between quantity supplied and price is also negative, indicating that higher prices are associated with lower quantities supplied. These relationships provide valuable insights into the dynamics of the market and can inform decisions on pricing and production.
The K-means clustering profiles reveal distinct patterns in the data, with five clusters emerging based on the variables analyzed. Cluster 0, for instance, is characterized by a relatively low price and high quantity demanded, resulting in a lower unit variable cost. In contrast, Cluster 2 has a significantly higher price and lower quantity demanded, leading to a higher unit variable cost. These clusters provide a framework for understanding the different market segments and tailoring strategies to each segment's unique characteristics.
The domain-specific analysis offers additional insights, particularly in the context of customer segments and loss-making segments. The customer segments analysis reveals that Cluster 0 has the highest quantity demanded, suggesting that this segment is the most price-sensitive. The loss-making segments analysis, on the other hand, highlights that all product IDs are currently unprofitable, emphasizing the need to re-evaluate pricing and cost structures.
The forecast analysis suggests that the unit variable cost is expected to increase, with a trend indicating a rising cost over time. This forecast is critical in planning for future production and pricing strategies. While the r2 score is relatively low, indicating some uncertainty in the forecast, the overall trend is still informative for decision-making purposes.
In synthesizing these findings, it becomes apparent that reducing the unit variable cost will require a multifaceted approach. One potential strategy is to focus on increasing quantity demanded, as this is associated with lower unit variable costs. This could be achieved through targeted marketing campaigns or discounts. Additionally, optimizing production processes to reduce costs while maintaining quality could also contribute to lowering the unit variable cost.
The most critical trend that emerges from the analysis is the strong relationship between price and unit variable cost. As prices increase, unit variable costs also rise, suggesting that pricing strategies have a direct impact on production costs. This dynamic is essential to consider when developing strategies to reduce the unit variable cost. By understanding and leveraging these relationships, businesses can make informed decisions that drive operational improvements and ultimately reduce costs.
In conclusion, the analysis highlights the importance of considering the intricate relationships between price, quantity demanded, and unit variable cost. By recognizing these patterns and trends, businesses can develop targeted strategies to reduce the unit variable cost, ultimately improving profitability and competitiveness. The single most important trend that the user should not ignore is the strong positive correlation between price and unit variable cost, as this relationship has the most significant implications for reducing costs and driving business success.
"""

# 4. ESECUZIONE DEL TEST
print("Avvio generazione PDF di prova...")
create_pdf_report(mock_report, histogram_path=test_img_path, domain="Finance")
print(f"PDF generato con successo: 4μ_Test_Report.pdf")

# Pulizia (opzionale: rimuove il grafico di test dopo l'uso)
# os.remove(test_img_path)

#OLD VERSION BELOW FOR REFERENCE
#def create_pdf_report(report_text, histogram_path=None):
#    """Crea il PDF includendo l'istogramma"""
#    file_path = "4μ_report.pdf"
#    doc = SimpleDocTemplate(file_path, pagesize=letter, topMargin=40, bottomMargin=40, leftMargin=30, rightMargin=30)
#    styles = getSampleStyleSheet()
#    story = [Paragraph("4μ - Executive Report", styles['h1']), Spacer(1, 20)]
    
#    if not report_text: 
 #       report_text = "No report content."
    
    # Aggiungi paragrafi con maggiore spaziatura
  #  for line in report_text.split('\n'):
   #     if line.strip():
    #        story.append(Paragraph(line.strip(), styles['Normal']))
     #       story.append(Spacer(1, 10))
    
    # Aggiungi istogramma alla fine se presente
    #if histogram_path and Path(histogram_path).exists():
     #   story.append(Spacer(1, 25))
      #  story.append(Paragraph("Variable Distribution", styles['Heading2']))
       # story.append(Spacer(1, 15))
        #img = RLImage(histogram_path, width=5*inch, height=3.75*inch)
        #story.append(img)
    
   # doc.build(story)
   # return file_path