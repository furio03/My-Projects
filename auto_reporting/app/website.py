import pandas as pd
import gradio as gr
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white
import time

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import from the main.py engine
from main import run_analysis
from LLM.RAG import build_collection_from_reports, query_reports, get_pdf_files_from_folder

def create_pdf_report(report_text, histogram_path=None, domain_plots=None, domain=None):
    file_path = "4μ_Executive_Report.pdf"
    # Slightly wider margins for a more executive look
    doc = SimpleDocTemplate(file_path, pagesize=letter, topMargin=50, bottomMargin=50, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    
    # --- Color Palette ---
    primary_blue = HexColor('#0D47A1')

    # --- Styles ---
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
        fontName='Helvetica-Oblique',
        letterSpacing=1.5, # Add spacing for readability and elegance
        spaceAfter=15
    )

    body_style = ParagraphStyle(
        'ModernBody', parent=styles['Normal'], fontSize=10.5, leading=14,
        alignment=4, spaceAfter=12  # 4 = Justified
    )

    caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Normal'],
    fontSize=10,
    textColor=colors.HexColor('#555555'),
    alignment=1,
    spaceAfter=10,
    italic=True
)

    story = []

    # --- Clean Header ---
    domain_display = domain.replace('_', ' ').upper() if domain else "GENERAL"
    
    # Main title with 4u Executive Report
    story.append(Paragraph("4u Executive Report", title_style))
    
    # Domain subtitle (e.g. EDUCATION, FINANCE, etc.)
    story.append(Paragraph(domain_display, domain_subtitle_style))
    
    # Single, clean, professional line
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, hAlign='CENTER'))
    story.append(Spacer(1, 25))

    # --- Text Analysis ---
    paragraphs = report_text.split('\n\n')
    chart_inserted = False
    domain_plots_inserted = set()

    for p_text in paragraphs:
        p_text = p_text.strip()
        if p_text:
            story.append(Paragraph(p_text, body_style))
            
            # --- Chart Integration ---
            # Insert the chart within the statistical context
            if not chart_inserted and histogram_path and any(word in p_text.lower() for word in ["statistical", "variables", "correlation"]):
                if Path(histogram_path).exists():
                    story.append(Spacer(1, 15))
                    img = RLImage(histogram_path, width=5.2*inch, height=3.5*inch)
                    img.hAlign = 'CENTER'
                    story.append(img)
                    # Coordinated blue caption
                    caption_style = ParagraphStyle('Caption', parent= domain_subtitle_style, fontSize=9, spaceBefore=5)
                    story.append(Paragraph(f"Figure 1: {domain_display} Distribution Analysis", caption_style))
                    story.append(Spacer(1, 25))
                    chart_inserted = True

            # --- Domain Plot Insertion (NEW) ---
            if domain_plots and len(domain_plots) > 0:
                p_lower = p_text.lower()
                
                for plot_idx, plot_path in enumerate(domain_plots):
                    if plot_idx in domain_plots_inserted:
                        continue  # Already inserted, skip
                    
                    if not Path(plot_path).exists():
                        continue  # File does not exist, skip
                    
                    figure_num = plot_idx + 2  # +2 because Figure 1 is the histogram
                    
                    # Check whether the LLM mentions "Figure 2", "Figure 3", etc. in this paragraph
                    if f"figure {figure_num}" in p_lower:
                        story.append(Spacer(1, 15))
                        
                        # Insert image
                        img = RLImage(plot_path, width=6*inch, height=4*inch)
                        img.hAlign = 'CENTER'
                        story.append(img)
                        
                        # Build caption
                        caption_text = Path(plot_path).stem
                        # Remove numeric prefixes (01_, 02_, etc.)
                        caption_text = ''.join(c for c in caption_text if not c.isdigit()).strip('_- ')
                        caption_text = caption_text.replace('_', ' ').title()
                        caption_style = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=10,
                        textColor=colors.HexColor('#555555'), alignment=1, spaceAfter=10, italic=True)
                        
                        story.append(Paragraph(f"Figure {figure_num}: {caption_text}", caption_style))
                        story.append(Spacer(1, 25))
                        
                        # Mark as inserted
                        domain_plots_inserted.add(plot_idx)

# --- Appendix for Remaining Plots ---
    if domain_plots:
        remaining_plots = [
            (i, p) for i, p in enumerate(domain_plots) 
            if i not in domain_plots_inserted and Path(p).exists()
        ]
        
        if remaining_plots:
            story.append(Spacer(1, 30))
            story.append(HRFlowable(width="100%", thickness=1, color=primary_blue))
            story.append(Spacer(1, 20))
            
            appendix_title_style = ParagraphStyle(
                'AppendixTitle', parent=styles['h2'],
                fontSize=16, textColor=primary_blue, 
                fontName='Helvetica-Bold', spaceAfter=15
            )
            story.append(Paragraph("Additional Visualizations", appendix_title_style))
            story.append(Spacer(1, 15))
            
            for original_idx, plot_path in remaining_plots:
                img = RLImage(plot_path, width=6*inch, height=4*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                
                caption_text = Path(plot_path).stem
                caption_text = ''.join(c for c in caption_text if not c.isdigit()).strip('_- ')
                caption_text = caption_text.replace('_', ' ').title()
                
                figure_num = original_idx + 2
                story.append(Paragraph(f"Figure {figure_num}: {caption_text}", caption_style))
                story.append(Spacer(1, 20))
    
    # --- Footer Line ---
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=primary_blue, hAlign='CENTER'))
    
    doc.build(story)
    return file_path

def show_upload_progress(file):
    """Show a progress bar while a file is being uploaded."""
    if file is None:
        return ""
    
    progress_html = """
    <div style='width: 100%; background-color: rgba(255,255,255,0.15); border-radius: 8px; height: 6px; margin-top: 12px; overflow: hidden;'>
        <div style='width: 100%; height: 100%; background: linear-gradient(90deg, #10b981, #059669); animation: loading 1.5s ease-in-out;'></div>
    </div>
    <style>
        @keyframes loading {
            0% { width: 0%; }
            100% { width: 100%; }
        }
    </style>
    <p style='color: #10b981; font-size: 13px; margin-top: 8px; font-weight: 600;'>✓ File uploaded successfully</p>
    """
    return progress_html

def website_analyze(file, user_objective):
    if file is None: 
        return "", None
    
    try:
        data = run_analysis(file.name, user_objective=user_objective)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return f"<p style='color:#ef4444; font-size: 14px; font-weight: 600;'>Error: {str(e)}</p>", None
    
    # Elegant domain card
    domain_html = f"""
    <div style='display: inline-block; background: #ffffff; padding: 16px 32px; border-radius: 12px; 
                margin: 24px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.12);'>
        <span style='color: #64748b; font-size: 11px; font-family: sans-serif; text-transform: uppercase; 
                     letter-spacing: 1.2px; font-weight: 700; display: block; margin-bottom: 6px;'>RECOGNISED DOMAIN</span>
        <span style='color: #0f172a; font-size: 20px; font-family: sans-serif; font-weight: 800;'>{data['suffix'].upper()}</span>
    </div>
    """
    
    global analysis_data, visible_sections
    analysis_data = data
    visible_sections = set()
    
    return domain_html, None

def show_correlation_matrix(selected_view):
    """Show the analysis corresponding to the clicked button."""
    global analysis_data
    
    if not analysis_data:
        return ""
    
    html_out = ""
    
    if selected_view == "quant":
        if analysis_data.get('basic_quant') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>📊 Quantitative Statistics (Cleaned)</h3>"
            html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['basic_quant'].to_html(classes='summary-table') + "</div>"
            # Add outliers statistics if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('basic_quant_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ Quantitative Statistics (Outliers)</h3>"
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['basic_quant_outliers'].to_html(classes='summary-table') + "</div>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No quantitative data available</p>"
    
    elif selected_view == "qual":
        if analysis_data.get('basic_qual') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>📝 Qualitative Statistics (Cleaned)</h3>"
            html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['basic_qual'].T.to_html(classes='summary-table') + "</div>"
            # Add outliers statistics if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('basic_qual_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ Qualitative Statistics (Outliers)</h3>"
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['basic_qual_outliers'].T.to_html(classes='summary-table') + "</div>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No qualitative data available</p>"
    
    elif selected_view == "correlation":
        if analysis_data.get('corr_matrix') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>🔗 Correlation Matrix (Cleaned)</h3>"
            html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['corr_matrix'].to_html(classes='summary-table') + "</div>"
            # Add outliers correlation if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('corr_matrix_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ Correlation Matrix (Outliers)</h3>"
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['corr_matrix_outliers'].to_html(classes='summary-table') + "</div>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No correlation matrix available</p>"
    
    elif selected_view == "kmeans":
        if analysis_data.get('cluster_profiles') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>🎯 K-Means Cluster Profiles (Cleaned)</h3>"
            html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['cluster_profiles'].to_html(classes='summary-table') + "</div>"
            # Add outliers clusters if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('cluster_profiles_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ K-Means Cluster Profiles (Outliers)</h3>"
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['cluster_profiles_outliers'].to_html(classes='summary-table') + "</div>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No cluster profiles available</p>"
    
    elif selected_view == "kmedoids":
        if analysis_data.get('kmedoids_profiles') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>📍 K-Medoids Cluster Profiles (Cleaned)</h3>"
            html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['kmedoids_profiles'].to_html(classes='summary-table') + "</div>"
            # Add outliers clusters if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('kmedoids_profiles_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ K-Medoids Cluster Profiles (Outliers)</h3>"
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + analysis_data['kmedoids_profiles_outliers'].to_html(classes='summary-table') + "</div>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No K-medoids cluster profiles available</p>"
    
    elif selected_view == "chisquare":
        if analysis_data.get('chi_square_results') is not None:
            html_out += "<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>χ² Chi-Square Test Results (Cleaned)</h3>"
            chi_results = analysis_data['chi_square_results']
            if isinstance(chi_results, pd.DataFrame):
                if len(chi_results) > 0:
                    html_out += "<div style='overflow-x: auto; width: 100%;'>" + chi_results.to_html(classes='summary-table') + "</div>"
                else:
                    html_out += "<p style='color:white; font-size: 15px;'>No significant chi-square relationships found (p-value < 0.05)</p>"
            elif isinstance(chi_results, dict):
                for test_name, test_data in chi_results.items():
                    html_out += f"<h4 style='color:white; font-size: 18px; font-weight: 600;'>{test_name}</h4>"
                    if isinstance(test_data, pd.DataFrame):
                        if len(test_data) > 0:
                            html_out += "<div style='overflow-x: auto; width: 100%;'>" + test_data.to_html(classes='summary-table') + "</div>"
                        else:
                            html_out += "<p style='color:white; font-size: 15px;'>No significant results found</p>"
                    else:
                        html_out += f"<p style='color:white; font-size: 15px;'>{str(test_data)}</p>"
            else:
                html_out += f"<p style='color:white; font-size: 15px;'>{str(chi_results)}</p>"
            
            # Add outliers chi-square if available
            if analysis_data.get('has_outliers_analysis') and analysis_data.get('chi_square_results_outliers') is not None:
                html_out += "<h3 style='color:white; font-size: 22px; margin-top: 32px; margin-bottom: 16px; font-weight: 700;'>⚠️ χ² Chi-Square Test Results (Outliers)</h3>"
                chi_results_out = analysis_data['chi_square_results_outliers']
                if isinstance(chi_results_out, pd.DataFrame):
                    if len(chi_results_out) > 0:
                        html_out += "<div style='overflow-x: auto; width: 100%;'>" + chi_results_out.to_html(classes='summary-table') + "</div>"
                    else:
                        html_out += "<p style='color:white; font-size: 15px;'>No significant chi-square relationships found (p-value < 0.05)</p>"
                elif isinstance(chi_results_out, dict):
                    for test_name, test_data in chi_results_out.items():
                        html_out += f"<h4 style='color:white; font-size: 18px; font-weight: 600;'>{test_name}</h4>"
                        if isinstance(test_data, pd.DataFrame):
                            if len(test_data) > 0:
                                html_out += "<div style='overflow-x: auto; width: 100%;'>" + test_data.to_html(classes='summary-table') + "</div>"
                            else:
                                html_out += "<p style='color:white; font-size: 15px;'>No significant results found</p>"
                        else:
                            html_out += f"<p style='color:white; font-size: 15px;'>{str(test_data)}</p>"
                else:
                    html_out += f"<p style='color:white; font-size: 15px;'>{str(chi_results_out)}</p>"
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No chi-square results available</p>"
    
    elif selected_view == "domain":
        if analysis_data.get('result_specific') is not None:
            html_out += f"<h3 style='color:white; font-size: 22px; margin-bottom: 16px; font-weight: 700;'>🌐 Analysis Modules for {analysis_data['suffix'].upper()}</h3>"
            result = analysis_data['result_specific']
            
            if isinstance(result, dict) and len(result) > 0:
                # 1. BADGE KEY METRICS (show available analyses)
                html_out += "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; margin-bottom: 25px;'>"
                for key in result.keys():
                    clean_name = key.replace('_', ' ').upper()
                    html_out += f"""
                        <div style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); 
                                    padding: 8px 16px; border-radius: 20px; color: white; font-weight: 600; 
                                    font-size: 13px; letter-spacing: 0.5px;'>
                            {clean_name}
                        </div>
                    """
                html_out += "</div>"
                
                # 2. DISPLAY DETAILED RESULTS for each analysis
                for key, value in result.items():
                    section_title = key.replace('_', ' ').title()
                    
                    if isinstance(value, pd.DataFrame):
                        # Show DataFrame as table
                        html_out += f"<h4 style='color:white; font-size: 18px; margin-top: 25px; margin-bottom: 12px; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 8px;'>📊 {section_title}</h4>"
                        # Limit rows for display
                        display_df = value.head(15) if len(value) > 15 else value
                        html_out += "<div style='overflow-x: auto; width: 100%;'>" + display_df.to_html(classes='summary-table') + "</div>"
                        if len(value) > 15:
                            html_out += f"<p style='color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 5px;'>Showing 15 of {len(value)} rows</p>"
                    
                    elif isinstance(value, dict):
                        # Show dictionary as key-value pairs
                        html_out += f"<h4 style='color:white; font-size: 18px; margin-top: 25px; margin-bottom: 12px; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 8px;'>📈 {section_title}</h4>"
                        html_out += "<div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-top: 10px;'>"
                        for k, v in value.items():
                            formatted_key = k.replace('_', ' ').title()
                            html_out += f"<p style='color:white; font-size: 14px; margin: 8px 0;'><strong>{formatted_key}:</strong> {v}</p>"
                        html_out += "</div>"
                    
                    elif isinstance(value, list):
                        # Show list as comma-separated values
                        html_out += f"<h4 style='color:white; font-size: 18px; margin-top: 25px; margin-bottom: 12px; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 8px;'>📋 {section_title}</h4>"
                        html_out += f"<div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;'>"
                        html_out += f"<p style='color:white; font-size: 14px;'>{', '.join(str(item) for item in value[:10])}</p>"
                        if len(value) > 10:
                            html_out += f"<p style='color: rgba(255,255,255,0.6); font-size: 12px;'>... and {len(value) - 10} more</p>"
                        html_out += "</div>"
                    
                    elif isinstance(value, (int, float)):
                        # Show numeric value with formatting
                        html_out += f"""
                        <div style='display: inline-block; background: rgba(255,255,255,0.1); padding: 12px 20px; 
                                    border-radius: 8px; margin-top: 15px; margin-right: 10px;'>
                            <span style='color: rgba(255,255,255,0.7); font-size: 12px; display: block;'>{section_title}</span>
                            <span style='color: white; font-size: 20px; font-weight: 700;'>{value:,.2f}</span>
                        </div>
                        """
                    
                    elif isinstance(value, str):
                        # Show string value
                        html_out += f"<h4 style='color:white; font-size: 18px; margin-top: 25px; margin-bottom: 12px; font-weight: 600;'>💡 {section_title}</h4>"
                        html_out += f"<p style='color:white; font-size: 14px; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;'>{value}</p>"
            
            elif isinstance(result, pd.DataFrame):
                html_out += "<div style='overflow-x: auto; width: 100%;'>" + result.to_html(classes='summary-table') + "</div>"
            
            # FALLBACK for other types
            else:
                html_out += f"""
                <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-top:10px;'>
                    <p style='color:white; font-size: 15px; margin: 0;'>
                        <strong>Note:</strong> Standard metrics calculated: {str(result)}
                    </p>
                </div>
                """
        else:
            html_out = "<p style='color:white; font-size: 15px;'>No domain-specific analysis available yet for this sector.</p>"
    
    return html_out

def get_histogram():
    """Return the histogram path."""
    global analysis_data
    if analysis_data and analysis_data.get('histogram_path'):
        hist_path = analysis_data['histogram_path']
        if isinstance(hist_path, str) and Path(hist_path).exists() and Path(hist_path).is_file():
            return hist_path
    return None

def toggle_analysis(button_name):
    """Toggle analysis visualization on/off."""
    global visible_sections, analysis_data
    
    if not analysis_data:
        return ""
    
    if button_name in visible_sections:
        visible_sections.remove(button_name)
        return ""
    
    visible_sections.clear()
    visible_sections.add(button_name)
    result = show_correlation_matrix(button_name)
    return result

def toggle_histogram():
    """Toggle histogram visualization on/off."""
    global visible_sections, analysis_data
    
    if not analysis_data:
        return gr.update(visible=False)
    
    if "hist" in visible_sections:
        visible_sections.remove("hist")
        return gr.update(value=None, visible=False)
    
    visible_sections.add("hist")
    hist = get_histogram()
    return gr.update(value=hist, visible=True)

analysis_data = {}
visible_sections = set()
rag_collection = None
rag_report_names = []

def website_report(file, user_objective):
    if file is None: 
        return None
    
    data = run_analysis(file.name, user_objective=user_objective)
    histogram_path = data.get('histogram_path', None)
    return create_pdf_report(data['report'], histogram_path, domain_plots=data.get('domain_plots', []), domain=data['suffix'])

# --- Full-Screen Professional Enterprise CSS ---
custom_css = """
/* Full-screen corporate blue background */
html, body, .gradio-container { 
    background-color: #1a365d !important; 
    margin: 0 !important; 
    padding: 0 !important;
    width: 100vw !important; 
    max-width: 100vw !important; 
    min-height: 100vh !important; 
    overflow-x: hidden !important;
}
.gradio-container { 
    padding: 50px 80px !important; 
    max-width: none !important;
}

h1, h3 { 
    color: white !important; 
    font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
}

/* Professional white buttons */
.gr-button { 
    background-color: #ffffff !important; 
    color: #0f172a !important; 
    border-radius: 10px !important; 
    border: none !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
    transition: all 0.2s ease !important;
    font-family: 'Helvetica Neue', sans-serif !important;
}
.gr-button:hover { 
    background-color: #f8fafc !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.18) !important;
    transform: translateY(-1px) !important;
}

/* Elegant file upload box */
.gr-file { 
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
.gr-file:hover {
    border-color: #cbd5e1 !important;
}

/* Clean white labels */
.gr-file label {
    color: white !important;
    background-color: transparent !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* File drop area */
.gr-file .file-preview,
.gr-file .upload-container {
    background-color: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Rounded professional tables */
.summary-table { 
    width: 100%; 
    background-color: #ffffff !important; 
    color: #0f172a !important; 
    border-collapse: separate !important;
    border-spacing: 0 !important;
    margin-bottom: 24px;
    table-layout: auto !important;
    word-wrap: break-word !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

.summary-table th { 
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    padding: 16px 18px !important;
    text-align: left;
    border-bottom: 2px solid #e2e8f0 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.summary-table td { 
    padding: 14px 18px !important; 
    color: #334155 !important; 
    text-align: left;
    font-size: 14px !important;
    max-width: 250px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

.summary-table tr:hover {
    background-color: #f8fafc !important;
}

.summary-table tr:last-child td {
    border-bottom: none !important;
}

/* Rounded images */
img {
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
"""

with gr.Blocks(title="4μ") as iface:
    iface.css = custom_css
    iface.theme = gr.themes.Base()
    
    gr.Markdown("<h1 style='text-align:center; font-size: 72px; color:white; margin-bottom:8px; font-weight: 700;'>4μ</h1>")
    gr.Markdown("<p style='text-align:center; color:rgba(255,255,255,0.95); font-size: 22px; margin-top:0; font-weight: 500;'>Deep insights, Smarter decisions</p>")
    
    file_input = gr.File(label="Upload CSV")
    upload_progress = gr.HTML()
    
    gr.Markdown("<h3 style='color:white; margin-top: 30px; margin-bottom: 10px; font-size: 18px; font-weight: 600;'>🎯 What would you like to analyze?</h3>")
    user_objective_input = gr.Textbox(
        label="Analysis Objective",
        placeholder="Describe your analysis goal... (e.g., 'I want to understand what factors influence house prices' )",
        lines=2,
        max_lines=4
    )
    
    analyze_btn = gr.Button("Analyze Dataset")
    domain_cell = gr.HTML()
    
    gr.Markdown("<h3 style='color:white; margin-top: 40px; font-size: 24px; font-weight: 700;'>📊 Analysis Results</h3>")
    
    quant_btn = gr.Button("📊 Quantitative Statistics", visible=False)
    quant_output = gr.HTML()
    
    qual_btn = gr.Button("📝 Qualitative Statistics", visible=False)
    qual_output = gr.HTML()
    
    hist_btn = gr.Button("📈 Distribution Chart", visible=False)
    hist_output = gr.Image(type="filepath", visible=False)
    
    corr_btn = gr.Button("🔗 Correlation Matrix", visible=False)
    corr_output = gr.HTML()
    
    kmeans_btn = gr.Button("🎯 K-Means Profiles", visible=False)
    kmeans_output = gr.HTML()
    
    kmedoids_btn = gr.Button("📍 K-Medoids Profiles", visible=False)
    kmedoids_output = gr.HTML()
    
    chi2_btn = gr.Button("χ² Chi-Square Test", visible=False)
    chi2_output = gr.HTML()
    
    domain_btn = gr.Button("🌐 Domain Analysis", visible=False)
    domain_output = gr.HTML()
    
    gr.Markdown("<div style='margin-top: 50px;'></div>")
    report_btn = gr.Button("Generate and Download Report")
    report_output_file = gr.File(label="Download Report")

    file_input.change(fn=show_upload_progress, inputs=file_input, outputs=upload_progress)
    
    def analyze_and_show_buttons(file, user_objective):
        result = website_analyze(file, user_objective)
        return result[0], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    
    analyze_btn.click(
        fn=analyze_and_show_buttons, 
        inputs=[file_input, user_objective_input], 
        outputs=[domain_cell, quant_btn, qual_btn, hist_btn, corr_btn, kmeans_btn, kmedoids_btn, chi2_btn, domain_btn]
    )
    
    quant_btn.click(fn=lambda: toggle_analysis("quant"), outputs=quant_output, show_progress=False)
    qual_btn.click(fn=lambda: toggle_analysis("qual"), outputs=qual_output, show_progress=False)
    hist_btn.click(fn=toggle_histogram, outputs=hist_output, show_progress=False)
    corr_btn.click(fn=lambda: toggle_analysis("correlation"), outputs=corr_output, show_progress=False)
    kmeans_btn.click(fn=lambda: toggle_analysis("kmeans"), outputs=kmeans_output, show_progress=False)
    kmedoids_btn.click(fn=lambda: toggle_analysis("kmedoids"), outputs=kmedoids_output, show_progress=False)
    chi2_btn.click(fn=lambda: toggle_analysis("chisquare"), outputs=chi2_output, show_progress=False)
    domain_btn.click(fn=lambda: toggle_analysis("domain"), outputs=domain_output, show_progress=False)
    report_btn.click(fn=website_report, inputs=[file_input, user_objective_input], outputs=report_output_file)

    # ── Report Intelligence Section (RAG) ──
    gr.Markdown("<div style='margin-top: 80px; border-top: 2px solid rgba(255,255,255,0.15); padding-top: 50px;'></div>")
    gr.Markdown("<h1 style='text-align:center; font-size: 36px; color:white; margin-bottom:8px; font-weight: 700;'>Report Intelligence</h1>")
    gr.Markdown("<p style='text-align:center; color:rgba(255,255,255,0.8); font-size: 16px; margin-top:0; font-weight: 400;'>Upload your generated reports and ask questions — powered by RAG</p>")

    report_upload = gr.File(
        label="Upload Report PDFs",
        file_count="multiple",
        file_types=[".pdf"],
    )
    report_index_btn = gr.Button("Index Reports")
    report_upload_status = gr.HTML()

    report_question = gr.Textbox(
        label="Ask a question about your reports",
        placeholder="e.g. What were the main findings on cost changes?",
        lines=2,
        max_lines=4,
        interactive=False,
    )
    report_ask_btn = gr.Button("Ask", interactive=False)
    report_answer = gr.Markdown()

    def ingest_reports(files):
        """Process uploaded PDF reports and build the RAG index."""
        global rag_collection, rag_report_names
        if not files:
            rag_collection = None
            rag_report_names = []
            return (
                "<p style='color:#ef4444; font-size:14px;'>Please upload at least one PDF report.</p>",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        pdf_paths = [f.name for f in files if f.name.endswith(".pdf")]
        if not pdf_paths:
            return (
                "<p style='color:#ef4444; font-size:14px;'>No valid PDF files found.</p>",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        rag_collection = build_collection_from_reports(pdf_paths)
        rag_report_names = [Path(p).name for p in pdf_paths]

        names_html = "".join(
            f"<span style='background:rgba(255,255,255,0.12); padding:6px 14px; border-radius:16px; "
            f"color:white; font-size:13px; font-weight:600; margin:4px;'>{n}</span>"
            for n in rag_report_names
        )
        status_html = (
            f"<div style='margin-top:14px;'>"
            f"<p style='color:#10b981; font-size:14px; font-weight:600;'>"
            f"✓ {len(pdf_paths)} report(s) indexed successfully</p>"
            f"<div style='display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;'>{names_html}</div>"
            f"</div>"
        )
        return (
            status_html,
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def ask_reports(question):
        """Query the RAG index with the user's question."""
        global rag_collection
        if rag_collection is None:
            return "Please index your reports first."
        if not question or not question.strip():
            return "Please enter a question."
        return query_reports(rag_collection, question.strip())

    report_index_btn.click(
        fn=ingest_reports,
        inputs=report_upload,
        outputs=[report_upload_status, report_question, report_ask_btn],
    )
    report_ask_btn.click(
        fn=ask_reports,
        inputs=report_question,
        outputs=report_answer,
    )

iface.launch(share=True)