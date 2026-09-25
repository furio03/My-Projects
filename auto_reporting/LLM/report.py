import pandas as pd
import requests
from statistics_functions.clustering import auto_kmeans, auto_kmedoids
from statistics_functions.chi_square import chi_square_test
from statistics_functions.graphs import generate_plot_description_for_llm
from LLM.keys import GROQ_API_KEY

def analizza_dataset(quant_results: None, qual_results: None, date_results: None, corr_matrix: None, 
                    full_df: None, domain:None, graphs: None, domain_plots: None = None, user_objective: str = None, user_language: str = "en") -> str:
    """
    Analizza un dataset usando Groq API (gratis).
    
    Args:
        quant_results: Statistiche quantitative
        qual_results: Statistiche qualitative  
        date_results: Statistiche temporali
        corr_matrix: Matrice di correlazione
        full_df: DataFrame completo
        domain: Analisi di dominio
        graphs: Percorso ai grafici
        domain_plots: Elenco dei percorsi ai grafici specifici del dominio
        user_objective: Obiettivo dell'analisi specificato dall'utente
        user_language: Lingua in cui scrivere il report (es: 'it', 'en', 'es')
    
    Returns:
        str: Risposta dell'LLM
    """
    # Language instruction
    language_instruction = f"IMPORTANT: Write the entire report in {user_language.upper()} language. All text must be in this language."
    
    # User objective context
    objective_context = ""
    if user_objective and user_objective.strip():
        objective_context = f"""\n\nUSER'S ANALYSIS OBJECTIVE:\n\"{user_objective}\"\n\nFocus your analysis primarily on addressing this specific objective. 
All insights and recommendations should be relevant to what the user wants to understand or achieve."""
    
    # Prepare the prompt
    instructions = f"""Act as a senior data strategist. Your goal is to extract actionable intelligence from the provided results, prioritizing the Domain-Specific Analysis: {domain if domain is not None else "General Analysis"} after using given descriptive statistics as introduction. 
    Instead of merely describing numbers, identify the 'so what' behind the data. Do not settle for a brief summary; explore every investigated module in depth, dedicating extensive commentary to each specific domain analysis provided. 
    Translate technical statistical concepts into practical evidence (e.g., translate p-values into 'certainty of patterns' and coefficients into 'strength of relationships'), making the report valuable for both experts and non-technical users."""

    details = """Your response must be a formal, structured, and highly detailed essay capable of spanning multiple pages. Use broad, well-developed paragraphs connected by sophisticated transitional phrases to ensure a logical and engaging narrative flow. 
    Avoid any form of bullet points or numbered lists if not necessary. Integrate column names and technical terms naturally into your professional prose without using bold markdown (**), maintaining an authoritative yet readable tone. 
    Explore every section—statistics, correlations, clusters, and domain modules—with the intent of guiding real-world decisions. Do not limit yourself to a single page if the data offers insights for further exploration."""

    last_info = """The final output must be visionary, highlighting the single most important trend or anomaly the user should not ignore. 
    Ensure the summary is intuitive and directly linked to the dataset's implied reality. Avoid excessive jargon; use clear language that focuses on outcomes and strategic relationships. Write in an easy-to-read yet commanding style."""
    
    # Format domain-specific analysis results for better LLM interpretation
    domain_analysis_text = ""
    if domain is not None and isinstance(domain, dict):
        domain_analysis_text = "\n### DOMAIN-SPECIFIC ANALYSIS RESULTS ###\n"
        for key, value in domain.items():
            section_name = key.replace('_', ' ').title()
            domain_analysis_text += f"\n--- {section_name} ---\n"
            
            if isinstance(value, pd.DataFrame):
                # Convert DataFrame to readable string format
                domain_analysis_text += value.to_string() + "\n"
            elif isinstance(value, dict):
                # Format dict as key-value pairs
                for k, v in value.items():
                    domain_analysis_text += f"  {k}: {v}\n"
            elif isinstance(value, list):
                # Format list items
                domain_analysis_text += f"  {', '.join(str(item) for item in value)}\n"
            else:
                # Simple value
                domain_analysis_text += f"  {value}\n"
    elif domain is not None:
        domain_analysis_text = f"\n### DOMAIN-SPECIFIC ANALYSIS ###\n{str(domain)}\n"
    
    domain_plots_text = ""
    if domain_plots and len(domain_plots) > 0:
        domain_plots_text = generate_plot_description_for_llm(domain_plots)
    
    # prepare dataset info
    info = f"""
{language_instruction}

{instructions}
{details}
{objective_context}

{domain_analysis_text if domain_analysis_text else "No domain-specific analysis available."}

{domain_plots_text}

Datasets, some may be empty or without relevant information:
{"" if quant_results is None else quant_results.to_string() + "Correlation matrix:" + corr_matrix.to_string()}
{"" if qual_results is None else qual_results.to_string()}
{"" if date_results is None else date_results.to_string()}

graphs related to variable distributions (if any):
{"" if graphs is None else graphs}
do not talk about graphs that are not present.

K-means clustering profiles (if not empty):
{"" if quant_results is None else auto_kmeans(full_df[quant_results.columns]).to_string()}
Recognise interesting variables and comment on them if there are any significant values in the profiles of the clusters.
Your comments should give non technical insights related to the context of the dataset.

K-medoids clustering profiles (if not empty):
{"" if qual_results is None else auto_kmedoids(full_df[qual_results.columns]).to_string()}
Recognise interesting variables and comment on them if there are any significant values in the profiles of the clusters.
Your comments should give non technical insights related to the context of the dataset.

Quantitative dataset columns: {"" if quant_results is None else quant_results.columns.tolist()}
Qualitative dataset columns: {"" if qual_results is None else qual_results.columns.tolist()}
Temporal dataset columns: {"" if date_results is None else date_results.columns.tolist()}

Chi-Square test results for categorical variables (if present) with pairwise significance p-values for qualitative variables:
{"" if qual_results is None else chi_square_test(full_df[qual_results.columns]).to_string()}

{last_info}
"""
    
    # Chiamata Groq API
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": info}]
        }
    )
    
    return response.json()['choices'][0]['message']['content']


def analizza_outliers(cleaned_report: str, basic_quant_outliers: None, basic_qual_outliers: None, 
                     basic_date_outliers: None, corr_matrix_outliers: None, df_outliers: None, 
                     profiles_outliers: None, chi_square_results_outliers: None, domain_results: None = None, user_language: str = "en") -> str:
    """
    Analizza gli outliers in modo comparativo rispetto al dataset cleaned.
    
    Args:
        cleaned_report: Report già generato per il dataset cleaned
        basic_quant_outliers: Statistiche quantitative degli outliers
        basic_qual_outliers: Statistiche qualitative degli outliers
        basic_date_outliers: Statistiche temporali degli outliers
        corr_matrix_outliers: Matrice di correlazione degli outliers
        df_outliers: DataFrame con solo outliers
        profiles_outliers: Profili K-means degli outliers
        chi_square_results_outliers: Risultati Chi-Square degli outliers
        domain_results: Risultati dell'analisi di dominio specifica per gli outliers (se disponibile)
        user_language: Lingua in cui scrivere il report
    
    Returns:
        str: Analisi comparativa degli outliers
    """
    
    # Language instruction
    language_instruction = f"IMPORTANT: Write the entire analysis in {user_language.upper()} language. All text must be in this language."
    
    # Istruzioni specifiche per l'analisi degli outliers
    instructions = """Act as a senior anomaly detection analyst. Your goal is to identify what makes these outliers fundamentally different from the main dataset. 
    You have already analyzed the cleaned dataset (see the CLEANED DATASET REPORT below). Now, focus on understanding the outlier population: What distinguishes them? 
    Are they extreme cases of normal behavior or do they represent entirely different patterns? Look for inversions in correlations, different cluster behaviors, 
    or categorical distributions that diverge from the norm."""
    
    details = """Your response must be a formal, structured essay. Start with a brief comparison summary, then dive into specific differences:
    - How do means/medians differ? Are outliers consistently higher/lower or just more variable?
    - Are correlations inverted or weakened compared to the main dataset?
    - Do clusters in outliers represent different behavioral segments?
    - Are there qualitative categories over/under-represented in outliers?
    
    Do not use bold markdown (**). Write in flowing paragraphs with appropriate spacing. Focus on actionable insights about what makes these observations anomalous."""
    
    last_info = """Conclude with strategic implications: Should these outliers be investigated individually? Do they represent risks, opportunities, or simply edge cases? 
    What business actions should be taken regarding these anomalous observations? Be specific and practical."""
    
    # Prepare outliers data
    outliers_data = f"""
=== CLEANED DATASET REPORT (for comparison) ===
{cleaned_report}

=== OUTLIERS ANALYSIS ===
Number of outliers detected: {len(df_outliers) if df_outliers is not None else 0}

Outliers Quantitative Statistics:
{"" if basic_quant_outliers is None else basic_quant_outliers.to_string()}

Outliers Correlation Matrix:
{"" if corr_matrix_outliers is None else corr_matrix_outliers.to_string()}

Outliers Qualitative Statistics:
{"" if basic_qual_outliers is None else basic_qual_outliers.to_string()}

Outliers Temporal Statistics:
{"" if basic_date_outliers is None else basic_date_outliers.to_string()}

Outliers K-means Clustering Profiles:
{"" if profiles_outliers is None else profiles_outliers.to_string()}
Comment on how outlier clusters differ from main dataset clusters.

Outliers Chi-Square Test Results:
{"" if chi_square_results_outliers is None else chi_square_results_outliers.to_string()}
"""
    
    prompt = f"""
{language_instruction}

{instructions}
{details}

{outliers_data}

{last_info}

IMPORTANT: Your analysis should highlight DIFFERENCES and CONTRASTS with the cleaned dataset. This is not a standalone analysis but a comparative one.
"""
    
    # Chiamata Groq API
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    
    return response.json()['choices'][0]['message']['content']



# OLD PROMPT FOR REFERENCE
# instructions="Commenta i risultati mettendo in relazione, quando possibile, i valori ottenuti con il contesto del dataset.  Puoi capire il contesto del dataset guardando le prime righe e i nomi delle colonne. Presenta alla fine un breve sommario delle osservazioni più rilevanti emerse dall'analisi senza utilizzare un linguaggio tecnico."
#details="non parlare di errori legati alla qualità dei dati. Fornisci osservazioni significative e pertinenti basate sui dati presentati. Non usare ** per i nomi delle colonne. Per l'elenco usa 1), 2),...."
#last_info="I commenti devono essere non tecnici per essere comprensibili a un pubblico generale e legati al contesto del dataset, ogni commento deve essere pertinente."
