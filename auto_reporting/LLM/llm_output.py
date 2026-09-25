import pandas as pd
import requests
import json
import ast
from data_manipulation.read_data import read, handle_Nans, correct_dtypes, identify_variables
from analystics.data_analyis import basic_analytics_quant, basic_analytics_qual, basic_analytics_date
from LLM.keys import GROQ_API_KEY

def recognise_domain( df : pd.DataFrame) -> str:
    """
    gives the domain of the datset using Groq API 
    """
    domains=['agriculture', 'business_economics', 'demographic', 'ecommerce','education', 'energy', 'environment',
             'finance', 'health', 'hr', 'industrial', 'insurance', 'logistics', 'macro_economics', 
             'politics', 'real_estate', 'retail', 'scientific_research','security','social_media',
             'sports', 'supply_chain', 'surveys', 'telecommunications', 'tourism']
    
    with open('knowledge_base.txt', 'r') as f:
        knowledge_base = f.read()
    
    # 2. Prepariamo l'evidenza statistica dai tuoi dati esistenti
    # Usiamo json.dumps per garantire che Groq legga correttamente i dizionari
    dict_vars = identify_variables(df)
    basic_quant = basic_analytics_quant(df, dict_vars)
    basic_qual = basic_analytics_qual(df, dict_vars)
    
    evidence = {
        "variable_types": dict_vars,
        # Convertiamo i DataFrame delle statistiche in dizionari semplici
        "numeric_summary": basic_quant.to_dict() if hasattr(basic_quant, 'to_dict') else basic_quant,
        "categorical_samples": basic_qual.to_dict() if hasattr(basic_qual, 'to_dict') else basic_qual
    }
    # Usiamo default=str per gestire qualsiasi residuo strano (NaN, Timestamp, ecc.)
    evidence_json = json.dumps(evidence, indent=2, default=str)

    info = f"""
    OBJECTIVE: Classify the dataset into one of these 26 domains: {domains}

    ### REFERENCE PROTOCOL (Domain Archetypes & Disambiguation):
    {knowledge_base}

    ### DATASET EVIDENCE:
    - METADATA & STATS: {evidence_json}
    - COLUMN NAMES: {df.columns.tolist()}

    ### INSTRUCTIONS:
    1. ANALYZE the 'numeric_stats' (min, max, mean). Look for specific ranges (e.g., Likert 1-5, percentages 0-1, or large financial floats).
    2. ANALYZE the 'categorical_samples' to understand the semantic context of the values.
    3. CROSS-REFERENCE the findings with the 'CRITICAL DISAMBIGUATION MATRIX' in the Reference Protocol.
    4. HIERARCHY OF TRUTH: If column names suggest one domain but statistical signatures (ranges, types) strongly match another archetype in the protocol, prioritize the archetype's signature.
    
    Final Output: Return ONLY the domain name from the provided list. No prose.
   
Columns: {df.columns.tolist()}
Sample: {df.head(5).to_string()}"""
    
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
    
    # Check if response is valid
    response_json = response.json()
    
    if 'choices' not in response_json:
        print(f"Warning: API response error - {response_json}")
        # Fallback: return a default domain
        return "business_economics"
    
    return response_json['choices'][0]['message']['content']




def needed_variables(df: pd.DataFrame,list_needed_variables: list)->dict :
    """return a dict to identify a variable with a standard name"""
    info = f"""Map these standard names {list_needed_variables} to dataset columns.
Output ONLY JSON: {{"standard_name": "dataset_column", ...}}

Columns: {df.columns.tolist()}
Sample: {df.head(5).to_string()}"""
    
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

    # Check if response is valid
    response_json = response.json()
    
    if 'choices' not in response_json:
        print(f"Warning: API response error - {response_json}")
        # Fallback: return empty dict
        return {}
    
    content = response_json['choices'][0]['message']['content']
    
    # Clean JSON if wrapped in markdown
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content.split('\n', 1)[1].rsplit('\n', 1)[0]
    if content.endswith("```"):
        content = content[:-3]
    
    try:
        dictionary = json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback: try ast.literal_eval
        try:
            dictionary = ast.literal_eval(content.strip())
        except (ValueError, SyntaxError):
            # If all parsing fails, return empty dict
            print(f"Warning: Could not parse LLM response: {content[:200]}")
            dictionary = {}
    
    return dictionary


def identify_target_variable(df: pd.DataFrame, user_objective: str) -> dict:
    """
    Identifica la variabile target per la regressione basandosi sull'obiettivo dell'utente.
    Rileva anche la lingua utilizzata dall'utente.
    
    Args:
        df: DataFrame pandas
        user_objective: Stringa con l'obiettivo dell'analisi fornito dall'utente
    
    Returns:
        dict: {'target_variable': nome_colonna, 'language': lingua_rilevata}
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    instructions = f"""User objective: "{user_objective}"
Numeric columns: {numeric_cols}
Identify best target variable for regression and detect user's language.
Output ONLY: {{"target_variable": "column_name", "language": "ISO_code"}}"""
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": instructions}]
        }
    )
    
    # Check if response is valid
    response_json = response.json()
    
    if 'choices' not in response_json:
        print(f"Warning: API response error - {response_json}")
        # Fallback: usa l'ultima colonna numerica e assume inglese
        return {"target_variable": numeric_cols[-1] if numeric_cols else None, "language": "en"}
    
    content = response_json['choices'][0]['message']['content']
    
    # Clean JSON if wrapped in markdown
    if content.startswith("```"):
        content = content.split('\n', 1)[1].rsplit('\n', 1)[0]
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    try:
        result = json.loads(content.strip())
    except:
        # Fallback: usa l'ultima colonna numerica e assume inglese
        result = {"target_variable": numeric_cols[-1] if numeric_cols else None, "language": "en"}
    
    # Verifica che la variabile esista
    if result.get('target_variable') not in numeric_cols:
        result['target_variable'] = numeric_cols[-1] if numeric_cols else None
    
    return result
    

#old prompt for reference: nstructions="Osserva le prime righe del dataset e i nomi delle colonne e identifica il dominio di appartenenza del dataset identificandondolo tra quelli nella lista fornita"+str(domains)



