import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chi_square_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Chi-Square tests for independence on all pairs of categorical variables in the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe containing categorical variables.
        
    Returns:
        pd.DataFrame: A dataframe containing pairs of variables and their corresponding p-values.
    """
    results = []

    # Identify categorical columns (object dtype)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # If no categorical columns, return empty dataframe
    if len(categorical_cols) < 2:
        return pd.DataFrame(columns=['Variable 1', 'Variable 2', 'p-value'])

    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            try:
                # Create a contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])
                
                # Perform Chi-Square test
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                
                results.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'p-value': p
                })
            except Exception as e:
                print(f"Warning: Could not perform chi-square test for {col1} and {col2}: {str(e)}")
                continue
    
    # Create dataframe from results or return empty dataframe if no results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['p-value'] < 0.05]
        results_df = results_df.sort_values(by='p-value').reset_index(drop=True)
    else:
        results_df = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'p-value'])

    return results_df

