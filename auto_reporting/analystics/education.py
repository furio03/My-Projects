import pandas as pd
import numpy as np
from LLM.llm_output import needed_variables

def analytics_education(df: pd.DataFrame, user_objective: str = None)->dict:
    """
    Perform analytics on education data.

    """
    list_variables=['età']
    dict_vars=needed_variables(df,list_variables)
    mean_age=np.mean(df[dict_vars['età']])

    

    return {
        'mean_age': float(mean_age)
    }