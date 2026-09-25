import numpy as np
from diffprivlib.mechanisms import Laplace

STRICT_PRIVACY_DOMAINS = {
    'health',              
    'finance',             
    'hr',                  
    'insurance',           
    'security',           
}

MODERATE_PRIVACY_DOMAINS = {
    'business_economics',
    'demographics',
    'ecommerce',
    'education',
    'energy',
    'environment',
    'industrial',
    'logistics',
    'macro_economics',
    'marketing',
    'politics',
    'real_estate',
    'retail',
    'social_media',
    'sports',
    'supply_chain',
    'surveys',
    'telecommunications',
    'tourism',
}

PUBLIC_DOMAINS = {
    'agriculture',
    'scientific_research',
}


def should_apply_dp(domain):
    """
    Determines whether to apply DP and the privacy level for the domain.
    
    Higher epsilon values = moderate privacy but usable results for the report.
    
    Args:
        domain (str): Name of the analysis domain
    
    Returns:
        tuple: (apply_dp: bool, epsilon: float, level: str)
               - apply_dp: True if to apply DP
               - epsilon: Recommended privacy budget
               - level: strict, moderate, light, 'none'

    """
    domain_lower = domain.lower().replace(' ', '_')
    
    if domain_lower in STRICT_PRIVACY_DOMAINS:
        return True, 3.0, 'Strict'      # High privacy
    elif domain_lower in MODERATE_PRIVACY_DOMAINS:
        return True, 5.0, 'Moderate'    # Medium privacy
    #elif domain_lower in PUBLIC_DOMAINS:
    #    return True, 7.0, 'Light'       # Light privacy
    else:
        return False, None, 'none'      # No DP


def protect_data(results, domain, epsilon=None, delta=0):
    """
    Applies Global Differential Privacy to results based on domain.
    
    Uses Laplace mechanism with calibrated sensitivity.
    Maintains balance between privacy and accuracy for business reports.
    
    Args:
        results (dict): Dictionary with domain analysis results
        domain (str): Domain name (e.g., health, finance, agriculture)
        epsilon (float): Privacy budget (if None, use recommended value for domain)
        delta (float): Failure probability (default 0 for Laplace)
    
    Returns:
        dict: Dictionary with protected results (controlled noise)
    
    Example:
        results = {mean: 50000, median: 48000}
        protected = protect_data(results, domain= health)
    """
    
    # Determine whether to apply DP and get the appropriate epsilon and privacy level
    apply_dp, default_epsilon, privacy_level = should_apply_dp(domain)
    
    if not apply_dp:
        return results
    
    # Use the provided epsilon or the default one based on the domain
    final_epsilon = epsilon if epsilon is not None else default_epsilon
    
    protected_results = {}
    
    for key, value in results.items():
        # Protect only numeric values (int, float) and ignore booleans
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Intelligent sensitivity: based on percentage of value
            # This reduces relative noise for large values
            # e.g.: for a value of 1000, sensitivity = 100 (10% of value)
            # instead of 1000 (100% of value)
            if value != 0:
                sensitivity = abs(value) * 0.10  # 10% del valore come range
            else:
                sensitivity = 1.0
            
            # Laplace mechanism with calibrated sensitivity
            dp_mechanism = Laplace(epsilon=final_epsilon, sensitivity=sensitivity)
            protected_results[key] = dp_mechanism.randomise(value)
        else:
            # Non numeric values are returned unchanged 
            protected_results[key] = value
    
    # Add metadata about privacy level applied (for transparency in report)
    #protected_results['_dp_applied'] = True
    protected_results['_privacy_level_applied'] = privacy_level
    #protected_results['_epsilon'] = final_epsilon
    
    return protected_results
    
   
