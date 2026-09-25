import pandas as pd
import numpy as np
from scipy import stats

def handle_outliers(df: pd.DataFrame, alpha_jackknife: float = 2.5, alpha_cooks: float = None, target_variable: str = None) -> dict:
    """ Detects and handles outliers in a DataFrame using Rstudentized Jackknife method and Cook's Distance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with numeric columns
    alpha_jackknife : float
        Threshold for Rstudentized Jackknife method (default: 2.5, |t| > 2.5 for outliers)
    alpha_cooks : float
        Threshold for Cook's Distance (default: 4/n rule of thumb)
    target_variable : str
        Name of the target variable for regression. If None, uses the last numeric column.
    
    Returns three versions of the DataFrame:
        - df_flagged: Original DataFrame with 'outlier_type' column:
            * 'Outlier': detected by Jackknife only
            * 'Influent': detected by Cook's Distance only
            * 'Critical': detected by both methods
            * np.nan: not an outlier
        - df_no_outliers: DataFrame with all outliers removed
        - df_only_outliers: DataFrame with only outliers"""
    
    # Copy DataFrame to avoid modifying the original
    df = df.copy()
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ===== FALLBACK FOR DATASETS WITH INSUFFICIENT NUMERIC DATA =====
    if len(numeric_cols) == 0:
        # NO numeric columns at all - return original data without flagging
        print("ℹ️  No numeric columns found - outlier detection skipped (qualitative-only dataset)")
        df_flagged = df.copy()
        # Don't add outlier_type column for fully qualitative datasets
        return {
            'df_flagged': df_flagged,
            'df_no_outliers': df.copy(),
            'df_only_outliers': pd.DataFrame(columns=df.columns)  # Empty with same structure
        }
    
    if len(numeric_cols) == 1:
        # Only 1 numeric column - can't do regression-based outlier detection
        print(f"ℹ️  Only 1 numeric column found ({numeric_cols[0]}) - using simple IQR method for outlier detection")
        
        col = numeric_cols[0]
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # IQR method: outliers are beyond 1.5*IQR from Q1/Q3
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Create flagged DataFrame
        df_flagged = df.copy()
        df_flagged['outlier_type'] = np.where(outlier_mask, 'Outlier', np.nan)
        
        # DataFrames
        df_cleaned = df[~outlier_mask].reset_index(drop=True)
        df_outliers = df[outlier_mask].reset_index(drop=True)
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"   → Detected {n_outliers} outliers in {col} using IQR method")
        
        return {
            'df_flagged': df_flagged,
            'df_no_outliers': df_cleaned,
            'df_only_outliers': df_outliers
        }
    
    # ===== NORMAL FLOW: 2+ NUMERIC COLUMNS =====
    # Determine target variable
    if target_variable is not None and target_variable in numeric_cols:
        # Riordina le colonne mettendo la target alla fine
        other_cols = [col for col in numeric_cols if col != target_variable]
        numeric_cols = other_cols + [target_variable]
    # else: usa l'ordine originale (ultima colonna come target)
    
    n = len(df)
    p = len(numeric_cols) - 1  # number of predictors
    
    # Set default Cook's Distance threshold
    if alpha_cooks is None:
        alpha_cooks = 4 / n  
    
    # ===== RSTUDENTIZED JACKKNIFE METHOD =====
    # Uses the last column as target, others as predictors
    X = df[numeric_cols[:-1]].values
    y = df[numeric_cols[-1]].values
    
    # Add constant for intercept
    X_with_const = np.column_stack([np.ones(n), X])
    
    # Calculate residuals and studentized residuals
    try:
        # Use normal equations: beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        beta = XtX_inv @ X_with_const.T @ y
        predictions = X_with_const @ beta
        residuals = y - predictions
        
        # Calculate residual standard error
        dof = n - p - 1
        sigma_sq = np.sum(residuals**2) / dof
        
        # Calculate leverage (diagonal of hat matrix)
        hat_matrix = X_with_const @ XtX_inv @ X_with_const.T
        leverage = np.diag(hat_matrix)
        
        # Rstudentized residuals (jackknifed)
        # t_i = e_i / (sigma * sqrt(1 - h_ii))
        rstudent = residuals / (np.sqrt(sigma_sq) * np.sqrt(1 - leverage + 1e-10))
        jackknife_outliers = np.abs(rstudent) > alpha_jackknife
        
    except np.linalg.LinAlgError:
        print("⚠️  Warning: Could not compute Jackknife residuals (singular matrix), using simplified method")
        jackknife_outliers = np.zeros(n, dtype=bool)
        leverage = np.zeros(n)
        sigma_sq = 1
    
    # ===== COOK'S DISTANCE METHOD =====
    try:
        # Cook's Distance: D_i = (e_i^2 / (p+1) * sigma^2) * (h_ii / (1 - h_ii))
        cooks_d = (residuals**2 / ((p + 1) * sigma_sq)) * (leverage / (1 - leverage + 1e-10))
        cooks_outliers = cooks_d > alpha_cooks
    except:
        print("⚠️  Warning: Could not compute Cook's Distance")
        cooks_outliers = np.zeros(n, dtype=bool)
    
    # ===== CLASSIFY OUTLIERS =====
    outlier_type = np.full(n, np.nan, dtype=object)
    
    # Critical: both Jackknife and Cook's Distance
    critical_mask = jackknife_outliers & cooks_outliers
    outlier_type[critical_mask] = 'Critical'
    
    # Outlier: Jackknife only
    jackknife_only = jackknife_outliers & ~cooks_outliers
    outlier_type[jackknife_only] = 'Outlier'
    
    # Influent: Cook's Distance only
    cooks_only = cooks_outliers & ~jackknife_outliers
    outlier_type[cooks_only] = 'Influent'
    
    # Create flagged DataFrame
    df_flagged = df.copy()
    df_flagged['outlier_type'] = outlier_type
    
    # Combine all outliers
    all_outliers_mask = jackknife_outliers | cooks_outliers
    
    # DataFrame without outliers
    df_cleaned = df[~all_outliers_mask].reset_index(drop=True)
    
    # DataFrame with only outliers
    df_outliers = df[all_outliers_mask].reset_index(drop=True)
    
    # Summary
    n_outliers = all_outliers_mask.sum()
    n_critical = critical_mask.sum()
    n_jackknife = jackknife_only.sum()
    n_cooks = cooks_only.sum()
    
    if n_outliers > 0:
        print(f"ℹ️  Outlier detection completed:")
        print(f"   → {n_critical} Critical outliers (both methods)")
        print(f"   → {n_jackknife} Jackknife-only outliers")
        print(f"   → {n_cooks} Cook's Distance-only outliers")
        print(f"   → {n_outliers} total outliers removed ({n_outliers/n*100:.1f}% of data)")
    else:
        print(f"✓  No outliers detected in {len(numeric_cols)} numeric columns")

    return {
        'df_flagged': df_flagged,
        'df_no_outliers': df_cleaned,
        'df_only_outliers': df_outliers
    }