from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def read(filename: str) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_path = data_dir / filename

    return pd.read_csv(data_path)

def handle_Nans(df: pd.DataFrame) -> pd.DataFrame: 
    """Handle NaN values in the DataFrame by detecting NaNs type and performing appropriate imputation if necessary.
    
    New logic:
    1. Remove columns with >35% missing values
    2. For remaining columns:
       - If MAR: use MICE (Iterative Imputer)
       - If MCAR and >5% missing: remove rows with missing values
       - If MCAR and <=5% missing: impute with median (numeric) or mode (categorical)
    """
    # Step 1: Remove columns with more than 35% missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    cols_to_keep = missing_percentage[missing_percentage <= 35].index.tolist()
    df = df[cols_to_keep]
    
    # Recalculate missing percentage after removing columns
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Step 2: Detect missing mechanism (MAR vs MCAR)
    missing_df = pd.DataFrame()
    for col in df.columns:
        missing_df[col] = df[col].isnull()

    p_values = pd.Series(index=df.columns)
    for col in missing_df.columns:
        if df[col].isnull().sum() > 0:
            observed = pd.crosstab(missing_df[col], df[col].notnull())
            chi2, p, dof, expected = chi2_contingency(observed)
            p_values[col] = p
        else:
            p_values[col] = 1.0  # No missing values

    missing_mechanism = {}
    cols_to_impute_mar = []
    cols_to_impute_mcar_high = []  # MCAR with >5% missing
    cols_to_impute_mcar_low = []   # MCAR with <=5% missing
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        p = p_values[col]
        nature = 'MAR' if p <= 0.05 else 'MCAR'
        missing_mechanism[col] = nature
        miss_pct = missing_percentage[col]
        
        if nature == 'MAR':
            cols_to_impute_mar.append(col)
        elif nature == 'MCAR':
            if miss_pct > 5:
                cols_to_impute_mcar_high.append(col)
            else:
                cols_to_impute_mcar_low.append(col)
    
    # Step 3: Handle MAR columns with MICE (only numeric columns)
    if cols_to_impute_mar:
        numeric_mar_cols = [col for col in cols_to_impute_mar if df[col].dtype in ['int64', 'float64']]
        if numeric_mar_cols:
            imputer = IterativeImputer(random_state=42, max_iter=10, verbose=0)
            df[numeric_mar_cols] = imputer.fit_transform(df[numeric_mar_cols])
        
        # For categorical MAR columns, use mode
        categorical_mar_cols = [col for col in cols_to_impute_mar if col not in numeric_mar_cols]
        for col in categorical_mar_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0])
    
    # Step 4: Handle MCAR columns with >5% missing by removing rows
    if cols_to_impute_mcar_high:
        df = df.dropna(subset=cols_to_impute_mcar_high)
    
    # Step 5: Handle MCAR columns with <=5% missing by imputing with median/mode
    if cols_to_impute_mcar_low:
        for col in cols_to_impute_mcar_low:
            # Use median for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            # Use mode for categorical/string columns
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0])

    return df


def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Try to detect and convert column types.

    For each column the function attempts conversions in this order:
      1) numeric  (using pandas.to_numeric with errors='raise')
      2) datetime (using pandas.to_datetime with errors='raise')
      3) fallback to string (astype(str))

    The function stops at the first successful conversion for a column so it will not
    convert a column multiple times (e.g. float then string).
    """

    for column in df.columns:
        series = df[column]

        try:
            df[column] = pd.to_numeric(series)
            continue

        except Exception:
            pass

        try:
            df[column] = pd.to_datetime(series, infer_datetime_format=True)
            continue

        except Exception:
            pass

        df[column] = series.astype(str)

    return df


def identify_variables(df: pd.DataFrame)-> dict:
    """
    Identify quantitative,date time and qualitative variables in the DataFrame.

    Quantitative variables are typically numeric types (int, float),
    while qualitative variables are categorical or object types (str, category).

    Returns a dict with two keys: 'quantitative' and 'qualitative', each containing a list of column names.
    """
    quantitative_vars = []
    qualitative_vars = []
    datetime_vars = []

    for column in df.columns:
        if df[column].dtype in ['object']:
            qualitative_vars.append(column)
        
        if df[column].dtype in ['datetime64[ns]']:
            datetime_vars.append(column)

        if (df[column].dtype in ['int64', 'float64']) :
            quantitative_vars.append(column)
        

    return {
        'quantitative': quantitative_vars,
        'qualitative': qualitative_vars,
        'datetime': datetime_vars
        
    }