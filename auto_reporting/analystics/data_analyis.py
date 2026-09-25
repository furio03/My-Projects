import pandas as pd

def basic_analytics_quant(df: pd.DataFrame , dic: dict) -> pd.DataFrame:
    """
    Perform basic analytics on the DataFrame for quantitative variables.

    """
    analytics = {}

    if len(dic['quantitative'])==0:
        return None
    
    quant_df = df[dic['quantitative']]
    corr_matrix = quant_df.corr().round(2)

    for column in dic['quantitative']:
        series = df[column]
        analytics[column] = {
            'mean': series.mean(),
            'std_dev': series.std(),
            'min': series.min(),
            'max': series.max(),
            "10th_percentile": series.quantile(0.1),
            "25th_percentile": series.quantile(0.25),
            "50th_percentile": series.quantile(0.5),
            "75th_percentile": series.quantile(0.75),
            "90th_percentile": series.quantile(0.9),
        }
    

    return pd.DataFrame(analytics), corr_matrix

def basic_analytics_qual(df: pd.DataFrame , dic: dict) -> pd.DataFrame:
    """
    Perform basic analytics on the DataFrame for qualitative variables.

    """
    analytics = {}

    if len(dic['qualitative'])==0:
        return None

    for column in dic['qualitative']:
        series = df[column]
        analytics[column] = {
            'mode': series.mode()[0] if not series.mode().empty else None,
            'unique_values': series.nunique(),
            'top_5_frequent': series.value_counts().head(5).to_dict(),
        }

    return pd.DataFrame(analytics)


def basic_analytics_date(df: pd.DataFrame , dic: dict) -> pd.DataFrame:
    """
    Perform basic analytics on the DataFrame for date variables.

    """
    analytics = {}

    if len(dic['datetime'])==0:
        return None

    for column in dic['datetime']:
        series = df[column]
        analytics[column] = {
            'earliest_date': series.min(),
            'latest_date': series.max(),
            'date_range_days': (series.max() - series.min()).days,
        }

    return pd.DataFrame(analytics)




 
   