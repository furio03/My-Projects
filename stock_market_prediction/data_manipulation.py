import pandas as pd


def shift_df(periods: int, df: pd.DataFrame):
    """Lag model inputs while keeping target dates and execution closes intact."""
    if periods < 1:
        raise ValueError("Use at least one session of lag for close execution.")
    result = df.sort_values('Date').copy()
    features = result.columns.difference(['Date', 'Close', 'action'])
    result[features] = result[features].shift(periods)
    return result.reset_index(drop=True)
