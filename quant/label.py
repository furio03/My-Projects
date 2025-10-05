import pandas as pd

def label_data(df: pd.DataFrame, threshold:float) -> pd.DataFrame:
    """
    df must contain a Close information: 'Close'
    threshold is the percentage of change to consider a buy or sell action.
    """
 
    df['action'] = 'hold'
    for i in range(len(df) - 1):
        actual_Close = df.iloc[i]['Close']
        future_Close = df.iloc[i + 1]['Close']
        if future_Close / actual_Close >= 1+ threshold / 100:
            df.at[i, 'action'] = 'buy'
        elif future_Close / actual_Close <= 1 - threshold / 100:
            df.at[i, 'action'] = 'sell'
        # otherwise remains 'hold'
    df = df[['Low','High','Close','Open', 'action', 'Volume', 'Date']].copy()
    return df





