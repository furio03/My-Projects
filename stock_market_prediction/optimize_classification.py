import pandas as pd

def optimize(df: pd.DataFrame, threshold: float):
    """
    If the model has a low prediction score for a buy or a sell, then convert it to a hold.
    Also, prevent more than 3 consecutive buys or sells.
    """
    df = df.copy()
    mask = ((df['prediction_label'] == 'buy') | (df['prediction_label'] == 'sell')) & (df['prediction_score'] < threshold)
    df.loc[mask, 'prediction_label'] = 'hold'

    count = 0
    current_side = None
    for idx, row in df.iterrows():
        label = row['prediction_label']
        if label in ['buy', 'sell']:
            if label == current_side:
                count += 1
            else:
                current_side = label
                count = 1
            if count > 3:
                df.at[idx, 'prediction_label'] = 'hold'
        else:
            current_side = None
            count = 0
        
    return df
