import pandas as pd
import numpy as np

def shift_df(i: int, df: pd.DataFrame):
    """
    Shift only the features by i rows UP, leaving 'action' unchanged.
    Insert a first row with action=None and features from the first row.
    """
    cols_to_shift = [col for col in df.columns if (col != 'action' and col != 'Date')]

    # 1. Create the first row with action=None and features from the first row
    first_row = {col: df.iloc[0][col] for col in cols_to_shift}
    first_row['action'] = None

    # 2. Shift the features up by i positions
    df_shifted = df.copy()
    df_shifted[cols_to_shift] = df[cols_to_shift].shift(-i)
    df_shifted = df_shifted.iloc[:-i] if i > 0 else df_shifted

    # 3. Build the new DataFrame
    df_train = pd.concat([
        pd.DataFrame([first_row]),
        df_shifted
    ], ignore_index=True)

    # Remove rows where action is None
    df_train = df_train[df_train['action'].notna()].reset_index(drop=True)

    return df_train

dict_example = {
    'Date': pd.date_range(start='2023-01-01', periods=4, freq='D'),
    'action': ['buy', 'sell', 'buy', 'hold'],
    'feature1': [1, 2, 3, 4],
    'feature2': [5, 6, 7, 8],
    'Close':[10,11,12,13]
}
df_example = pd.DataFrame(dict_example)

# Example usage
if __name__ == "__main__":
    i = 1
    df_shifted = shift_df(i, df_example)
    print(df_shifted)