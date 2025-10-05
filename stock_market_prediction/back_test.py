import pandas as pd

def back_test_one_symbol(df_instructions: pd.DataFrame, position_value: float = 1000.0):
    df_inverted = df_instructions.iloc[::-1].reset_index(drop=True)
    df_performance = df_inverted.copy()
    df_performance = df_performance[['Date', 'prediction_score', 'prediction_label', 'Close']].copy()
    df_performance['cumulative_profit'] = 0.0

    position = 0  # +1 long, -1 short, 0 flat
    quantity = 0.0
    avg_price = 0.0
    cumulative_profit = 0.0

    for i, row in df_performance.iterrows():
        action = row['prediction_label']
        close = row['Close']

        if action == 'buy':
            if position >= 0:
                qty = position_value / close
                avg_price = (avg_price * quantity + close * qty) / (quantity + qty) if quantity > 0 else close
                quantity += qty
                position = 1
            else: # close short
                profit = (avg_price - close) * quantity
                cumulative_profit += profit
                position = 0
                quantity = 0.0
                avg_price = 0.0
        elif action == 'sell':
            if position <= 0:
                qty = position_value / close
                avg_price = (avg_price * quantity + close * qty) / (quantity + qty) if quantity > 0 else close
                quantity += qty
                position = -1
            else:
                # close long
                profit = (close - avg_price) * quantity
                cumulative_profit += profit
                position = 0
                quantity = 0.0
                avg_price = 0.0

        df_performance.at[i, 'cumulative_profit'] = cumulative_profit

    return df_performance



