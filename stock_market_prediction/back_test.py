import numpy as np
import pandas as pd

def back_test_one_symbol(df_instructions: pd.DataFrame, position_value: float = 1000.0,
                         initial_cash: float = 3000.0):
    """Execute at the dated close and mark every open position to market."""
    if not np.isfinite([position_value, initial_cash]).all() or min(position_value, initial_cash) <= 0:
        raise ValueError("Position value and initial cash must be finite and positive.")
    df_performance = df_instructions.copy()
    df_performance['Date'] = pd.to_datetime(df_performance['Date'], errors='raise')
    df_performance = df_performance.sort_values('Date').reset_index(drop=True)
    if df_performance.Date.isna().any() or df_performance.Date.duplicated().any():
        raise ValueError('Execution dates must be known and unique.')
    if not np.isfinite(df_performance.Close).all() or (df_performance.Close <= 0).any():
        raise ValueError('Execution prices must be finite and positive.')
    if not df_performance.prediction_label.isin(['buy', 'sell', 'hold']).all():
        raise ValueError('Instructions must be buy, sell, or hold.')
    for column in ['realized_profit', 'unrealized_profit', 'cumulative_profit',
                   'cash', 'exposure', 'equity']:
        df_performance[column] = 0.0

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

        exposure = position * quantity * close
        unrealized = position * (close - avg_price) * quantity
        cash = initial_cash + cumulative_profit - position * avg_price * quantity
        df_performance.loc[i, ['realized_profit', 'unrealized_profit', 'cumulative_profit',
                               'cash', 'exposure', 'equity']] = [
            cumulative_profit, unrealized, cumulative_profit + unrealized,
            cash, exposure, cash + exposure,
        ]

    return df_performance
