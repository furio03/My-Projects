import numpy as np
import pandas as pd


def back_test_one_symbol(df_instructions, position_value=1000., initial_cash=3000.,
                         max_positions=3):
    """Execute dated signals with fixed entry sizes and mark open positions daily."""
    if not np.isfinite([position_value, initial_cash]).all() or min(position_value, initial_cash) <= 0:
        raise ValueError('Position value and initial cash must be finite and positive.')
    if not isinstance(max_positions, int) or max_positions < 1:
        raise ValueError('The maximum number of positions must be a positive integer.')
    result = df_instructions.copy()
    result['Date'] = pd.to_datetime(result['Date'], errors='raise')
    result = result.sort_values('Date').reset_index(drop=True)
    if result.Date.isna().any() or result.Date.duplicated().any():
        raise ValueError('Execution dates must be known and unique.')
    if not np.isfinite(result.Close).all() or (result.Close <= 0).any():
        raise ValueError('Execution prices must be finite and positive.')
    if not result.prediction_label.isin(['buy', 'sell', 'hold']).all():
        raise ValueError('Instructions must be buy, sell, or hold.')

    for column in ['realized_profit', 'unrealized_profit', 'cumulative_profit',
                   'cash', 'exposure', 'equity', 'reserved_capital', 'free_capital']:
        result[column] = 0.0
    result['open_entries'] = 0
    result['executed_label'] = 'hold'
    position = 0
    entries = 0
    quantity = 0.0
    avg_price = 0.0
    realized = 0.0

    for i, row in result.iterrows():
        side = {'buy': 1, 'sell': -1, 'hold': 0}[row.prediction_label]
        close = row.Close
        available = initial_cash + realized - entries * position_value
        if side and position == -side:
            realized += position * (close - avg_price) * quantity
            position, entries, quantity, avg_price = 0, 0, 0.0, 0.0
            result.at[i, 'executed_label'] = row.prediction_label
        elif side and entries < max_positions and available >= position_value - 1e-8:
            added = position_value / close
            avg_price = (avg_price * quantity + close * added) / (quantity + added)
            quantity += added
            entries += 1
            position = side
            result.at[i, 'executed_label'] = row.prediction_label

        # Entry notional is reserved on either side; short proceeds are not extra capital.
        reserved = entries * position_value
        exposure = position * quantity * close
        unrealized = position * (close - avg_price) * quantity
        cash = initial_cash + realized - position * avg_price * quantity
        result.loc[i, ['realized_profit', 'unrealized_profit', 'cumulative_profit',
                       'cash', 'exposure', 'equity', 'reserved_capital', 'free_capital']] = [
            realized, unrealized, realized + unrealized, cash, exposure,
            cash + exposure, reserved, initial_cash + realized - reserved,
        ]
        result.at[i, 'open_entries'] = entries
    return result
