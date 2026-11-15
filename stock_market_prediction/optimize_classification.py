import numpy as np

try:
    from .back_test import back_test_one_symbol
except ImportError:
    from back_test import back_test_one_symbol


def optimize(df, threshold=0.5, max_positions=3, position_value=1000., initial_cash=3000.):
    """Filter confidence, then apply the same execution limits as the backtest."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('The minimum prediction score must be between zero and one.')
    if not np.isfinite(df.prediction_score).all() or not df.prediction_score.between(0, 1).all():
        raise ValueError('Prediction scores must be finite probabilities.')
    result = df.sort_values('Date').reset_index(drop=True).copy()
    result.loc[result.prediction_score < threshold, 'prediction_label'] = 'hold'
    executions = back_test_one_symbol(result, position_value, initial_cash, max_positions)
    result['prediction_label'] = executions.executed_label.to_numpy()
    return result
