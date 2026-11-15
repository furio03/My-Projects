"""Select settings on validation data, then evaluate once on the final test period."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_market_prediction.back_test import back_test_one_symbol
from stock_market_prediction.data_manipulation import shift_df
from stock_market_prediction.indicators import calculate_indicators
from stock_market_prediction.label import label_data

ROOT = Path(__file__).resolve().parent


def read_prices(path):
    """Accept the existing Yahoo CSV, including its optional ticker header row."""
    frame = pd.read_csv(path)
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame = frame.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if frame.Date.duplicated().any():
        raise ValueError('Price dates must be unique.')
    return frame


def prepare_frame(prices, threshold, lag=1):
    frame = prices.sort_values('Date').reset_index(drop=True).copy()
    frame = calculate_indicators(label_data(frame, threshold))
    frame = shift_df(lag, frame)
    # Retain the unknown final target so the last session can still be predicted.
    inputs = frame.columns.difference(['action'])
    return frame.dropna(subset=inputs).reset_index(drop=True)


def split_periods(frame, validation_size=365, test_size=365):
    """Purge one target row at each boundary: its label uses the following close."""
    if validation_size < 1 or test_size < 1:
        raise ValueError('Validation and test sizes must be positive.')
    test_start = len(frame) - test_size
    validation_start = test_start - validation_size
    if validation_start < 3:
        raise ValueError('Not enough observations for training, validation, and test.')
    train = frame.iloc[:validation_start - 1].dropna(subset=['action'])
    validation = frame.iloc[validation_start:test_start - 1].dropna(subset=['action'])
    test = frame.iloc[test_start:].copy()
    return train, validation, test


def fit_and_forecast(train, evaluation, forecast, **options):
    predictions, metrics = forecast(train, evaluation, **options)
    k = metrics['model']['pipeline'].named_steps['groups'].n_clusters
    return predictions, metrics, k


def train_and_evaluate(prices, forecast, validation_size=365, test_size=365, **options):
    best_score = -1
    best_threshold = None
    for threshold in (0.2, 0.3, 0.4):
        frame = prepare_frame(prices, threshold)
        train, validation, _ = split_periods(frame, validation_size, test_size)
        _, metrics, _ = fit_and_forecast(train, validation, forecast, **options)
        score = metrics['test_report']['macro avg']['precision']
        if score > best_score:
            best_score, best_threshold = score, threshold
    frame = prepare_frame(prices, best_threshold)
    _, _, test = split_periods(frame, validation_size, test_size)
    # Refit on all pre-test observations, purging the target next to the test.
    development = frame.iloc[:len(frame) - test_size - 1].dropna(subset=['action'])
    predictions, metrics, k = fit_and_forecast(development, test, forecast, **options)
    return predictions, metrics, {
        'threshold': best_threshold, 'shift': 1, 'best_k': k,
        'trained_through': development.Date.max(), 'test_start': test.Date.min(),
    }


def save_artifact(model, settings, path):
    """Persist the fitted pipeline, label mapping, timing, and selected settings."""
    joblib.dump({'format_version': 1, 'model': model, **settings}, path)


def main():
    from stock_market_prediction.forecasting import forecast, print_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'AAPL_historical_data_1d.csv')
    parser.add_argument('--validation-size', type=int, default=365)
    parser.add_argument('--test-size', type=int, default=365)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', choices=['xgboost', 'rf'], default='xgboost')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, metrics, settings = train_and_evaluate(
        read_prices(args.csv), forecast, args.validation_size, args.test_size,
        target='action', fold=args.folds, n_iter=args.iterations, model_used=args.model, n_jobs=args.jobs,
    )
    print('Selected settings:', settings)
    print_metrics(metrics)
    back_test_one_symbol(predictions).to_csv(args.output_dir / 'performance.csv', index=False)
    save_artifact(metrics['model'], settings, args.output_dir / 'best_model.pkl')


if __name__ == '__main__':
    main()
