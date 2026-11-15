"""Predict with a saved training pipeline; never fit on deployment data."""

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_market_prediction.back_test import back_test_one_symbol
from stock_market_prediction.forecasting import evaluate, predict, print_metrics
from stock_market_prediction.main import ROOT, prepare_frame, read_prices
from stock_market_prediction.optimize_classification import optimize


def load_artifact(path):
    artifact = joblib.load(path)
    if artifact.get('format_version') != 1:
        raise ValueError('Legacy model format. Retrain with main.py to save the complete pipeline.')
    return artifact


def predict_saved(artifact, prices):
    frame = prepare_frame(prices, artifact['threshold'], artifact['shift'])
    # Use earlier prices only as indicator history; never report training predictions.
    frame = frame.loc[
        (frame.Date >= pd.Timestamp(artifact['test_start']))
        & (frame.Date > pd.Timestamp(artifact['trained_through']))
    ]
    if frame.empty:
        raise ValueError('No prediction sessions remain after the training cutoff.')
    return predict(artifact['model'], frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'AAPL_historical_data_1d.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'best_model.pkl')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--min-score', type=float, default=0.5)
    args = parser.parse_args()
    predictions = predict_saved(load_artifact(args.model), read_prices(args.csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    if predictions.action.notna().any():
        print_metrics(evaluate(predictions))
    back_test_one_symbol(predictions).to_csv(args.output_dir / 'performance.csv', index=False)
    filtered = optimize(predictions, threshold=args.min_score)
    back_test_one_symbol(filtered).to_csv(args.output_dir / 'performance_optimized.csv', index=False)
    print(f'Saved {len(predictions)} predictions. The final target may still be unknown.')


if __name__ == '__main__':
    main()
