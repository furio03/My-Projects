"""Small deterministic checks for the stock prediction workflow."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stock_market_prediction.back_test import back_test_one_symbol
from stock_market_prediction.data_manipulation import shift_df
from stock_market_prediction.label import label_data
from stock_market_prediction.main import split_periods, train_and_evaluate, save_artifact
from stock_market_prediction.deploy_model import load_artifact
from stock_market_prediction.forecasting import predict
from unittest.mock import patch
from stock_market_prediction.forecasting import forecast, temporal_folds
from stock_market_prediction.indicators import add_kmeans


class TimingTests(unittest.TestCase):
    def test_lag_preserves_execution_price_and_target_date(self):
        frame = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=3),
            'Close': [100., 110., 99.], 'feature': [1., 2., 3.],
            'action': ['buy', 'sell', None],
        })
        shifted = shift_df(1, frame.iloc[::-1])
        self.assertEqual(shifted.loc[1, 'Close'], 110)
        self.assertEqual(shifted.loc[1, 'Date'], frame.loc[1, 'Date'])
        self.assertEqual(shifted.loc[1, 'feature'], 1)
        self.assertEqual(shifted.loc[1, 'action'], 'sell')
        self.assertTrue(pd.isna(shifted.loc[0, 'feature']))
        with self.assertRaises(ValueError):
            shift_df(0, frame)

    def test_last_target_is_unknown(self):
        frame = pd.DataFrame({
            'Close': [100., 110., 99.], 'Date': pd.date_range('2025-01-01', periods=3),
            'Low': [99., 109., 98.], 'High': [101., 111., 100.],
            'Open': [100., 110., 99.], 'Volume': [1000.] * 3,
        })
        labeled = label_data(frame, 1)
        self.assertEqual(labeled.action.iloc[:2].tolist(), ['buy', 'sell'])
        self.assertTrue(pd.isna(labeled.action.iloc[-1]))

    def test_backtest_sorts_dates_instead_of_reversing_input(self):
        frame = pd.DataFrame({
            'Date': ['2025-01-01', '2025-01-02'], 'Close': [100., 110.],
            'prediction_label': ['buy', 'sell'], 'prediction_score': [1., 1.],
        })
        result = back_test_one_symbol(frame)
        self.assertAlmostEqual(result.cumulative_profit.iloc[-1], 100.)
        pd.testing.assert_frame_equal(result, back_test_one_symbol(frame.iloc[::-1]))


class SelectionTests(unittest.TestCase):
    def test_threshold_selection_never_sees_final_test(self):
        frame = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=30),
            'Close': np.arange(30) + 100., 'action': ['hold'] * 30,
        })
        calls = []

        def fake_forecast(train, evaluation, forecast, **options):
            calls.append((train.copy(), evaluation.copy()))
            score = [0.2, 0.8, 0.4, 0.9][len(calls) - 1]
            return evaluation, {'test_report': {'macro avg': {'precision': score}}}, 2

        with patch('stock_market_prediction.main.prepare_frame', return_value=frame), patch(
            'stock_market_prediction.main.fit_and_forecast', side_effect=fake_forecast
        ):
            _, _, settings = train_and_evaluate(frame, None, validation_size=5, test_size=5)
        self.assertEqual(settings['threshold'], 0.3)
        self.assertEqual(len(calls), 4)
        for train, validation in calls[:3]:
            self.assertLess(train.Date.max(), validation.Date.min())
            self.assertEqual(validation.Date.max(), frame.Date.iloc[23])
        self.assertEqual(calls[-1][0].Date.max(), frame.Date.iloc[23])
        self.assertEqual(calls[-1][1].Date.min(), frame.Date.iloc[25])
        train, validation, test = split_periods(frame, 5, 5)
        self.assertEqual(train.index[-1], 18)
        self.assertEqual(validation.index[0], 20)
        self.assertEqual(validation.index[-1], 23)
        self.assertEqual(test.index[0], 25)


class TemporalTests(unittest.TestCase):
    def test_each_fold_trains_on_past_only_with_target_gap(self):
        data = np.arange(120).reshape(-1, 1)
        labels = np.tile([0, 1, 2], 40)
        for train, validation in temporal_folds(data, labels, 3):
            self.assertLess(train[-1] + 1, validation[0])

    def test_actual_search_fits_groups_inside_each_training_fold(self):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=90),
            'Close': 100., 'feature': rng.normal(size=90),
            'action': np.tile(['buy', 'hold', 'sell'], 30),
        })
        seen = []
        from stock_market_prediction.forecasting import KMeansFeatures
        original = KMeansFeatures.fit

        def record(model, X, y=None):
            seen.append(X.index.to_numpy())
            return original(model, X, y)

        with patch.object(KMeansFeatures, 'fit', record):
            predictions, metrics = forecast(frame.iloc[:75], frame.iloc[75:],
                                            fold=2, n_jobs=1, n_iter=1, model_used='rf')
        self.assertEqual(len(seen), 3)  # Two folds and the final development refit.
        expected = temporal_folds(frame.iloc[:75], np.tile([0, 1, 2], 25), 2)
        for actual, (training, _) in zip(seen, expected):
            np.testing.assert_array_equal(actual, training)
        self.assertEqual(len(predictions), 15)
        self.assertIn('pipeline', metrics['model'])


class ScalingTests(unittest.TestCase):
    def test_prediction_reuses_training_scale_regardless_of_batch(self):
        training = pd.DataFrame({'feature': [0., 1., 2., 100., 101., 102.]})
        _, model, _ = add_kmeans(training, 2)
        mean = model.named_steps['standardscaler'].mean_.copy()
        small = pd.DataFrame({'feature': [0., 1., 2.]})
        extreme = pd.DataFrame({'feature': [0., 1., 10000.]})
        with patch('sklearn.preprocessing.StandardScaler.fit', side_effect=AssertionError('Refit on prediction data')):
            first = add_kmeans(small, 2, model)
            second = add_kmeans(extreme, 2, model)
            alone = add_kmeans(small.iloc[:1], 2, model)
        self.assertEqual(first.iloc[0], second.iloc[0])
        self.assertEqual(first.iloc[0], alone.iloc[0])
        np.testing.assert_array_equal(model.named_steps['standardscaler'].mean_, mean)


class PersistenceTests(unittest.TestCase):
    def test_saved_model_predicts_without_refitting_and_keeps_unknown_targets(self):
        rng = np.random.default_rng(24)
        frame = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=90),
            'Close': 100., 'feature': rng.normal(size=90),
            'action': np.tile(['buy', 'hold', 'sell'], 30),
        })
        expected, metrics = forecast(frame.iloc[:75], frame.iloc[75:], fold=2,
                                      n_jobs=1, n_iter=1, model_used='rf')
        evaluation = frame.iloc[75:].copy()
        evaluation.loc[evaluation.index[-1], 'action'] = None
        settings = {'threshold': 0.3, 'shift': 1, 'trained_through': frame.Date.iloc[74],
                    'test_start': frame.Date.iloc[75]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            save_artifact(metrics['model'], settings, path)
            loaded = load_artifact(path)
            with patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('Deployment refit')):
                actual = predict(loaded['model'], evaluation)
        np.testing.assert_array_equal(actual.prediction_label, expected.prediction_label)
        np.testing.assert_allclose(actual.prediction_score, expected.prediction_score)
        self.assertTrue(pd.isna(actual.action.iloc[-1]))
        with patch('joblib.load', return_value={'model': object()}):
            with self.assertRaisesRegex(ValueError, 'Retrain'):
                load_artifact('legacy.pkl')


class AccountingTests(unittest.TestCase):
    def instructions(self, actions, closes):
        return pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=len(actions)),
            'Close': closes, 'prediction_label': actions,
        })

    def test_open_long_and_short_are_marked_to_market(self):
        for action, profit, cash in [('buy', 100., 2000.), ('sell', -100., 4000.)]:
            result = back_test_one_symbol(self.instructions([action, 'hold'], [100., 110.]))
            last = result.iloc[-1]
            self.assertEqual(last.realized_profit, 0)
            self.assertAlmostEqual(last.unrealized_profit, profit)
            self.assertAlmostEqual(last.cumulative_profit, profit)
            self.assertAlmostEqual(last.cash, cash)
            self.assertAlmostEqual(last.equity, 3000 + profit)
            np.testing.assert_allclose(result.equity, result.cash + result.exposure)
            np.testing.assert_allclose(result.equity - 3000, result.cumulative_profit)

    def test_closing_position_does_not_double_count_unrealized_profit(self):
        result = back_test_one_symbol(self.instructions(['buy', 'hold', 'sell'], [100., 110., 120.]))
        self.assertAlmostEqual(result.realized_profit.iloc[-1], 200.)
        self.assertEqual(result.unrealized_profit.iloc[-1], 0)
        self.assertAlmostEqual(result.equity.iloc[-1], 3200.)
        self.assertAlmostEqual(result.cash.iloc[-1], 3200.)

    def test_multiple_entries_use_quantity_weighted_cost(self):
        result = back_test_one_symbol(self.instructions(['buy', 'buy', 'sell'], [100., 200., 150.]))
        self.assertAlmostEqual(result.realized_profit.iloc[-1], 250.)
        self.assertAlmostEqual(result.equity.iloc[-1], 3250.)

    def test_invalid_execution_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            back_test_one_symbol(self.instructions(['buy'], [0.]))
        frame = self.instructions(['buy', 'sell'], [100., 110.])
        frame['Date'] = frame.Date.iloc[0]
        with self.assertRaisesRegex(ValueError, 'unique'):
            back_test_one_symbol(frame)


if __name__ == '__main__':
    unittest.main()
