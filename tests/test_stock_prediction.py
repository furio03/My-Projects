"""Small deterministic checks for the stock prediction workflow."""

import unittest

import numpy as np
import pandas as pd

from stock_market_prediction.back_test import back_test_one_symbol
from stock_market_prediction.data_manipulation import shift_df
from stock_market_prediction.label import label_data
from stock_market_prediction.main import split_periods, train_and_evaluate
from unittest.mock import patch


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


if __name__ == '__main__':
    unittest.main()
