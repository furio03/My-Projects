"""Small deterministic checks for the stock prediction workflow."""

import unittest

import numpy as np
import pandas as pd

from stock_market_prediction.back_test import back_test_one_symbol
from stock_market_prediction.data_manipulation import shift_df
from stock_market_prediction.label import label_data


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


if __name__ == '__main__':
    unittest.main()
