import unittest

import numpy as np
import pandas as pd

from portfolio_analysis.frontier import EfficientFrontier


def sample():
    rng = np.random.default_rng(30)
    return pd.DataFrame(rng.multivariate_normal(
        [0.0003, 0.0007, 0.0005],
        [[0.0001, 0.00002, 0.00003], [0.00002, 0.0004, 0.00005], [0.00003, 0.00005, 0.0002]],
        5000,
    ), columns=['AAA', 'BBB', 'CCC'])


class FrontierTests(unittest.TestCase):
    def test_curve_is_feasible_and_on_the_efficient_branch(self):
        returns = sample()
        model = EfficientFrontier(returns)
        curve = model.curve(30)
        weights = curve.filter(like='Weight ').to_numpy()
        self.assertTrue((weights >= -1e-10).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_allclose(curve['Annual return (%)'], weights @ model.mean * 100)
        self.assertTrue((curve['Annual return (%)'].diff().dropna() >= -1e-6).all())
        self.assertTrue((curve['Annual volatility (%)'].diff().dropna() >= -1e-5).all())
        self.assertAlmostEqual(curve['Annual return (%)'].iloc[-1], returns.mean().max() * 252 * 100, places=5)
        cloud = model.opportunity_set(10000)
        self.assertLessEqual(curve['Annual volatility (%)'].iloc[0], cloud['Annual volatility (%)'].min() + 1e-5)

    def test_two_asset_minimum_matches_the_closed_form(self):
        model = EfficientFrontier(sample()[['AAA', 'BBB']])
        covariance = model.covariance
        expected = np.clip((covariance[1, 1] - covariance[0, 1]) /
                           (covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]), 0, 1)
        self.assertAlmostEqual(model.minimum_weights[0], expected, places=5)

    def test_single_duplicate_and_equal_mean_assets(self):
        returns = sample()[['AAA']]
        for frame in (returns, returns.assign(BBB=returns.AAA)):
            model = EfficientFrontier(frame)
            curve = model.curve()
            self.assertEqual(len(curve), 1)
            self.assertTrue(np.isfinite(curve.to_numpy()).all())
        returns = sample()
        equal = returns - returns.mean() + 0.0005
        self.assertEqual(len(EfficientFrontier(equal).curve()), 1)

    def test_current_allocation_matches_direct_mean_and_covariance(self):
        returns = sample()
        weights = np.array([0.5, 0.3, 0.2])
        point = EfficientFrontier(returns).point(dict(zip(returns.columns, weights)))
        portfolio = returns @ weights
        self.assertAlmostEqual(point['Annual return (%)'], portfolio.mean() * 252 * 100)
        self.assertAlmostEqual(point['Annual volatility (%)'], portfolio.std() * np.sqrt(252) * 100)
        with self.assertRaises(ValueError):
            EfficientFrontier(returns).point([1, -0.1, 0.1])


if __name__ == '__main__':
    unittest.main()
