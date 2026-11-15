import unittest

import numpy as np
from scipy import stats

from portfolio_analysis.evt import ExtremeValueRisk, binomial_interval, fit_tail


def synthetic_returns():
    rng = np.random.default_rng(19)
    body = rng.uniform(-0.025, 0.02, 9000)
    tail = 0.02 + stats.genpareto.rvs(0.25, scale=0.01, size=1000, random_state=rng)
    return np.expm1(-np.concatenate([body, tail]))


class ExtremeValueTests(unittest.TestCase):
    def test_recovers_a_known_tail_index(self):
        rng = np.random.default_rng(11)
        sample = stats.genpareto.rvs(0.3, scale=0.02, size=20000, random_state=rng)
        shape, scale = fit_tail(sample)
        self.assertAlmostEqual(shape, 0.3, delta=0.04)
        self.assertAlmostEqual(scale, 0.02, delta=0.002)

    def test_month_uses_compounding_and_matches_one_day_tail_formula(self):
        model = ExtremeValueRisk(synthetic_returns(), 0.9)
        level = 0.06
        cutoff = -np.log1p(-level)
        expected = model.tail_probability * stats.genpareto.sf(
            cutoff - model.threshold, model.shape, scale=model.scale,
        )
        losses = model.simulate_month(100000, sessions=1)
        self.assertAlmostEqual(np.mean(losses >= level), expected, delta=0.001)
        self.assertTrue((losses <= 1).all())
        np.testing.assert_array_equal(losses, model.simulate_month(100000, sessions=1))

    def test_month_compounds_all_21_sessions(self):
        model = ExtremeValueRisk(synthetic_returns(), 0.9)
        model.body = np.array([0.01])
        model.tail_probability = 0
        np.testing.assert_allclose(model.simulate_month(100), 1 - np.exp(-0.21))

    def test_probability_table_is_monotone_and_counts_rare_events_honestly(self):
        model = ExtremeValueRisk(synthetic_returns(), 0.9)
        table = model.probability_table(10000)
        self.assertEqual(table.index.tolist(), [1., 3., 5., 10., 15., 20., 25.])
        self.assertTrue((table['Probability (%)'].diff().dropna() <= 0).all())
        self.assertTrue((table['MC lower 95% (%)'] <= table['Probability (%)']).all())
        self.assertTrue((table['MC upper 95% (%)'] >= table['Probability (%)']).all())
        lower, upper = binomial_interval(0, 10000)
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 0)
        self.assertEqual(binomial_interval(10000, 10000)[1], 1)

    def test_threshold_diagnostics_and_bootstrap_are_reproducible(self):
        model = ExtremeValueRisk(synthetic_returns(), 0.95)
        self.assertGreaterEqual(len(model.threshold_stability()), 3)
        interval = model.shape_interval(resamples=30)
        self.assertLess(interval[0], interval[1])
        np.testing.assert_array_equal(interval, model.shape_interval(resamples=30))

    def test_insufficient_or_invalid_data_fail_explicitly(self):
        for values in ([0.01] * 100, [-1., 0.1], [np.nan, 0.1]):
            with self.assertRaises(ValueError):
                ExtremeValueRisk(values)
        with self.assertRaisesRegex(ValueError, '50 threshold'):
            ExtremeValueRisk(np.random.default_rng(5).normal(0, 0.01, 100))


if __name__ == '__main__':
    unittest.main()
