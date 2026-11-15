"""Offline checks for return alignment, copula sampling, and loss calculations."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_analysis import (
    EmpiricalCopula, PortfolioAnalysis, loss_risk, main, prepare_returns, read_weights,
)


def sample_returns(rows=1000):
    rng = np.random.default_rng(12)
    values = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], rows) * 0.01
    return pd.DataFrame(values, columns=["AAA", "BBB"])


class ReturnTests(unittest.TestCase):
    def test_price_changes_use_previous_price_and_preserve_dates(self):
        prices = pd.DataFrame({
            "AAA": [100, 110, 99, np.nan, 100, 120, 132],
            "BBB": [100, 101, 100, 102, 103, 104, 108],
        }, index=pd.date_range("2020-01-01", periods=7))
        returns = prepare_returns(prices.iloc[:3], observations=2)
        np.testing.assert_allclose(returns.AAA, [0.1, -0.1])
        # Reverse input dates and leave a gap. Neither side of the gap is filled.
        aligned = prepare_returns(prices.iloc[::-1], observations=2)
        self.assertEqual(list(aligned.index), list(prices.index[-2:]))
        np.testing.assert_allclose(aligned.AAA, [0.2, 0.1])
        with self.assertRaisesRegex(ValueError, "found 4"):
            prepare_returns(prices, observations=5)

    def test_bad_prices_and_short_history_fail_clearly(self):
        for values in ([100, 0, 110], [100, np.inf, 110], [100, -10, 110]):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                prepare_returns(pd.DataFrame({"AAA": values}), 2)
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            prepare_returns(pd.DataFrame({"AAA": [100, 100, 100]}), 2)

    def test_allocation_input_rejects_invalid_numbers_and_duplicates(self):
        answers = ["AAA", "nan", "AAA", "101", "AAA", "bad", "AAA", "60", "AAA", "BBB", "40"]
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(read_weights(), {"AAA": 0.6, "BBB": 0.4})


class RiskTests(unittest.TestCase):
    def test_loss_sign_and_fractional_tail(self):
        var, es = loss_risk([-0.01, -0.02, -0.03, -0.04], 0.625)
        self.assertAlmostEqual(var, 0.03)
        self.assertAlmostEqual(es, (0.04 + 0.5 * 0.03) / 1.5)
        var, es = loss_risk([0.01, 0.02], 0.95)
        self.assertAlmostEqual(var, -0.01)
        self.assertAlmostEqual(es, -0.01)

    def test_ties_do_not_include_the_entire_var_boundary(self):
        var, es = loss_risk([0, 0, 0, -0.10], 0.5)
        self.assertEqual(var, 0)
        self.assertAlmostEqual(es, 0.05)

    def test_summary_uses_weighted_daily_returns_and_covariance(self):
        returns = sample_returns()
        analysis = PortfolioAnalysis(returns, {"BBB": 0.4, "AAA": 0.6})
        portfolio = returns.AAA * 0.6 + returns.BBB * 0.4
        summary = analysis.summary().loc["Portfolio"]
        self.assertAlmostEqual(summary["Annual mean (%)"], portfolio.mean() * 252 * 100)
        variance = np.array([0.6, 0.4]) @ returns.cov().to_numpy() @ np.array([0.6, 0.4])
        self.assertAlmostEqual(summary["Annual volatility (%)"], np.sqrt(variance * 252) * 100)
        for weights in ({"AAA": 1}, {"AAA": 0.6, "BBB": 0.5}, {"AAA": np.nan, "BBB": 1}):
            with self.assertRaises(ValueError):
                PortfolioAnalysis(returns, weights)


class CopulaTests(unittest.TestCase):
    def test_gaussian_samples_preserve_margins_and_dependence(self):
        returns = sample_returns()
        model = EmpiricalCopula(returns, "gaussian")
        simulated = model.simulate(30_000)
        np.testing.assert_array_equal(simulated, model.simulate(30_000))
        self.assertTrue((simulated >= returns.min().to_numpy()).all())
        self.assertTrue((simulated <= returns.max().to_numpy()).all())
        np.testing.assert_allclose(np.quantile(simulated, [0.05, 0.5, 0.95], axis=0),
                                   returns.quantile([0.05, 0.5, 0.95]), atol=0.0006)
        self.assertAlmostEqual(stats.kendalltau(*simulated.T).statistic,
                               returns.corr(method="kendall").iloc[0, 1], delta=0.03)

    def test_student_fit_recovers_heavy_tailed_dependence(self):
        rng = np.random.default_rng(4)
        scores = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 3000)
        scores /= np.sqrt(rng.chisquare(4, 3000) / 4)[:, None]
        # Normal margins isolate copula tail dependence from marginal heavy tails.
        returns = pd.DataFrame(stats.norm.ppf(stats.t.cdf(scores, 4)) * 0.01)
        model = EmpiricalCopula(returns, "student")
        self.assertGreater(model.df, 2.01)
        self.assertLess(model.df, 8)
        simulated = model.simulate(30_000)
        np.testing.assert_array_equal(simulated, model.simulate(30_000))
        gaussian = EmpiricalCopula(returns, "gaussian").simulate(30_000)
        threshold = returns.quantile(0.01).to_numpy()
        self.assertGreater(np.mean((simulated < threshold).all(axis=1)),
                           np.mean((gaussian < threshold).all(axis=1)))

    def test_single_asset_and_duplicate_assets_remain_finite(self):
        returns = sample_returns()
        single = returns[["AAA"]]
        np.testing.assert_array_equal(EmpiricalCopula(single, "gaussian").simulate(1000),
                                      EmpiricalCopula(single, "student").simulate(1000))
        duplicate = single.assign(BBB=single.AAA)
        for family in ("gaussian", "student"):
            simulated = EmpiricalCopula(duplicate, family).simulate(1000)
            self.assertTrue(np.isfinite(simulated).all())
            np.testing.assert_allclose(simulated[:, 0], simulated[:, 1], atol=0.0001)

    def test_report_scales_monetary_losses(self):
        analysis = PortfolioAnalysis(sample_returns(400), {"AAA": 0.6, "BBB": 0.4})
        table = analysis.risk_table(scenarios=2000, value=10_000)
        self.assertEqual(len(table), 6)
        np.testing.assert_allclose(table["VaR (money)"], table["VaR (%)"] * 100)
        self.assertTrue((table["ES (%)"] >= table["VaR (%)"] - 1e-12).all())
        for confidence in ("95%", "99%"):
            historical = table.loc[("Historical", confidence)]
            var, es = loss_risk(analysis.portfolio_returns, float(confidence[:-1]) / 100)
            self.assertAlmostEqual(historical["VaR (%)"], var * 100)
            self.assertAlmostEqual(historical["ES (%)"], es * 100)

    def test_cli_download_to_report(self):
        returns = sample_returns(400)
        prices = 100 * (1 + returns).cumprod()
        prices.index = pd.date_range("2020-01-01", periods=len(prices))
        downloaded = pd.concat({"Adj Close": prices}, axis=1)
        arguments = ["portfolio_analysis.py", "--observations", "300", "--scenarios", "2000", "--value", "10000"]
        with patch("sys.argv", arguments), patch("builtins.input", side_effect=["AAA", "60", "BBB", "40"]), patch("yfinance.download", return_value=downloaded) as download, contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        download.assert_called_once_with(["AAA", "BBB"], period="max", auto_adjust=False)
        for text in ("Annual volatility", "One-day loss risk", "Student copula", "VaR (money)", "300 daily returns"):
            self.assertIn(text, output.getvalue())


if __name__ == "__main__":
    unittest.main()
