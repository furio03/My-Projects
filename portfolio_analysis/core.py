"""Daily portfolio risk with empirical margins and Gaussian or Student t copulas."""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize, stats
from .evt import ExtremeValueRisk


TRADING_DAYS = 252


def prepare_returns(prices, observations=3780):
    """Keep returns whose current and previous quotes exist for every asset."""
    if observations < 2:
        raise ValueError("At least two return observations are required.")
    if prices.empty or prices.index.has_duplicates or prices.columns.has_duplicates:
        raise ValueError("Prices must be nonempty with unique dates and tickers.")
    prices = prices.sort_index().astype(float)
    quoted = prices.to_numpy()
    if np.isinf(quoted).any() or (prices <= 0).any().any():
        raise ValueError("Quoted prices must be finite and positive.")
    # Calculate before dropping missing rows: a gap must not become a daily return.
    returns = prices.pct_change(fill_method=None).dropna().tail(observations)
    if len(returns) < observations:
        raise ValueError(
            f"Need {observations} common daily returns; found {len(returns)}. "
            "Check the tickers or lower --observations."
        )
    if (returns.std() == 0).any():
        raise ValueError("Each asset must have nonconstant returns to fit a copula.")
    return returns


def regularize_correlation(correlation):
    """Make a correlation matrix positive definite, including duplicate assets."""
    values, vectors = np.linalg.eigh(correlation)
    matrix = (vectors * np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


class EmpiricalCopula:
    """Fit dependence on ranks and retain each asset's historical return range."""

    def __init__(self, returns, family="student"):
        if family not in ("gaussian", "student"):
            raise ValueError("Copula family must be 'gaussian' or 'student'.")
        self.family = family
        self.returns = returns.to_numpy()
        self.dimension = self.returns.shape[1]
        self.df = None
        uniform = returns.rank(method="average").to_numpy() / (len(returns) + 1)
        if self.dimension == 1:
            self.correlation = np.eye(1)
        elif family == "gaussian":
            scores = stats.norm.ppf(uniform)
            self.correlation = regularize_correlation(np.corrcoef(scores, rowvar=False))
        else:
            # For elliptical copulas, rho = sin(pi * Kendall's tau / 2).
            self.correlation = regularize_correlation(
                np.sin(np.pi * returns.corr(method="kendall").to_numpy() / 2)
            )
            # Estimate degrees of freedom with the rank-based correlation fixed.
            def negative_log_likelihood(df):
                scores = stats.t.ppf(uniform, df)
                joint = stats.multivariate_t.logpdf(scores, shape=self.correlation, df=df)
                margins = stats.t.logpdf(scores, df).sum(axis=1)
                return -np.sum(joint - margins)

            fit = optimize.minimize_scalar(
                negative_log_likelihood, bounds=(2.01, 50), method="bounded"
            )
            if not fit.success or not np.isfinite(fit.fun):
                raise ValueError("Student t copula fitting failed.")
            self.df = float(fit.x)

    def simulate(self, scenarios=100_000, seed=42):
        """Return joint daily scenarios in decimal units, in the original column order."""
        if scenarios < 1:
            raise ValueError("The number of scenarios must be positive.")
        rng = np.random.default_rng(seed)
        if self.dimension == 1:
            uniform = rng.uniform(size=(scenarios, 1))
        else:
            scores = rng.multivariate_normal(np.zeros(self.dimension), self.correlation, scenarios)
            if self.family == "student":
                # One shared scale per scenario creates joint tail dependence.
                scores /= np.sqrt(rng.chisquare(self.df, scenarios) / self.df)[:, None]
                uniform = stats.t.cdf(scores, self.df)
            else:
                uniform = stats.norm.cdf(scores)
        return np.column_stack([
            np.quantile(self.returns[:, column], uniform[:, column])
            for column in range(self.dimension)
        ])


def loss_risk(returns, confidence):
    """Return loss VaR and ES; positive numbers mean losses, negative ones gains."""
    losses = -np.asarray(returns, dtype=float)
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError("Risk requires a nonempty vector of finite returns.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    var = np.quantile(losses, confidence, method="inverted_cdf")
    # Average the worst (1-confidence) probability mass, splitting the boundary
    # observation when necessary. This also handles ties at the VaR correctly.
    ordered = np.sort(losses)[::-1]
    tail_size = (1 - confidence) * len(ordered)
    whole = int(np.floor(tail_size))
    remainder = tail_size - whole
    es = (ordered[:whole].sum() + remainder * ordered[whole]) / tail_size
    return float(var), float(es)


class PortfolioAnalysis:
    """Analyze fixed beginning-of-day portfolio weights on aligned daily returns."""

    def __init__(self, returns, weights):
        self.returns = returns.copy()
        self.weights = pd.Series(weights, dtype=float).reindex(returns.columns)
        if set(weights) != set(returns.columns) or self.weights.isna().any():
            raise ValueError("Weights must match the return columns exactly.")
        if (
            not np.isfinite(self.weights).all()
            or (self.weights <= 0).any()
            or not np.isclose(self.weights.sum(), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError("Weights must be positive, finite, and sum to one.")
        if (
            returns.empty or returns.shape[1] == 0 or returns.columns.has_duplicates
            or not np.isfinite(returns.to_numpy()).all()
        ):
            raise ValueError("Returns must be nonempty, finite, and have unique tickers.")
        if len(returns) < 2 or (returns.std() == 0).any():
            raise ValueError("Copula fitting requires at least two nonconstant observations.")
        self.portfolio_returns = returns @ self.weights
        self.copulas = {}

    def summary(self):
        daily = pd.concat([self.returns, self.portfolio_returns.rename("Portfolio")], axis=1)
        return pd.DataFrame({
            "Annual mean (%)": daily.mean() * TRADING_DAYS * 100,
            "Annual volatility (%)": daily.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100,
        })

    def monthly_risk(self, scenarios=100_000, seed=42, quantile=0.95):
        model = ExtremeValueRisk(self.portfolio_returns, quantile)
        return model, model.probability_table(scenarios=scenarios, seed=seed)

    def risk_table(self, scenarios=100_000, seed=42, value=None):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Portfolio value must be finite and positive.")
        samples = {"Historical": self.portfolio_returns.to_numpy()}
        for family in ("gaussian", "student"):
            model = EmpiricalCopula(self.returns, family)
            self.copulas[family] = model
            samples[family.title() + " copula"] = model.simulate(scenarios, seed) @ self.weights.to_numpy()
        rows = []
        for method, returns in samples.items():
            for confidence in (0.95, 0.99):
                var, es = loss_risk(returns, confidence)
                row = {"Method": method, "Confidence": f"{confidence:.0%}",
                       "VaR (%)": var * 100, "ES (%)": es * 100}
                if value is not None:
                    row.update({"VaR (money)": var * value, "ES (money)": es * value})
                rows.append(row)
        return pd.DataFrame(rows).set_index(["Method", "Confidence"])


def read_weights():
    """Read long-only allocations without evaluating user input as Python."""
    weights = {}
    while sum(weights.values()) < 1 - 1e-8:
        ticker = input("Ticker: ").strip().upper()
        if not ticker or ticker in weights:
            print("Enter a nonempty ticker that has not already been added.")
            continue
        try:
            percent = float(input(f"Portfolio allocation for {ticker} (50 means 50%): "))
        except ValueError:
            print("Enter a number for the allocation.")
            continue
        remaining = (1 - sum(weights.values())) * 100
        if not np.isfinite(percent) or percent <= 0 or percent > remaining + 1e-6:
            print(f"Allocation must be positive and no greater than {remaining:.6f}%.")
            continue
        weights[ticker] = percent / 100
    return weights


def download_prices(tickers):
    data = yf.download(tickers, period="max", auto_adjust=False)
    if data is None or data.empty or "Adj Close" not in data.columns:
        raise ValueError("No adjusted prices downloaded. Check the tickers and connection.")
    return data["Adj Close"].reindex(columns=tickers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=3780, help="Common daily returns to use (default: 3780).")
    parser.add_argument("--scenarios", type=int, default=100_000, help="Scenarios per copula (default: 100000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--value", type=float, help="Portfolio value for monetary VaR and ES.")
    parser.add_argument("--evt-threshold", type=float, default=0.95, help="Daily log-loss threshold quantile (default: 0.95).")
    args = parser.parse_args()
    if args.observations < 2 or args.scenarios < 1 or args.seed < 0:
        parser.error("Observations must be >= 2, scenarios >= 1, and seed >= 0.")
    if args.value is not None and (not np.isfinite(args.value) or args.value <= 0):
        parser.error("Portfolio value must be finite and positive.")
    print("Enter portfolio positions until allocations total 100%.")
    print("All prices must be in the same currency; FX conversion is not included.")
    try:
        weights = read_weights()
        returns = prepare_returns(download_prices(list(weights)), args.observations)
        analysis = PortfolioAnalysis(returns, weights)
        print(f"\nSample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} daily returns)")
        print("\nAnnualized statistics (252 sessions; arithmetic mean, not CAGR)")
        print(analysis.summary().round(3).to_string())
        correlation = returns.corr()
        print("\nAsset return correlation")
        print(correlation.round(3).to_string())
        explained = np.linalg.eigvalsh(correlation)[::-1] / len(correlation) * 100
        print("PCA explained variance (%):", np.round(explained, 2))
        print("PCA describes linear concentration; it does not certify diversification.")
        print(f"\nOne-day loss risk ({args.scenarios:,} scenarios per copula; seed {args.seed})")
        print(analysis.risk_table(args.scenarios, args.seed, args.value).round(3).to_string())
        df = analysis.copulas["student"].df
        if df is not None:
            print(f"Student t degrees of freedom: {df:.2f} (fit bounds: 2.01 to 50).")
        else:
            print("Single asset: both copulas reduce to the same empirical margin.")
        print("Positive VaR/ES means a loss. Empirical margins cannot exceed historical asset extremes.")
        print("\nEVT end-of-month losses (21 independent sessions, daily rebalanced weights)")
        try:
            tail, probabilities = analysis.monthly_risk(args.scenarios, args.seed, args.evt_threshold)
            interval = tail.shape_interval(seed=args.seed)
            print(f"Daily log-loss tail index xi: {tail.shape:.3f}; bootstrap 95% interval: {interval}")
            print(f"Threshold quantile: {tail.quantile:.1%}; exceedances: {len(tail.excesses)}")
            print(probabilities.round(5).to_string())
            print("MC intervals describe simulation counting error, not parameter or model uncertainty.")
            print("Zero simulated events do not imply zero risk. Daily independence excludes volatility clustering.")
            print("Threshold sensitivity:")
            print(tail.threshold_stability().round(4).to_string(index=False))
        except ValueError as error:
            print(f"EVT unavailable: {error}")
        print("Static in-sample estimates; volatility clustering and out-of-sample validation are not included.")
    except (ValueError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")
