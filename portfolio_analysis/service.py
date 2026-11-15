"""The calculation layer shared by the desktop interface and offline checks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import PortfolioAnalysis, download_prices, prepare_returns
from .frontier import EfficientFrontier


@dataclass(frozen=True)
class Settings:
    weights: dict
    value: float = 100_000.
    observations: int = 3780
    scenarios: int = 100_000
    quantile: float = 0.95
    seed: int = 42

    def __post_init__(self):
        if (not self.weights or any(not ticker.strip() for ticker in self.weights)
                or not np.isfinite(list(self.weights.values())).all()
                or min(self.weights.values()) <= 0
                or not np.isclose(sum(self.weights.values()), 1, rtol=0, atol=1e-8)):
            raise ValueError('Enter unique tickers with positive allocations totaling 100%.')
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError('Portfolio value must be a positive number.')
        if not isinstance(self.observations, int) or self.observations < 252:
            raise ValueError('Use at least 252 observations; EVT usually needs more history.')
        if not isinstance(self.scenarios, int) or not 1000 <= self.scenarios <= 1_000_000:
            raise ValueError('Use between 1,000 and 1,000,000 scenarios.')
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('The random seed must be a nonnegative integer.')
        if not 0.85 <= self.quantile <= 0.99:
            raise ValueError('The EVT threshold must be between 85% and 99%.')


def demo_prices(settings):
    """Synthetic correlated prices for exploring the app without a connection."""
    rng = np.random.default_rng(settings.seed)
    count = settings.observations + 100
    dimension = len(settings.weights)
    common = rng.standard_t(6, (count, 1)) * 0.007
    individual = rng.normal(0.0003, 0.008, (count, dimension))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(common + individual, axis=0)),
        index=pd.bdate_range(end='2026-01-01', periods=count),
        columns=list(settings.weights),
    )


def analyze(settings, demo=False, progress=lambda message: None):
    progress('Creating synthetic prices…' if demo else 'Downloading adjusted prices…')
    prices = demo_prices(settings) if demo else download_prices(list(settings.weights))
    returns = prepare_returns(prices, settings.observations)
    analysis = PortfolioAnalysis(returns, settings.weights)
    progress('Fitting copulas and calculating daily risk…')
    result = {
        'settings': settings, 'demo': demo, 'analysis': analysis,
        'summary': analysis.summary(),
        'daily': analysis.risk_table(settings.scenarios, settings.seed, settings.value),
        'start': returns.index.min(), 'end': returns.index.max(),
        'evt': None, 'evt_error': None,
    }
    progress('Fitting the extreme-value tail and simulating 21 sessions…')
    try:
        tail, probabilities = analysis.monthly_risk(settings.scenarios, settings.seed, settings.quantile)
        result.update({'evt': probabilities, 'tail': tail,
                       'xi_interval': tail.shape_interval(seed=settings.seed),
                       'stability': tail.threshold_stability()})
    except ValueError as error:
        result['evt_error'] = str(error)
    progress('Optimizing the long-only efficient frontier…')
    frontier = EfficientFrontier(returns)
    result.update({'frontier_model': frontier, 'frontier': frontier.curve(),
                   'opportunity_set': frontier.opportunity_set(seed=settings.seed)})
    return result
