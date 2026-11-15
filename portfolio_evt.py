"""Peaks-over-threshold estimates for end-of-month portfolio losses."""

import numpy as np
import pandas as pd
from scipy import stats


LOSS_LEVELS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25)


def fit_tail(excesses):
    """Fit a GPD with zero location, scaling the data for numerical stability."""
    excesses = np.asarray(excesses, dtype=float)
    scale = excesses.mean()
    if (not np.isfinite(excesses).all() or np.any(excesses <= 0)
            or not np.isfinite(scale) or scale <= 0 or np.ptp(excesses) == 0):
        raise ValueError('Tail excesses must be positive and nonconstant.')
    shape, _, fitted_scale = stats.genpareto.fit(excesses / scale, floc=0)
    fitted_scale *= scale
    if (not np.isfinite([shape, fitted_scale]).all() or fitted_scale <= 0
            or np.any(1 + shape * excesses / fitted_scale <= 0)):
        raise ValueError('The fitted Pareto tail has invalid parameters or support.')
    return float(shape), float(fitted_scale)


def binomial_interval(events, samples):
    """Exact 95% interval for Monte Carlo counting error, conditional on the model."""
    lower = 0. if events == 0 else stats.beta.ppf(0.025, events, samples - events + 1)
    upper = 1. if events == samples else stats.beta.ppf(0.975, events + 1, samples - events)
    return float(lower), float(upper)


class ExtremeValueRisk:
    """Splice an empirical daily log-loss body with a fitted GPD upper tail.

    The shape xi is the extreme-value index of daily portfolio log losses.
    It is not a direct fit to monthly returns or a monthly maximum distribution.
    """

    def __init__(self, returns, quantile=0.95):
        values = np.asarray(returns, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('EVT requires a nonempty series of finite daily returns.')
        if np.any(values <= -1):
            raise ValueError('EVT requires daily returns greater than -100%.')
        if not 0.85 <= quantile <= 0.99:
            raise ValueError('Choose a tail threshold quantile between 0.85 and 0.99.')
        self.losses = -np.log1p(values)
        self.quantile = quantile
        self.threshold = float(np.quantile(self.losses, quantile))
        if self.threshold <= 0:
            raise ValueError('The selected EVT threshold must lie in the loss tail.')
        self.excesses = self.losses[self.losses > self.threshold] - self.threshold
        if len(self.excesses) < 50:
            raise ValueError(
                f'EVT needs at least 50 threshold exceedances; found {len(self.excesses)}. '
                'Use more history or a lower threshold.'
            )
        self.shape, self.scale = fit_tail(self.excesses)
        self.tail_probability = len(self.excesses) / len(values)
        self.body = self.losses[self.losses <= self.threshold]

    def shape_interval(self, resamples=100, seed=42):
        """Bootstrap xi conditional on the selected threshold and independent excesses."""
        if resamples < 20:
            raise ValueError('Use at least 20 bootstrap resamples.')
        rng = np.random.default_rng(seed)
        shapes = []
        for _ in range(resamples):
            try:
                shape, _ = fit_tail(rng.choice(self.excesses, len(self.excesses)))
                shapes.append(shape)
            except ValueError:
                continue
        if len(shapes) < resamples * 0.8:
            raise ValueError('Too many bootstrap tail fits failed to report a xi interval.')
        return tuple(np.quantile(shapes, [0.025, 0.975]))

    def threshold_stability(self):
        """Expose sensitivity of the fitted index to nearby threshold choices."""
        rows = []
        for quantile in sorted(set([0.90, 0.925, 0.95, self.quantile])):
            threshold = np.quantile(self.losses, quantile)
            excesses = self.losses[self.losses > threshold] - threshold
            if len(excesses) >= 50 and threshold > 0:
                shape, scale = fit_tail(excesses)
                rows.append({'Threshold quantile': quantile, 'Exceedances': len(excesses),
                             'Xi': shape, 'Scale': scale})
        return pd.DataFrame(rows)

    def simulate_month(self, scenarios=100_000, sessions=21, seed=42):
        """Sum independent daily log losses, then convert back to simple losses."""
        if not isinstance(scenarios, int) or scenarios < 1:
            raise ValueError('Scenarios must be a positive integer.')
        if not isinstance(sessions, int) or sessions < 1:
            raise ValueError('Sessions must be a positive integer.')
        rng = np.random.default_rng(seed)
        total = np.zeros(scenarios)
        for _ in range(sessions):
            daily = rng.choice(self.body, scenarios)
            tail = rng.random(scenarios) < self.tail_probability
            daily[tail] = self.threshold + stats.genpareto.rvs(
                self.shape, scale=self.scale, size=int(tail.sum()), random_state=rng,
            )
            total += daily
        if not np.isfinite(total).all():
            raise ValueError('The fitted tail produced nonfinite monthly scenarios.')
        # Positive log losses can be arbitrarily large; simple loss stays below 100%.
        with np.errstate(over='raise'):
            try:
                return -np.expm1(-total)
            except FloatingPointError as error:
                raise ValueError('Monthly return simulation overflowed.') from error

    def probability_table(self, scenarios=100_000, sessions=21, seed=42):
        losses = self.simulate_month(scenarios, sessions, seed)
        rows = []
        for level in LOSS_LEVELS:
            events = int(np.count_nonzero(losses >= level))
            lower, upper = binomial_interval(events, scenarios)
            rows.append({
                'Loss at least (%)': level * 100,
                'Probability (%)': events / scenarios * 100,
                'MC lower 95% (%)': lower * 100,
                'MC upper 95% (%)': upper * 100,
                'Simulated events': events,
            })
        return pd.DataFrame(rows).set_index('Loss at least (%)')
