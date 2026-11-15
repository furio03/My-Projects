"""Long-only mean-variance optimization on the same returns used for risk analysis."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .core import TRADING_DAYS


class EfficientFrontier:
    """Minimum variance for each target return, with nonnegative weights summing to one."""

    def __init__(self, returns):
        if (len(returns) < 2 or returns.shape[1] < 1 or returns.columns.has_duplicates
                or not np.isfinite(returns.to_numpy()).all()):
            raise ValueError('The frontier requires finite returns with unique assets.')
        self.assets = list(returns.columns)
        self.mean = returns.mean().to_numpy() * TRADING_DAYS
        self.covariance = returns.cov().to_numpy() * TRADING_DAYS
        self.dimension = len(self.assets)
        self.spread = np.ptp(self.mean)
        self.normalized_mean = (self.mean - self.mean.min()) / self.spread if self.spread > 1e-10 else np.zeros(self.dimension)
        # Scaling the objective improves convergence without changing the optimum.
        self.scaled_covariance = self.covariance / max(np.diag(self.covariance).max(), 1e-12)
        self.minimum_weights = self.solve(np.full(self.dimension, 1 / self.dimension))

    def solve(self, initial, target=None):
        if self.dimension == 1:
            return np.ones(1)
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1,
                        'jac': lambda w: np.ones(self.dimension)}]
        if target is not None and self.spread > 1e-10:
            normalized_target = (target - self.mean.min()) / self.spread
            constraints.append({
                'type': 'ineq', 'fun': lambda w: w @ self.normalized_mean - normalized_target,
                'jac': lambda w: self.normalized_mean,
            })
        fit = minimize(
            lambda w: w @ self.scaled_covariance @ w, initial,
            jac=lambda w: 2 * self.scaled_covariance @ w,
            bounds=[(0, 1)] * self.dimension, constraints=constraints,
            method='SLSQP', options={'ftol': 1e-11, 'maxiter': 500},
        )
        if not fit.success:
            raise ValueError(f'Efficient frontier optimization failed: {fit.message}')
        weights = np.clip(fit.x, 0, 1)
        weights /= weights.sum()
        if target is not None and weights @ self.mean < target - 1e-7:
            raise ValueError('The optimizer did not meet the requested return target.')
        return weights

    def point(self, weights):
        if isinstance(weights, (dict, pd.Series)):
            weights = pd.Series(weights).reindex(self.assets).to_numpy()
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (self.dimension,) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-8)):
            raise ValueError('Frontier weights must match assets, be nonnegative, and total one.')
        return {
            'Annual volatility (%)': np.sqrt(max(weights @ self.covariance @ weights, 0)) * 100,
            'Annual return (%)': weights @ self.mean * 100,
        }

    def curve(self, points=50):
        if not isinstance(points, int) or points < 2:
            raise ValueError('Use at least two requested frontier points.')
        minimum_return = self.minimum_weights @ self.mean
        maximum_return = self.mean.max()
        targets = np.linspace(minimum_return, maximum_return, points)
        if maximum_return - minimum_return <= 1e-10:
            targets = np.array([minimum_return])
        maximum_weights = np.eye(self.dimension)[np.argmax(self.mean)]
        rows = []
        for target in targets:
            if target == minimum_return:
                weights = self.minimum_weights
            else:
                fraction = (target - minimum_return) / (maximum_return - minimum_return)
                initial = (1 - fraction) * self.minimum_weights + fraction * maximum_weights
                weights = self.solve(initial, target)
            rows.append({**self.point(weights), **{
                f'Weight {asset}': weight for asset, weight in zip(self.assets, weights)
            }})
        return pd.DataFrame(rows)

    def opportunity_set(self, scenarios=1000, seed=42):
        weights = np.random.default_rng(seed).dirichlet(np.ones(self.dimension), scenarios)
        volatility = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', weights, self.covariance, weights), 0))
        return pd.DataFrame({'Annual volatility (%)': volatility * 100,
                             'Annual return (%)': weights @ self.mean * 100})
