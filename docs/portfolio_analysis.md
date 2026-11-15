# Portfolio analysis

Run the script from the repository root:

```sh
/usr/local/bin/python3 -m pip install -r requirements-portfolio.txt
/usr/local/bin/python3 portfolio_analysis.py --value 100000
```

Enter each ticker and its percentage allocation. For example, enter `AAPL`, `60`,
then `MSFT`, `40`. Allocations must be positive and total 100%. The optional
`--value` argument adds monetary losses in the same currency as the portfolio.
Without it, the report contains percentages only.

Defaults are 3,780 common daily returns, 100,000 scenarios per copula, and seed 42.
For a shorter sample or a quicker run:

```sh
/usr/local/bin/python3 portfolio_analysis.py --observations 1260 --scenarios 20000 --seed 7
```

A shorter history changes the estimate, not just the runtime. Use `--help` for all
options. Only the portfolio's assets are downloaded; benchmark indices are no
longer included in the risk calculation.

## Data and returns

Yahoo adjusted closes account for splits and dividends. The script requests them
explicitly with `auto_adjust=False`. It sorts dates, calculates simple returns as
`current_price / previous_price - 1`, and then selects the latest common returns.
It does not fill missing prices: a return is excluded if either quote is missing
for any asset. Thus a missing quote is not turned into a zero return or a return
spanning several rows. An insufficient common history produces a clear error.

Calculations use decimal returns internally: `0.01` means 1%. Output converts them
to percentages. All assets must already be quoted in the same currency; there is
no FX conversion. Different exchange closing times can still affect measured
daily dependence.

The portfolio return is the weighted sum of asset returns. Weights represent
beginning-of-day allocations; historical calculations assume daily rebalancing
without transaction costs. The annual mean is the daily arithmetic mean times
252, not CAGR. Annual volatility is the daily portfolio standard deviation times
`sqrt(252)`, so it includes covariance between assets. This scaling assumes no
serial correlation. Correlation and PCA use asset returns, not price levels. PCA
is a description of linear concentration, not a diversification guarantee.

## How the simulations work

`EmpiricalCopula` separates the shape of each asset's return distribution from
its dependence on the other assets:

1. Convert each historical return to `rank / (sample_size + 1)`. Average ranks
   handle ties, and the denominator keeps values away from zero and one.
2. Fit a Gaussian copula using correlations between normal rank scores. Fit a
   Student t copula using `sin(pi * Kendall_tau / 2)` for its correlation matrix,
   then maximize the copula log likelihood over degrees of freedom from 2.01 to
   50 with that matrix fixed. This is a two-step fit, not joint maximum likelihood.
3. Draw correlated latent normal values. For the t copula, divide all asset values
   in each scenario by one shared chi-square scale. Apply the corresponding
   univariate CDF to obtain simulated percentiles.
4. Map each asset's percentiles through its historical quantiles, using linear
   interpolation between observations. Combine the simulated returns with the
   portfolio weights to obtain one daily portfolio return per scenario.

Correlation matrices receive a small eigenvalue floor for numerical stability,
including when two assets have identical histories. With one asset, both copulas
reduce to the same marginal simulation; dependence parameters are not estimated.
The report prints the fitted t degrees of freedom. Values near a fit bound should
be read as a limitation of the estimate, rather than a precise tail measurement.

The model follows the multivariate t construction documented by
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).
The t copula can represent joint extremes but assumes symmetric dependence in the
upper and lower tails. Both copulas are static over the selected sample.

## Reading VaR and ES

The report compares historical losses with Gaussian and Student t copula losses
at 95% and 99% confidence, all for a **one-day horizon**.

- Loss is the negative of the portfolio return. Positive VaR or ES means a loss;
  negative values mean even that part of the sample contains gains.
- VaR is the empirical loss quantile, using the inverse empirical CDF. A 99% VaR
  of 3% is the model's 99th percentile of daily losses, not a maximum loss.
- Expected Shortfall (ES) averages the worst 5% or 1% of probability mass. A
  fractional boundary observation is included when necessary, so ties at VaR do
  not accidentally expand the tail being averaged.
- Monetary risk is percentage risk, expressed as a fraction, times `--value`.

The historical row uses observed weighted returns directly. The copula rows use
simulated returns. More simulations reduce Monte Carlo noise; they do not create
new historical evidence. Interpolated empirical margins cannot generate returns
outside each asset's historical range, although joint combinations can be new.

These are in-sample risk estimates. There is no rolling out-of-sample backtest,
volatility model, asymmetric copula, multi-day scaling, or extrapolation of marginal
tails in this version. Model comparison here does not select or validate a winner.

## Code and verification

- `prepare_returns`: validate prices and align daily returns.
- `EmpiricalCopula`: fit dependence and simulate asset returns.
- `PortfolioAnalysis`: aggregate weights, summarize returns, and build the risk table.
- `loss_risk`: calculate loss VaR and ES independently of the model.
- `main`: read arguments and allocations, download prices, and print the report.

Importing the module does not prompt for input or download anything. Offline tests
cover known price moves, missing quotes, covariance, loss signs, tail boundaries,
reproducibility, empirical margins, t dependence, single and duplicate assets, and
the command-line reporting flow:

```sh
/usr/local/bin/python3 -m unittest discover -s tests -v
```
