# Portfolio Lab

A desktop research tool for daily portfolio risk, monthly extreme losses, and
long-only mean-variance analysis. The calculation code is separate from the UI.
All interface labels, source comments, and exports are in English.

## Start the desktop app

From the repository root:

```sh
python3 -m pip install -r requirements-portfolio.txt
python3 portfolio_analysis/run_guy.py
```

`portfolio_analysis/run_guy.py` is the only GUI entry point. Importing `portfolio_analysis`, its modules,
or the launcher does not create a window. The old standalone `portfolio_analysis.py`
has been replaced by the `portfolio_analysis/` package. You can also open the `portfolio_analysis` folder in your editor and run
`run_guy.py`, or launch it from that folder with `python3 run_guy.py`. Do not run
`gui.py` directly: it defines the interface but does not start it. The launcher
resolves imports from its own location, so it also works from another working
directory when passed an absolute path.

Tkinter support must be
available in the Python installation; it is not installed by pip. Matplotlib uses
Tkinter to embed charts in the desktop window.

To explore the app without a connection or market-data download:

```sh
python3 portfolio_analysis/run_guy.py --demo
```

The demo is clearly labeled as synthetic data. It uses a fixed generated sample;
its numbers are not estimates for the real tickers entered in the form.

## Using the interface

1. Enter each ticker and its percentage weight. The initial example is AAPL at
   60% and MSFT at 40%. Add or remove rows as needed. All weights must be positive,
   tickers must be unique, and allocations must total 100%.
2. Enter the portfolio value, history length, number of scenarios, and EVT threshold.
   Defaults are 100,000 monetary units, 3,780 common daily returns, 100,000 scenarios,
   and a 95th-percentile daily log-loss threshold. The random seed is 42.
3. Click **Analyze portfolio** for Yahoo data or **Explore with synthetic data**.
   Downloads and calculations run in a worker thread; progress and results update
   on the main UI thread. Editing the form does not change an analysis already
   running: the displayed results correspond to the submitted allocation.
4. Inspect **Portfolio & daily risk**, **Monthly EVT losses**, and **Efficient
   frontier**. Pages and tables scroll when the available window is smaller.
5. Use **Export results** to save CSV tables, the frontier PNG, and metadata in a
   chosen directory. Existing files with the same names in that directory are
   overwritten. Choose a different folder to preserve earlier analyses.

All holdings must be priced in the same currency. The app does not fetch FX rates;
monetary risk uses the portfolio value in that common currency.

The interface permits 252 or more observations and 1,000–1,000,000 scenarios.
These are software limits, not sample-size guarantees. At a 95% EVT threshold,
roughly 1,000 daily observations are needed just to reach the minimum of 50 tail
exceedances. If EVT cannot be estimated, the app explains why and still shows the
daily risk and frontier. Use more history or a lower threshold where appropriate;
changing the sample or threshold changes the statistical estimate.

## Prices and daily returns

The downloader explicitly requests `auto_adjust=False` and uses `Adj Close` to
account for splits and dividends. Prices are sorted by date and simple returns are
calculated as `current_price / previous_price - 1` before removing missing rows.
Only complete returns common to all holdings are retained. A missing quote is not
forward-filled into a zero return or compressed into a multi-row daily return.
Different exchange closing times can still distort daily dependence estimates.

Internal returns use decimals: `0.01` means 1%. Output converts them to percentages.
The portfolio return is the weighted sum of asset returns, assuming fixed
beginning-of-day weights and daily rebalancing without transaction costs.
The annual arithmetic mean is the daily mean times 252, not CAGR. Annual volatility
is the daily portfolio standard deviation times `sqrt(252)`, incorporating asset
covariance. This scaling assumes no serial correlation. Correlation/PCA calculations
use returns rather than price levels; PCA does not certify diversification.

## Daily copula risk

`EmpiricalCopula` separates marginal return distributions from asset dependence:

1. Convert observations into `rank / (sample_size + 1)`, with average ranks for ties.
2. Fit a Gaussian copula to normal rank scores. For a Student t copula, estimate
   correlation as `sin(pi * Kendall_tau / 2)` and maximize copula likelihood over
   degrees of freedom from 2.01 to 50, holding that correlation fixed.
3. Draw correlated latent values and transform them to uniform percentiles. The
   Student t simulation uses a shared chi-square scale across assets per scenario.
4. Map percentiles through interpolated historical asset quantiles and combine
   asset returns using portfolio weights.

The [SciPy multivariate t documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html)
provides the distribution used in this construction. The Student t copula permits
symmetric joint tail dependence. A small eigenvalue floor stabilizes correlations
for duplicate assets; one asset needs no dependence parameters.

The table compares observed portfolio returns with both simulated models at 95%
and 99% confidence, all for **one day**. Loss is negative return; positive VaR/ES
means loss, while a negative value means gains even at that part of the distribution.
VaR is an inverse-empirical-CDF loss quantile. ES averages the worst tail probability
mass, splitting a boundary observation if necessary so ties do not inflate its size.
Monetary risk equals fractional risk times the supplied portfolio value.

These copula margins cannot exceed historical asset extremes. Their daily estimates
are separate from the EVT monthly model described below.

## EVT: end-of-month loss probabilities

The question answered is:

> What is the model probability that, after 21 trading sessions, portfolio value is
> at least 1%, 3%, 5%, 10%, 15%, 20%, or 25% below its initial value?

This is not a calendar-month forecast or the probability of reaching an intramonth
minimum. The reference value is the portfolio value at the start of the simulation,
not a moving peak.

### Fitting the tail

`ExtremeValueRisk` uses a Peaks Over Threshold (POT) approach on aggregate portfolio
returns, rather than fitting seven unrelated distributions or extrapolating the
daily copula scenarios:

1. Transform daily portfolio returns to log losses: `L = -log(1 + R)`.
2. Choose a high empirical threshold `u`, defaulting to the 95th percentile of `L`.
3. Fit a Generalized Pareto Distribution (GPD) to positive excesses `L - u`, with
   location fixed at zero. Require at least 50 exceedances and validate the fitted
   support. Exceedance frequency is measured from the sample, including any ties.
4. Retain empirical observations at or below `u` as the distribution body. The tail
   receives the observed exceedance probability; the body gets the remaining mass.

The GPD shape **ξ** is the estimated extreme-value index of **daily log losses**:
positive ξ gives a heavy unbounded tail on that scale, zero is the exponential
limit, and negative ξ implies a finite upper endpoint. It is not a direct fit to
monthly percentage losses or block maxima. The POT rationale and tail-index
interpretation are described by [NIST](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/dehaan.htm);
the implementation uses [SciPy's GPD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html).

The interface reports a percentile bootstrap 95% interval for ξ, based on 100
resamples of excesses, plus estimates at nearby threshold quantiles. The bootstrap
holds the selected threshold fixed and assumes independent excesses. It does not
include threshold-selection uncertainty or validate the tail approximation. Strong
threshold sensitivity is a reason to question the result, not to choose the most
attractive probability.

### Constructing a month

For each scenario, draw 21 independent daily log losses from the empirical-body/GPD
mixture. Both ordinary gains and losses remain in the simulation. Sum daily log
losses to get `S`, then compute the month's simple loss as `1 - exp(-S)`. This is
compounding, not daily probability multiplied by 21 or a square-root-of-time rule.
Using log losses permits tail extrapolation while keeping simple losses bounded
by 100% for the positive-value, long-only portfolio model.

The table counts scenarios with monthly loss at least each requested level. All
levels use the same scenarios, so the estimated probabilities are monotone. Alongside
each estimate, the app reports the simulated event count and an exact binomial 95%
interval for **Monte Carlo counting error conditional on the fitted model**. Zero
events still have a positive upper probability bound. For very rare levels, increase
scenarios to improve numerical resolution, rather than treating a rounded zero as
proof the event is impossible.

These counting intervals exclude tail-parameter uncertainty. The separate ξ
bootstrap does not propagate parameter uncertainty to the probability table.
Neither interval captures changing regimes, volatility clustering, estimation bias,
or the limitations of independent-day resampling. There is no GARCH filtering,
declustering, or rolling out-of-sample calibration in this version.

## Efficient frontier

`EfficientFrontier` uses the same aligned asset sample and annualizes arithmetic
means and sample covariance over 252 sessions. It first solves global minimum
variance, then minimizes variance subject to a sequence of expected-return targets.
Only the efficient branch, from minimum variance to the highest estimated asset
mean, is shown. Each portfolio has nonnegative weights summing to one; there is
no leverage, short selling, risk-free asset, or assumed risk-free rate.

Optimization uses [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html),
with analytical gradients, a scaled objective, and checks on convergence and
feasibility. Single assets and equal expected returns can produce one efficient
point instead of an artificial curve. Singular covariance from duplicate assets
is covered by the tests.

The plot marks the entered portfolio, minimum-variance allocation, individual assets,
and a cloud of feasible allocations. Tables compare current and minimum-variance
weights and statistics. The exported curve contains weights for each optimized
point. The frontier PNG can be used separately from the application.

This is historical mean-variance analysis. Expected returns are noisy estimates,
not forecasts. A frontier allocation does not optimize EVT tail loss, and the
monthly probability table continues to describe the user's submitted portfolio.

## Exports

- `portfolio_summary.csv`: annual mean and volatility per asset and portfolio.
- `daily_risk.csv`: historical and copula one-day VaR/ES, including monetary values.
- `monthly_evt_probabilities.csv`: the seven terminal loss probabilities, event
  counts, and conditional Monte Carlo intervals, when EVT is available.
- `evt_threshold_stability.csv`: tail-index sensitivity to threshold choices.
- `efficient_frontier.csv`: annual frontier statistics and optimized asset weights.
- `efficient_frontier.png`: the frontier chart.
- `analysis_metadata.json`: submitted weights, dates, value, sample size, scenario
  count, seed, synthetic-data flag, EVT parameters, ξ interval, and assumptions.

## Code and verification

The package is organized into `core.py` for returns and daily copula risk, `evt.py`
for tail estimation, `frontier.py` for optimization, `service.py` for running a
complete analysis, and `gui.py` for desktop presentation. `run_guy.py` owns window
creation and the event loop.

Run the calculation and service tests without downloading prices or opening windows:

```sh
python3 -m unittest discover -s tests -v
```

Three desktop tests are skipped by default. On a machine with a working Tk display,
run them explicitly to verify input validation, background analysis, chart rendering,
CSV/PNG export, and recovery from download errors:

```sh
RUN_PORTFOLIO_GUI_TESTS=1 python3 -m unittest discover -s tests -p test_portfolio_gui.py -v
```

Numerical checks include a known GPD shape, analytical one-day tail probabilities,
21-session compounding, probability monotonicity, bootstrap reproducibility,
closed-form two-asset minimum variance, and frontier feasibility. Synthetic samples
and a real-price AAPL/MSFT calculation have also exercised the complete workflow;
these are implementation checks, not evidence of forecasting accuracy.
