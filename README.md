# Quantitative Finance and Data Science Projects

Python projects covering portfolio risk, financial time-series classification,
automated statistical reporting, and probability simulations. The emphasis is on
making the analysis interpretable: how inputs become model estimates, how those
estimates are evaluated, and what conclusions the results can support.

The first two projects focus on quantitative research and model evaluation. The
reporting application connects statistical analysis with an interface for business
users. The final two projects are smaller experiments I built during the first two
years of my bachelor's degree. I keep them here as a record of where I started and how my
approach to statistics and programming has developed since then.

| Project | Main question | Methods and outputs |
|---|---|---|
| [Portfolio analysis](#1-portfolio-analysis) | How does dependence between assets affect portfolio losses? | Copulas, EVT/Pareto tails, Monte Carlo, VaR/ES, monthly loss probabilities, efficient frontier |
| [Stock market prediction](#2-stock-market-prediction) | Can historical market features classify the next session's price movement? | Feature engineering, XGBoost, Random Forest, K-Means, temporal validation, trading simulation |
| [Automated reporting](#3-automated-reporting) | How can a tabular dataset become an interpretable analytical report? | Data preparation, exploratory statistics, clustering, influence diagnostics, LLM-assisted interpretation, PDF retrieval |
| [Prior-weighted density estimation](#4-prior-weighted-density-estimation) | How does a prior change the shape of an estimated density? | Gaussian KDE, numerical integration, density weighting, interval estimation |
| [Random-walk simulation](#5-random-walk-simulation) | How does uncertainty accumulate over repeated independent steps? | Monte Carlo paths, variance accumulation, central limit approximation |

## 1. Portfolio analysis

**Purpose.** Analyze a user-defined portfolio through three complementary views:
daily losses under different dependence assumptions, end-of-month extreme-loss
probabilities, and the mean-variance trade-off between alternative allocations.
Rather than treating each holding in isolation, the analysis models their joint
behavior and aggregates simulated returns using the portfolio weights.

**Data and methodology.** The application downloads adjusted historical prices from
Yahoo Finance, calculates simple returns, and selects a common chronological sample
across holdings. Missing prices are not forward-filled. The default sample contains
3,780 common daily returns, with a configurable history length.

The analysis combines:

- **Return and volatility estimation:** annualized arithmetic mean returns and
  portfolio volatility calculated from weighted daily returns, including covariance.
- **Linear dependence diagnostics:** asset return correlations and PCA explained
  variance, showing how much standardized variation is concentrated in common factors.
- **Copula modeling:** a Gaussian copula fitted to normal rank scores and a Student t
  copula with correlation estimated from Kendall's tau. The t degrees of freedom
  are fitted by copula likelihood with the correlation matrix held fixed.
- **Monte Carlo simulation:** 100,000 reproducible scenarios per copula by default.
  Simulated percentiles are mapped through each asset's historical quantiles, then
  combined into portfolio returns. This separates the marginal distributions from
  the dependence model.
- **Loss measurement:** one-day Value at Risk and Expected Shortfall at 95% and 99%,
  compared with a direct historical estimate. Monetary losses are available when a
  portfolio value is supplied.
- **Extreme Value Theory:** a Peaks Over Threshold model fits a Generalized Pareto
  Distribution to extreme daily portfolio log losses. It estimates the tail index
  ξ, provides a conditional bootstrap interval, and exposes threshold sensitivity.
  An empirical body plus the fitted tail generates 21-session compounded paths.
- **Monthly loss probabilities:** a table estimates the probability of finishing
  21 trading sessions at least **1%, 3%, 5%, 10%, 15%, 20%, or 25% below the initial
  portfolio value**. Event counts and exact Monte Carlo intervals accompany each
  estimate, so zero simulated events are not presented as proof of zero risk.
- **Efficient frontier:** constrained mean-variance optimization finds long-only,
  fully invested allocations from the global minimum-variance point to the highest
  estimated asset return. The chart locates the current portfolio and individual
  assets, with optimized weights available in the exported curve.

**How to interpret the output.** VaR is a quantile of daily losses; Expected Shortfall
measures the average loss in the corresponding worst tail. Comparing the copula
estimates shows how the dependence assumption changes estimated portfolio risk
while retaining the same empirical asset distributions. The t copula can represent
joint tail events. PCA provides a complementary view of linear concentration;
it does not establish that a portfolio is adequately diversified.

The desktop interface brings these results into separate daily-risk, monthly-EVT,
and efficient-frontier views. It supports editable holdings, configurable samples,
a synthetic-data demo, and CSV/PNG exports with analysis metadata. Downloads and
calculations run in the background so the window stays responsive.

**How to read the monthly results.** These are terminal losses relative to today's
value, not the probability of touching a drawdown during the month. The EVT model
fits the aggregate portfolio's daily log losses; its ξ is not a direct estimate of
the shape of monthly percentage losses. Monthly scenarios assume independent days
and daily rebalancing. The daily copula comparison is a separate model.

**Scope.** Estimates use same-currency historical assets and static model parameters.
Copula margins stay within historical asset extremes; the separate EVT tail can
extrapolate beyond observed daily log losses. The Monte Carlo probability intervals
cover counting error only, while the ξ bootstrap is conditional on the selected
threshold and independent exceedances. Neither resolves model uncertainty,
volatility clustering, or changing regimes. The efficient frontier uses uncertain
historical means and covariance, not forward-looking return forecasts. FX
conversion, transaction costs, and out-of-sample risk backtesting are not implemented.

**Stack:** NumPy, pandas, SciPy, yfinance, Tkinter, Matplotlib.

[Source](portfolio_analysis/) · [Desktop launcher](portfolio_analysis/run_guy.py) ·
[Methodology, assumptions, and usage](docs/portfolio_analysis.md)

```sh
python3 -m pip install -r requirements-portfolio.txt
python3 portfolio_analysis/run_guy.py
```

The desktop window opens **only** through `portfolio_analysis/run_guy.py`. Importing the package or
its modules does not open a GUI. For an offline demonstration:

```sh
python3 portfolio_analysis/run_guy.py --demo
```

## 2. Stock market prediction

**Purpose.** Build and evaluate a supervised classification strategy for financial
time series, following the full path from OHLCV data to dated trading instructions.
The target is the next session's close-to-close return: a configurable percentage
threshold separates `buy`, `hold`, and `sell` outcomes.

**Features and models.** Historical market data is transformed into moving averages,
exponentially weighted averages, RSI, a MACD-style oscillator, rolling dispersion,
price/volume comparisons, consecutive movement counts, and Chaikin Money Flow.
A standardized K-Means representation adds a market-profile feature. The classifier
is XGBoost by default, with Random Forest available as a baseline.

**Evaluation design.** Model evaluation follows the order in which information
would become available:

1. Construct causal indicators and lag model inputs by one trading session. Keep
   the actual execution date and close separate from those delayed inputs.
2. Divide observations into chronological training, validation, and final test
   periods. Purge one target row at each boundary because its label uses the next
   session's price.
3. Tune classifier parameters and cluster count with expanding-window temporal
   cross-validation, a one-session gap, and macro F1 scoring. Scaling and clustering
   are fitted independently inside each training fold.
4. Choose the return-label threshold using validation macro precision. Refit on
   pre-test data, then evaluate the selected configuration once on the final test.
5. Save the complete fitted pipeline, feature order, label encoder, and date cutoffs.
   Deployment reuses these objects without refitting on prediction data.

**Trading simulation and interpretation.** Classification reports describe how well
the model distinguishes the three outcomes. A separate execution simulation translates
signals into fixed-notional long or short entries, subject to capital availability
and a maximum of three open entries. Opposite signals close the existing side;
hold signals do not release capacity. Reports separate realized and unrealized
profit, cash, exposure, and equity, so open final positions remain visible.

This makes it possible to distinguish classification quality from economic results:
a higher classification score does not, by itself, establish a profitable strategy.
The backtest assumes execution at the dated close and omits fees, slippage, borrowing
costs, and broker margin rules. It is a research simulation, without a demonstrated
claim of live profitability. The checked-in legacy model and performance CSVs predate
the corrected workflow and must be regenerated before use.

**Stack:** pandas, NumPy, scikit-learn, XGBoost, joblib; yfinance for data acquisition.

[Source and project guide](stock_market_prediction/README.md) ·
[Training](stock_market_prediction/main.py) ·
[Forecasting pipeline](stock_market_prediction/forecasting.py) ·
[Backtest](stock_market_prediction/back_test.py)

```sh
python3 -m pip install -r stock_market_prediction/requirements.txt
python3 stock_market_prediction/main.py --iterations 10
python3 stock_market_prediction/deploy_model.py
```

## 3. Automated reporting

**Purpose.** Turn a CSV dataset and a user's analytical objective into exploratory
statistics, visualizations, and a written report. The project explores how computed
evidence can be made accessible to a reader who needs to understand a dataset
without writing the analysis themselves.

**Statistical workflow.** The Python engine handles variable-type detection and
missing-data treatment, including iterative numeric imputation and simpler
median/mode approaches. It calculates numerical summaries, quantiles, categorical
frequencies, date ranges, and correlations. Additional components include:

- **Segmentation:** standardized K-Means for numerical profiles, plus a custom
  K-Medoids implementation with Manhattan distance for encoded categorical/ordinal
  profiles. Candidate cluster counts are compared using silhouette scores.
- **Association analysis:** pairwise chi-square tests of categorical variables,
  used as exploratory signals for relationships worth investigating.
- **Outlier and influence diagnostics:** regression residual diagnostics and Cook's
  distance, with an IQR fallback for a single numerical variable. The workflow
  separates flagged observations from the main sample and can generate a separate
  analysis when the flagged subset is sufficiently large.
- **Domain-specific analysis:** modules for business contexts such as retail,
  e-commerce, and HR, with the available analyses depending on the dataset's columns
  and the maturity of the corresponding module.

**LLM and retrieval layer.** Groq-hosted language models help identify the dataset's
domain, map column names to analytical roles, interpret the user's objective, and
write the narrative. Numerical summaries are computed in Python and supplied as
context. The Gradio interface presents results and exports reports with charts
using ReportLab.

A separate retrieval-augmented generation workflow lets users query PDF reports:
`pdfplumber` extracts text, overlapping chunks are embedded in ChromaDB, cosine
similarity retrieves relevant passages, and the LLM receives those passages with
source filenames to construct an answer.

**How to interpret the output.** The report combines a statistical description of
the selected data with an automatically generated interpretation. Clusters describe
similar profiles; associations and influential observations suggest questions for
further analysis. They do not establish causal relationships. The retrieval feature
supports inspecting report content, rather than generating a new statistical study.

**Scope.** This is a prototype with uneven coverage across domain modules. The
current main workflow analyzes at most the first 500 rows and columns at positions
2 through 7, so its output does not describe an arbitrary uploaded dataset in full.
Missingness heuristics and LLM interpretations require review. An experimental
Laplace-noise component perturbs selected numerical domain results; it does not
establish an end-to-end differential privacy guarantee for the application.
The repository also contains a separate experimental partial-order clustering
module, which is not part of the default reporting workflow.

**Stack:** pandas, NumPy, SciPy, scikit-learn, Matplotlib/Seaborn, Gradio, ReportLab,
Groq, ChromaDB, pdfplumber, diffprivlib.

[Application](auto_reporting/app/website.py) · [Analysis engine](auto_reporting/main.py) ·
[PDF retrieval](auto_reporting/LLM/RAG.py) ·
[Example generated report](auto_reporting/app/reports%20generated/revenue_analysis.pdf) ·
[Project setup notes](auto_reporting/README.md)

After installing the application's dependencies and setting `GROQ_API_KEY`, launch
it from its project directory so relative resource paths resolve correctly:

```sh
cd auto_reporting
python3 app/website.py
```

## 4. Prior-weighted density estimation

*A smaller project built during the first two years of my bachelor's degree,
kept as a reminder of where I started.*

**Purpose.** Explore how an estimated data density changes when it is weighted by
a prior expressing a different initial location. The script generates 3,000 normal
observations with a known mean, estimates their density using Gaussian kernel
density estimation, and combines it with a normal prior on a numerical grid.

**Techniques and outputs.** The product is normalized by numerical integration.
The script then calculates its mode, an equal-tail 95% interval, and plots the
observations, KDE, prior, and resulting weighted density. Comparing these curves
shows how the prior pulls weight toward its preferred region and changes the
shape of the estimate.

**Interpretation.** This is an exploratory density-weighting experiment. The KDE
estimates the distribution of observations; it is not a likelihood for their
unknown mean. Consequently, the resulting curve and interval should not be treated
as a formal Bayesian posterior and credible interval for that mean. A rigorous
parameter-inference version would need an explicit sampling likelihood.

**Stack:** NumPy, SciPy, Matplotlib.

[Source: `Posterior_estimation.py`](Posterior_estimation.py)

## 5. Random-walk simulation

*A smaller project built during the first two years of my bachelor's degree,
kept as a reminder of where I started.*

**Purpose.** Visualize how uncertainty grows when independent, zero-mean increments
are accumulated over time. The script simulates five paths, each with 100,000
steps drawn uniformly from `{-1, 0, 1}`.

**Techniques and outputs.** Each increment has variance `2/3`, so after `t` steps
the walk has variance `2t/3`. The plotted reference envelope is approximately
`±2 × sqrt(2t/3)`, illustrating square-root-of-time uncertainty growth and the
normal approximation for the distribution at a fixed horizon.

**Interpretation.** The paths illustrate how large excursions can arise without a
trend in the underlying step distribution. The approximate 95% envelope applies
to individual horizons; it is not a 95% guarantee that an entire path stays inside
the boundaries. This distinction matters when moving from a single-horizon risk
estimate to monitoring a process repeatedly through time.

**Stack:** Python standard library, Matplotlib.

[Source: `Random_Walk.py`](Random_Walk.py)

## Repository layout

```text
.
├── portfolio_analysis/           # Returns, copulas, EVT, frontier and desktop UI
│   └── run_guy.py                # The desktop GUI entry point
├── requirements-portfolio.txt
├── docs/portfolio_analysis.md    # Risk methodology and assumptions
├── stock_market_prediction/      # Temporal ML pipeline and trading simulation
├── auto_reporting/               # Statistical reporting, PDF export and RAG
├── Posterior_estimation.py        # Exploratory prior-weighted density experiment
├── Random_Walk.py                # Stochastic paths and uncertainty envelopes
└── tests/                        # Portfolio and forecasting regression tests
```

The portfolio and forecasting tests use deterministic examples and synthetic data
to check return calculations, temporal boundaries, model persistence, risk measures,
and execution accounting without downloading market data. Separate opt-in desktop
tests exercise input validation, background calculation, charts, exports, and error
recovery. See the portfolio guide for running those tests with a display:

```sh
python3 -m unittest discover -s tests -v
```
