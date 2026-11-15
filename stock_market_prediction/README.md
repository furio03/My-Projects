# Stock market prediction

Train a next-session buy/hold/sell classifier, select its settings on a separate
validation period, and evaluate the chosen strategy on a final chronological test.
Code, prompts, and reports are in English.

## Run

From the repository root:

```sh
python3 -m pip install -r stock_market_prediction/requirements.txt
python3 stock_market_prediction/main.py --iterations 10
python3 stock_market_prediction/deploy_model.py
```

Training uses XGBoost by default. `--model rf` selects a Random Forest baseline.
`--folds`, `--iterations`, and `--jobs` control the search cost. The defaults are
10 folds, 30 parameter combinations, and one job. `--csv` accepts a different OHLCV
history, and `--output-dir` changes where artifacts and reports are saved. Paths
otherwise resolve relative to the script directory, not the shell's current folder.

The default dataset is the checked-in AAPL history. Prices must be valid, unique,
chronological trading sessions in one currency. Yahoo's optional ticker header row
is ignored when loading the existing CSV. No data download happens during training.

## Timing and evaluation

The target on date t describes Close[t+1] / Close[t] - 1. A threshold of 0.3 means
buy above +0.3%, sell below -0.3%, and hold otherwise. The last target is unknown.

Features used for date t are lagged by one session. The execution price is the
actual Close[t], not the lagged close. This intentionally uses information available
before date t's close and assumes execution at that close without slippage. The
model predicts the following close-to-close move using those delayed inputs.

Rolling indicators are causal and use the preceding history across split boundaries.
The last 365 rows form the final test, and the preceding 365 rows form validation;
`--test-size` and `--validation-size` change those lengths. One target row is purged
before each boundary because its label uses the next session's price.

Thresholds 0.2, 0.3, and 0.4 are compared only on validation macro precision. Each
candidate uses expanding-window temporal cross-validation with a one-session gap
and macro F1 scoring. Cluster count and classifier settings are searched together.
Every fold fits its own scaler, K-Means, and classifier. Early folds lacking any
training class are excluded; at least two usable folds are required.

After selecting the threshold, the model is refitted and tuned on all pre-test
observations, with the same boundary purge. The final test is then evaluated once.
Training accuracy is explicitly in-sample. The final unknown target is predicted
but excluded from classification metrics.

## Saved models and deployment

The artifact includes the fitted classifier pipeline, the training scaler and
K-Means centroids, the feature order, label encoder, selected settings, and date
cutoffs. Deployment reuses them without fitting anything. It uses earlier prices
only for indicator context and predicts dates at or after the saved test start,
always later than the last training observation. Predictions for the final session
are available even though its outcome is not yet known.

Legacy `best_model.pkl` files do not contain this pipeline and must be regenerated
by running the new training script. The checked-in old model and performance CSVs
are historical artifacts, not results of the corrected workflow. Training writes
`best_model.pkl` and `performance.csv`; deployment also writes `predictions.csv`
and `performance_optimized.csv`. Use a separate `--output-dir` to keep old files.

The cluster scaler is fitted once on training observations and stored with the
centroids. Predicting one row or a batch with unrelated outliers does not change
that row's scale or group assignment.

## Backtest accounting

Each accepted entry adds a fixed 1,000 monetary units at that session's
actual close. An opposite signal closes the existing side; it does not also open
the opposite side on the same row. Hold leaves the position unchanged.

The default initial capital is 3,000 units. `realized_profit` includes closed trades,
`unrealized_profit` marks the current position at each close, and `cumulative_profit`
is their sum. `cash` records financing flows, `exposure` is signed market value,
and `equity = cash + exposure = initial_cash + cumulative_profit`. An open final
position remains marked to market instead of disappearing from the result.
Short positions have negative exposure. There are no fees, slippage, borrow costs,
interest, dividends credited separately, or margin calls in this accounting model.

The limit is three currently open entries, including entries separated by hold
signals. Additional signals become `executed_label = hold` when that limit or
available capital prevents entry. Opposite signals close the whole position and
release its reserved capital; a hold does neither. `open_entries`, `reserved_capital`,
and `free_capital` expose these constraints. Both the baseline and filtered backtest
enforce the same limits.

Each entry reserves its initial notional on both long and short trades. Available
capital is initial cash plus realized P&L minus reserved notional; short-sale
proceeds cannot finance additional entries. This is an explicit fixed-notional
budget, not a broker margin model or a guarantee that short losses are bounded.

Deployment filters scores below `--min-score` (default 0.5), independently of the
percentage threshold used to construct target labels. The filter uses the same
execution engine as the backtest, so holds cannot reset a separate position counter.

## Files and tests

- `main.py`: read prices, prepare delayed inputs, select settings, and save the model.
- `forecasting.py`: temporal model search, fitted pipeline prediction, and metrics.
- `indicators.py`: causal rolling features and fitted cluster features.
- `label.py`: next-session labels, leaving the final outcome unknown.
- `data_manipulation.py`: lag predictors without moving execution prices or dates.
- `deploy_model.py`: load the artifact and predict without refitting.
- `back_test.py`: execute dated signals and report strategy P&L.
- `optimize_classification.py`: filter low-confidence signals and position additions.

Run the deterministic tests without downloading data:

```sh
python3 -m unittest discover -s tests -v
```
