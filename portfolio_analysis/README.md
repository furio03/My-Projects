# Portfolio analysis

Desktop portfolio research with daily copula VaR/ES, extreme-value estimates of
21-session terminal loss probabilities, and a long-only efficient frontier.

From the repository root:

```sh
python3 -m pip install -r requirements-portfolio.txt
python3 portfolio_analysis/run_guy.py
```

Use `python3 portfolio_analysis/run_guy.py --demo` for an explicitly labeled synthetic-data example.
If your terminal is already inside `portfolio_analysis`, use `python3 run_guy.py`
instead. In VS Code, open **run_guy.py** and press **Run Python File**. Do not run
`gui.py` directly.

The GUI opens only through `run_guy.py`; package imports do not create windows.

See the [full portfolio guide](../docs/portfolio_analysis.md) for inputs, EVT fitting,
uncertainty, frontier constraints, exported files, and numerical/desktop tests.

| Module | Responsibility |
|---|---|
| `run_guy.py` | Desktop launcher; the only file that creates the application window |
| `core.py` | Aligned returns, portfolio statistics, Gaussian/Student t copulas, daily VaR and ES |
| `evt.py` | GPD tail index, threshold diagnostics, bootstrap, monthly loss probabilities |
| `frontier.py` | Long-only mean-variance optimization and feasible allocations |
| `service.py` | Validated settings, data loading, demo data, complete calculation workflow |
| `gui.py` | Desktop inputs, background execution, result tables, embedded plots, exports |
