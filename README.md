# My Projects

A collection of Python projects focused on data analysis, statistical modeling, portfolio research, and automation.

## Overview

This repository brings together several experiments and small projects covering:

- financial forecasting and trading strategy analysis
- Bayesian inference and probabilistic estimation
- portfolio risk analysis
- automated reporting workflows

## Featured projects

### auto_reporting
A lightweight automation project for report generation and publishing workflows. It is designed to help streamline the process of creating and sharing reports with a clean, reproducible setup.

For detailed instructions on how to run and configure this project, see the README inside the `auto_reporting/` folder: `auto_reporting/README.md`.

### stock_market_prediction
A machine learning-based stock market project that:

- downloads and prepares financial data
- engineers predictive features
- trains classification models
- evaluates strategy performance through backtesting

### Additional scripts

- `Posterior_estimation.py` — Bayesian estimation of an unknown mean
- `Random_Walk.py` — probabilistic range estimation for random-walk processes
- `portfolio analysis.py` — portfolio return analysis and risk concentration checks

---

## Repository structure

```text
My-Projects/
├── auto_reporting/
│   └── README.md
├── stock_market_prediction/
│   └── README.md
├── Posterior_estimation.py
├── Random_Walk.py
├── portfolio analysis.py
├── README.md
└── ...
```

---

## Notes

- The `stock_market_prediction` folder contains a more detailed project README with deeper implementation notes.
- More information about running `auto_reporting` is available in `auto_reporting/README.md`.

---

## Files description

### 1) `Posterior_estimation.py`
Implements a Bayesian approach to estimate the mean of an unknown distribution by combining prior knowledge with observed data.

### 2) `Random_Walk.py`
Creates approximate 95% confidence bounds for random walk processes to estimate likely future variation.

### 3) `portfolio analysis.py`
Performs portfolio analysis by calculating returns, standard deviations, and concentration of risk using PCA.

## Summary

These projects are small but practical experiments in quantitative analysis, machine learning, and automation. They are useful examples for learning how to work with financial data, statistical modeling, and report generation in Python.
