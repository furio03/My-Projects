Files description:
1) posterior_estimation.py: Uses a Bayesian framework to estimate an unknown parameter.
The script demonstrates how, within Bayesian inference, we can combine prior knowledge and observed data to make the best probabilistic estimate of the parameter.

3) random_walk.py: Creates 95% upper and lower bounds for random walk processes, which will contain approximately 95% of future observations.
This program illustrates that, although it is impossible to predict the exact future of such processes, it is still possible to estimate their likely range of variation.

3) portfolio_analysis.py: Performs standard portfolio analysis by calculating returns and standard deviations of individual positions and the overall portfolio.
The most interesting part is at the end, where Principal Component Analysis (PCA) is applied to detect concentration of risk in the portfolio.
The script generates an alert if a large portion of the portfolio’s volatility is attributed to a few assets.
