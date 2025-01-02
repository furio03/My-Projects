import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Observed data (simulated)
true_mean = 2
observations = np.random.normal(loc=true_mean, scale=1, size=3000)  # Data generated from a normal distribution
num_obs = len(observations)  # Number of observations

# Density estimation with KDE
kde = gaussian_kde(observations, bw_method=0.3)
mu_grid = np.linspace(-6, 10, num_obs)  # Grid of mu values for evaluation
kde_density = kde(mu_grid)

# Normal prior (mean = 0, variance = 3)
prior_mean = 0  # Mean of the prior
prior_std = 3  # Standard deviation of the prior
prior_density = (1 / (np.sqrt(2 * np.pi * prior_std**2))) * np.exp(-0.5 * ((mu_grid - prior_mean) ** 2) / prior_std**2)

# Posterior distribution (product of KDE and prior)
posterior_density = kde_density * prior_density

# Normalize posterior to make it a valid probability distribution
posterior_area = np.trapz(posterior_density, mu_grid)  # Area under the curve (trapezoidal method)
posterior_density /= posterior_area  # Normalize to ensure total area = 1

# Compute 95% credible interval (percentiles)
cumulative_posterior = np.cumsum(posterior_density) * (mu_grid[1] - mu_grid[0])  # Numerical integration for the cumulative
lower_credible = mu_grid[np.searchsorted(cumulative_posterior, 0.025)]  # 2.5% quantile
upper_credible = mu_grid[np.searchsorted(cumulative_posterior, 0.975)]  # 97.5% quantile

# Create a figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the data distribution (histogram and KDE)
axes[0].hist(observations, bins=30, density=True, alpha=0.5, color="gray", label="Data histogram")
axes[0].plot(mu_grid, kde_density, label="KDE estimate", color="blue")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Probability density")
axes[0].set_title("Data Distribution")
axes[0].legend()

# Plot the posterior distribution
axes[1].plot(mu_grid, posterior_density, label="Posterior distribution", color="red")
axes[1].fill_between(mu_grid, 0, posterior_density, where=(mu_grid >= lower_credible) & (mu_grid <= upper_credible), 
                     color="red", alpha=0.2, label="95% credible interval")

# Overlay the prior distribution
axes[1].plot(mu_grid, prior_density, label="Normal prior", color="green", linestyle="--")

axes[1].set_xlabel("Mean value")
axes[1].set_ylabel("Probability density")
axes[1].set_title("Prior vs Posterior")
axes[1].legend()

# Display the figure
plt.tight_layout()
plt.show()

# Print the credible interval
print(f"95% credible interval: ({round(lower_credible, 3)}, {round(upper_credible, 3)})")

# Find the maximum of the posterior distribution (MAP)
map_index = np.argmax(posterior_density)
map_estimate = mu_grid[map_index]

# Print the results
print('MAP estimator:', round(map_estimate, 3))
print('True mean:', true_mean)
print("Hypothesized mean before observing data (prior mean):", prior_mean)
