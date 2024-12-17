import random as rd
import matplotlib.pyplot as plt
import math

# Number of random walks
num_walks = 5  

# Initial standard deviation
initial_sigma = math.sqrt(2/3)

# Plot setup
plt.figure(figsize=(10, 6))

# Loop to execute and plot multiple random walks
for walk in range(num_walks):
    step = 0
    cumulative_sum = 0
    y_values = [0]  # List for Y-axis values (random walk)
    x_values = []   # List for X-axis values (iterations)
    sigma_upper = [0]  # List for positive standard deviation boundary
    sigma_lower = [0]  # List for negative standard deviation boundary

    # Run the random walk for a given number of iterations
    while step < 100000:  # Number of iterations
        cumulative_sum += rd.randint(-1, 1)
        y_values.append(cumulative_sum)
        x_values.append(step)

        if step > 0:
            sigma = 2 * initial_sigma * math.sqrt(step)  
            # According to the Central Limit Theorem, after ~30 iterations, 
            # the random walk can be approximated by a normal distribution.
            sigma_upper.append(sigma)   # Using 2 * std to include ~95% of values within the bounds
            sigma_lower.append(-sigma)
        else:
            sigma_upper.append(0)
            sigma_lower.append(0)
        
        step += 1

    # Append the final iteration value to keep list lengths consistent
    x_values.append(step)

    # Add the random walk plot
    plt.plot(x_values, y_values, label=f"Random Walk {walk+1}", alpha=0.6)

# Plot standard deviation boundaries
plt.plot(x_values, sigma_upper, color='green', linestyle='--', label="y = + std deviation", alpha=0.7)
plt.plot(x_values, sigma_lower, color='green', linestyle='--', label="y = - std deviation", alpha=0.7)

# Horizontal line at zero
plt.axhline(y=0, color='red', linestyle='--', label="y = 0")

# Titles and labels
plt.title(f'{num_walks} Random Walk Simulations')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.show()
