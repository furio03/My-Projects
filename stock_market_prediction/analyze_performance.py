import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('performance.csv')
df['Date'] = pd.to_datetime(df['Date'])


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['cumulative_profit'], label='Cumulative Profit')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit')
plt.title('Cumulative Profit Over Time')
plt.grid(True)
plt.legend()
plt.show()


df = pd.read_csv('performance_optimized.csv')
df['Date'] = pd.to_datetime(df['Date'])


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['cumulative_profit'], label='Cumulative Profit')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit')
plt.title('Cumulative Profit Over Time')
plt.grid(True)
plt.legend()
plt.show()