# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf

print('ONLY ENTER TICKERS WITH AT LEAST 15 YEARS OF HISTORICAL DATA')
print('')
print('')
print('')
weights = {}
indices = [["S&P500","^GSPC"],["FTSE UK","^FTSE"],["Russell 2000 USA small caps","^RUT"],
     ["STOXX 50 Europe","^STOXX50E"],["Nikkei Japan","^N225"],["Hang Seng HK","^HSI"],
     ["SSE composite China","000001.SS"]]
s = 1
total=0
while total<100:
    s = input('type the ticker correctly ')
    k = eval(input('type the percentage composition in your portfolio for the ticker entered (for example write 50% as 50) '))
    weights[s] = (k / 100)
    indices.append([k, s])
    total+=k
   


tickers = list(np.array(indices)[:, 1])


# Request the full history and keep the adjusted-close column explicitly.
download = yf.download(tickers, period="max", auto_adjust=False)
if download is None or download.empty or "Adj Close" not in download.columns:
    raise SystemExit("Unable to download adjusted prices. Check the tickers and try again.")
dw = download["Adj Close"]

dw = dw.reindex(tickers, axis=1)

insufficient_history = dw.count()[dw.count() < 3780]
if not insufficient_history.empty:
    details = ", ".join(f"{ticker}: {count}" for ticker, count in insufficient_history.items())
    raise SystemExit(
        "At least 3780 daily prices are required for each ticker. "
        f"Missing or insufficient history ({details}). Check the tickers and try again."
    )


performance = {}

# Use the latest 3780 available prices (about 15 trading years) per ticker.
# Keep prices in chronological order; missing quotes are not forward-filled.
data = {}

for index in dw:
    closes = dw[index].sort_index().dropna().tail(3780)
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any():
        raise SystemExit(f"Invalid prices for {index}: prices must be finite and positive.")
    # Simple percentage return: (current price / previous price - 1) * 100.
    data[index] = {
        'closes': closes.tolist(),
        'returns': closes.pct_change(fill_method=None).dropna().mul(100).tolist(),
    }

# Calculating standard deviation as the average of the standard deviation every 45 sessions to weight more short-term fluctuations
for index in data:
    performance[index] = {'average return': [], 'std_dev_percentage': []}
    # Annualized arithmetic mean in percent, assuming 252 sessions per year.
    # This is an expected-return estimate, not a compounded annual return (CAGR).
    performance[index]['average return'].append(np.mean(data[index]['returns']) * 252)

for index in data:
    temp = []
    n = 1
    while n < 84:
        temp.append(((np.std(data[index]['closes'][n * 45 - 45:n * 45])) / data[index]['closes'][n * 45 - 45]) * 100)
        n += 1
     
    performance[index]['std_dev_percentage'].append(np.mean(temp))
print('RISK AND RETURN OF A BASKET OF TICKERS INCLUDING YOUR POSITIONS')
print('Average returns: annualized arithmetic mean (%), using 252 sessions per year.')
print('')
print(performance)
print('')
portfolio = {'average_return': 0, 'std_dev_percentage': 0}
for index in weights:
    for ticker in weights:
        if ticker == index:
            portfolio['average_return'] += weights[ticker] * performance[index]['average return'][0]
            portfolio['std_dev_percentage'] += weights[ticker] * performance[index]['std_dev_percentage'][0]
print('')
print('RISK AND RETURN OF YOUR PORTFOLIO')
print('')
print(portfolio)           

# Correlation matrix
df = pd.DataFrame()
i = 0
for index in data:
    df.insert(i, index, data[index]['closes'])
print('')
print('CORRELATION MATRIX OF YOUR PORTFOLIO ')
print('')
print(df.iloc[:, 0:len(weights)].corr())


# Risk detection

corr_matrix = df.iloc[:, 0:len(weights)].corr()

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

# Sort eigenvalues in descending order
sorted_eig_values = np.sort(eigenvalues)[::-1]

# Percentage of explained variance
explained_variance = sorted_eig_values / np.sum(sorted_eig_values) * 100


print("Eigenvalues of the correlation matrix:")
print(sorted_eig_values)
print("\nPercentage of variance explained by each eigenvalue:")
print(explained_variance)

# If the first eigenvalue explains more than 50% of the total variance:
if explained_variance[0] > 50:
    print("\n⚠️ WARNING: A single factor dominates the portfolio! Concentrated risk.")
else:
    print("\n✅ Good portfolio diversification.")

    




    
        
        



    


 
        






    
        




            
                




     




 
    

     


        
        









