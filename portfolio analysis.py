# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf

print('ONLY ENTER TICKERS WITH AT LEAST 15 YEARS OF HISTORY')
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


dw = yf.download(tickers)["Adj Close"]

# Reordering the columns that I don't understand why they get shuffled
dw = dw.reindex(tickers, axis=1)


performance = {}

# There are about 252 trading days in a year, I consider the last 15 years to avoid missing values in comparisons, then I cleaned the series
data = {}

for index in dw:
    data[index] = {'returns': [], 'closes': []}
    temp = []

    for value in dw[index]:
        if str(value) != 'nan':
            temp.append(value)
    temp = reversed(temp)
    temp = list(temp)
    i = 0
    while len(data[index]['closes']) < 3780:
        data[index]['closes'].append(temp[i])
        i += 1


# Calculating daily returns

for index in data:
    i = 1
    k = 0
    while i < len(data[index]['closes']):
        data[index]['returns'].append(((data[index]['closes'][k] - data[index]['closes'][i]) / data[index]['closes'][k]) * 100)
        k += 1
        i += 1

# Calculating standard deviation as the average of the standard deviation every 45 sessions to weight more short-term fluctuations
for index in data:
    performance[index] = {'average return': [], 'std_dev_percentage': []}
    sum_returns = 0
    for value in data[index]['returns']:
        sum_returns += value
    performance[index]['average return'].append(sum_returns / 15)

for index in data:
    temp = []
    n = 1
    while n < 84:
        temp.append(((np.std(data[index]['closes'][n * 45 - 45:n * 45])) / data[index]['closes'][n * 45 - 45]) * 100)
        n += 1
     
    performance[index]['std_dev_percentage'].append(np.mean(temp))
print('RISK AND RETURN OF A BASKET OF TICKERS INCLUDING YOUR POSITIONS')
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

    




    
        
        



    


 
        






    
        




            
                




     




 
    

     


        
        








