import yfinance as yf
import pandas as pd

def get_data(symbol, start_date, end_date, interval='1d'):
    """
    Returns:
    - Download DataFrame with historical stock data.
    Options intervals: '1m','2m','5m','15m','30m','60m','90m','1d','5d','1wk','1mo','3mo'
    """
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    # Keep only the columns of interest
    df = df[['Date', 'Close', 'Volume','Low', 'High', 'Open']]

    df.to_csv(f'{symbol}_historical_data_{interval}.csv', index=False)

    return

get_data('AAPL', '1980-01-01', '2026-07-01', interval='1d')