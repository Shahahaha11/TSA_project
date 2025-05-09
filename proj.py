import datetime as dt
import pandas as pd
import yfinance as yf

# Fixed date range
start = dt.date(2020, 5, 1)
end   = dt.date(2025, 5, 1)

# Map instrument names to tickers
tickers = {
    'Equity_Index': '^GSPC',    # S&P 500
    'Company_Stock': 'AAPL',    # Apple Inc.
    'FX_Pair': 'EURUSD=X',      # EUR/USD
    'Commodity': 'GC=F',        # Gold futures
    'Crypto': 'ETH-USD'         # Ethereum
}

# Download adjusted close for all symbols in one call
df = yf.download(
    list(tickers.values()),
    start=start,
    end=end,
    progress=False
)['Close']

# Rename columns from ticker symbols to your friendly names
prices = df.rename(columns={sym: name for name, sym in tickers.items()})

# Ensure business‐day frequency and forward‐fill
prices = prices.asfreq('B').ffill()

print(prices.head())
"""    
file_path = "/Users/shah/TSA_project/TSA_project/prices.pkl"
prices.to_pickle(file_path)
"""
train = prices.iloc[:-261]  
test  = prices.iloc[-261:] 

train.tail(3)

file_path = "/Users/shah/TSA_project/TSA_project/train.pkl"
train.to_pickle(file_path)

file_path = "/Users/shah/TSA_project/TSA_project/test.pkl"
test.to_pickle(file_path)



