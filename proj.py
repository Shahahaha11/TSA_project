import datetime as dt
import pandas as pd
from pandas_datareader import data as web

# Define date range
end   = dt.date.today()
start = end - dt.timedelta(days=5*365)

# Map instrument names to tickers (examples)
tickers = {
    'Equity_Index': '^GSPC',    # S&P 500
    'Company_Stock': 'AAPL',    # Apple Inc.
    'FX_Pair': 'EURUSD=X',      # EUR/USD
    'Commodity': 'GC=F',        # Gold futures
    'Crypto': 'ETH-USD'         # Ethereum
}

# Fetch and combine into one DataFrame of Adj Close
prices = pd.DataFrame({
    name: web.DataReader(sym, 'yahoo', start, end)['Adj Close']
    for name, sym in tickers.items()
})

# Ensure daily frequency (forward-fill any missing)
prices = prices.asfreq('B').ffill()

print(prices.tail())
