import pandas as pd
import datetime as dt
import yfinance as yf
# Import necessary libraries
import pandas as pd # for data processing
import numpy as np # here mostly for series generation

import matplotlib.pyplot as plt # for vizualization
import matplotlib.dates as mdates # for data formatting when visualizing
import matplotlib.ticker as ticker # for more advanced axis formatting

import seaborn as sns

import statsmodels.api as sm # for linear regression model, OLS estimation
import statsmodels.stats.diagnostic as smd # for Breusch-Godfrey test

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # ACF, PACF plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from IPython.display import Markdown
#%%
"""    
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
file_path = "/Users/shah/TSA_project/TSA_project/prices.pkl"
prices.to_pickle(file_path)
"""
"""
train = prices.iloc[:-261]  
test  = prices.iloc[-261:] 
"""

"""
file_path = "/Users/shah/TSA_project/TSA_project/train.pkl"
train.to_pickle(file_path)

file_path = "/Users/shah/TSA_project/TSA_project/test.pkl"
test.to_pickle(file_path)

"""

file_path = "/Users/shah/TSA_project/TSA_project/prices.pkl"
prices=pd.read_pickle(file_path)

cols = ['Company_Stock', 'Crypto', 'FX_Pair', 'Commodity', 'Equity_Index']
returns = prices[cols].pct_change().add_suffix('_ret')
prices = pd.concat([prices, returns], axis=1).dropna()
print(prices.filter(like='_ret').head())


def portfolio_return(df):
    assets = ['Company_Stock_ret', 'Crypto_ret', 'FX_Pair_ret', 'Commodity_ret', 'Equity_Index_ret']
    weight = 1 / len(assets)
    df['portfolio'] = df[assets].mul(weight).sum(axis=1)
    return df

print(prices.filter(like='_ret').head())

prices = portfolio_return(prices)
prices.columns

train = prices.iloc[:-261]  
test  = prices.iloc[-261:] 

"""
file_path = "/Users/shah/TSA_project/TSA_project/test.pkl"
test= pd.read_pickle(file_path)
"""
#%%

def plot_sharpe_frontier(df, rf=0.0, n_portfolios=5000):
    """Plots the efficient frontier & tangent (CML) using actual means & covariances."""
    mu  = df.mean()        # actual mean returns
    cov = df.cov()         # actual covariance matrix
    n   = len(mu)
    # random weight draws that sum to 1
    W   = np.random.dirichlet(np.ones(n), n_portfolios)
    # portfolio returns and volatilities
    rets = W.dot(mu)
    vols = np.sqrt(np.einsum('ij,ji->i', W.dot(cov), W.T))
    # pick the tangency portfolio
    sharpe = (rets - rf) / vols
    t    = np.argmax(sharpe)
    sr   = sharpe[t]

#%%
    # plot everything
    plt.scatter(vols, rets, s=10, alpha=0.6)
    x = np.linspace(0, vols.max(), 100)
    plt.plot(x, rf + sr * x, 'r--', linewidth=2)
    plt.scatter(vols[t], rets[t], c='red', zorder=5)

    plt.xlabel('Volatility (Std. Dev.)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier & Capital Market Line')
    plt.show()

train_drop_nan = train.dropna()
train_drop_nan[['Company_Stock_ret', 'Crypto_ret', 'FX_Pair_ret', 'Commodity_ret', 'Equity_Index_ret', 'portfolio']].plot(figsize=(10, 6), title="PLOT")
plt.show()
#%%
def adf_test0(series, max_lag=3):
    result = adfuller(series.dropna(), maxlag=max_lag, autolag=None)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")

#%%

train.tail(3)
train.columns

################### initial eda ################
train[['Company_Stock', 'Crypto', 'FX_Pair', 'Commodity', 'Equity_Index', 'portfolio']].plot(figsize=(10, 6), title="PLOT")
plt.show()
#%%
############# ADF TEST ##############
train.columns

adf_test0(train['Company_Stock_ret'])
adf_test0(train['Crypto_ret'])
adf_test0(train['FX_Pair_ret'])
adf_test0(train['Commodity_ret'])
adf_test0(train['Equity_Index_ret'])

train.head()
#%%
######################### ADF TEST WITH  AUGMENTATIONS ###########################################
def adf_test(series, max_aug=10, version='c'):
    
    results = []

    y = series.diff()
    X = pd.DataFrame({'y_lag': series.shift()})

    if version == 'c' or version == 't': # constant to be added optionally 
        X = sm.add_constant(X)
    if version == 't': # (deterministic) trend component to be added optionally
        X['trend'] = range(len(X))

    for i in range(0, max_aug): # iterating through different numbers of augmentations
        
        for aug in range(1, i+1): # adding augmentations max_aug is reached
            X['aug_'+str(aug)] = y.shift(aug)

        model = sm.OLS(series.diff(), X, missing='drop').fit() # fitting a linear regression with OLS

        ts = model.tvalues['y_lag'] # test statistic
        nobs = model.nobs # number of observations

        if version == 'n': # critical values for basic version of ADF
            if nobs > 100:
                cv01 = -2.567; cv05 = -1.941; cv10 = -1.616 # critical values for more than 500 observations
            else:
                cv01 = np.nan; cv05 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually
        if version == 'c': # critical values for version with constant
            if nobs > 100:
                cv01 = -3.434; cv05 = -2.863; cv10 = -2.568 # critical values for more than 500 observations
            else:
                cv01 = np.nan; cv05 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually
        if version == 't': # critical values for version with constant and (deterministic) trend component
            if nobs > 100:
                cv01 = -3.963; cv05 = -3.413; cv10 = -3.128 # critical values for more than 500 observations
            else:
                cv01 = np.nan; cv05 = np.nan; cv10 = np.nan # if number of observations is lower than 500, we should check the critical values manually

        bg_test01 = smd.acorr_breusch_godfrey(model, nlags=1);  bg_pvalue01 = round(bg_test01[1],4); bg_test01 = round(bg_test01[0],4); 
        bg_test05 = smd.acorr_breusch_godfrey(model, nlags=5);  bg_pvalue05 = round(bg_test05[1],4); bg_test05 = round(bg_test05[0],4); 
        bg_test10 = smd.acorr_breusch_godfrey(model, nlags=10); bg_pvalue10 = round(bg_test10[1],4); bg_test10 = round(bg_test10[0],4);
        bg_test15 = smd.acorr_breusch_godfrey(model, nlags=15); bg_pvalue15 = round(bg_test15[1],4); bg_test15 = round(bg_test15[0],4);

        results.append([i, ts, cv01, cv05, cv10, 
                        bg_test01, bg_pvalue01, bg_test05, bg_pvalue05, bg_test10, bg_pvalue10, bg_test15, bg_pvalue15])

    results_df = pd.DataFrame(results)
    results_df.columns = ['number of augmentations', 
                          'ADF test statistic', 'ADF critival value (1%)', 'ADF critival value (5%)', 'ADF critival value (10%)', 
                          'BG test (1 lag) (statistic)', 'BG test (1 lag) (p-value)', 
                          'BG test (5 lags) (statistic)', 'BG test (5 lags) (p-value)', 
                          'BG test (10 lags) (statistic)', 'BG test (10 lags) (p-value)', 
                          'BG test (15 lags) (statistic)', 'BG test (15 lags) (p-value)']
    
    return results_df
#%% 
adf_test(train['Company_Stock_ret'])
#%%
adf_test(train['Crypto_ret'])
#%%

adf_test(train['FX_Pair_ret'])
#%%
adf_test(train['Commodity_ret'])
#%%

train.head()

#%%
adf_test(train['Equity_Index_ret'])
#%%

import numpy as np
from arch import arch_model
from scipy.stats import norm

#%%
def fit_model_rescaled(train, test_index, model='GARCH', alpha=0.05):
    # 1) rescale to pct
    y_pct = train['portfolio'] * 100

    # 2) fit
    am = arch_model(y_pct, vol=model, p=1, q=1, dist='normal')
    res = am.fit(disp='off')

    # 3) in‐sample annualized vol (in pct units), then back to decimals
    vol_pct_ann = res.conditional_volatility
    vol_ann = vol_pct_ann / 100

    # 4) 1‐step var forecasts (in pct units)
    h1_var_pct = res.forecast(start=test_index[0], horizon=1).variance['h.1']
    sigma_pct = np.sqrt(h1_var_pct)
    
    # 5) VaR in decimal returns
    z = norm.ppf(alpha)
    VaR = -(z * sigma_pct) / 100
    mdl= train.copy()
    mdl['portfolio_100']= mdl['portfolio']*100
    # full series of annualized realized vol (one-step ahead proxy)
    mdl['realized_vol'] = np.sqrt(mdl['portfolio'].pow(2).shift(1).rolling(3).sum())
    
    return vol_ann, VaR, mdl, test_index
#%%
#  2MARKDOWN
#  
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def fit_garch(train_returns_pct):
    """Fit GARCH(1,1) and return result object."""
    model = arch_model(train_returns_pct, vol='GARCH', p=1, q=1, dist='normal')
    return model.fit(disp='off')

def one_step_forecast(res, test_index):
    """Produce one-step-ahead sig² forecasts over test_index."""
    h = len(test_index)
    fc = res.forecast(horizon=h, reindex=False)
    # last row contains forecasts for t = T_train+1 … T_train+h
    var_vals = fc.variance.values[-1, :]
    return pd.Series(var_vals, index=test_index, name='predicted_variance')

def filtered_variance(res, train_pct, test_pct):
    """Re-run GARCH recursion to get sig² filtered over train+test."""
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta  = res.params['beta[1]']
    mean  = res.params.get('mu', 0)

    train_sigma2 = (res.conditional_volatility**2).values
    all_r = np.concatenate([train_pct.values, test_pct.values])
    T     = len(all_r)
    sig2    = np.empty(T)
    sig2[:len(train_sigma2)] = train_sigma2

    for t in range(len(train_sigma2), T):
        eps2    = (all_r[t-1] - mean)**2
        sig2[t]   = omega + alpha * eps2 + beta * sig2[t-1]

    return pd.Series(sig2[len(train_sigma2):], index=test_pct.index, name='actual_variance')

def main(train, test):
    train_pct = train['portfolio'] * 100
    test_pct  = test['portfolio']  * 100

    res        = fit_garch(train_pct)
    pred_var   = one_step_forecast(res, test.index)
    actual_var = filtered_variance(res, train_pct, test_pct)

    compare = pd.DataFrame({'predicted_variance': pred_var,
                            'actual_variance':    actual_var})
    compare.plot(title='GARCH(1,1) sig²: Forecast vs. Filtered')
    plt.show()

if __name__ == '__main__':
    main(train, test)

#%%
import matplotlib.pyplot as plt
compare[['predicted', 'actual']].plot()
plt.show()

#%%
mdl_test(train,test.index).plot()


mdl_test(train, test.index).dropna()[['pred','actual']].plot()


#%%
# Usage:
g_vol, g_VaR, mdl, = fit_model_rescaled(train, test.index, model='GARCH')
e_vol, e_VaR, mdl = fit_model_rescaled(train, test.index, model='EGARCH')
#%%

pd.DataFrame({'GARCH': g_vol, 'actual': mdl['realized_vol'] }).plot()
pd.DataFrame({'EGARCH': e_vol, 'actual': mdl['realized_vol']}).plot()
#%%
pd.DataFrame({'EGARCH': e_vol,'GARCH': g_VaR}).plot()
#%%
g_var = pd.Series(g_VaR).rename('g_var')

# %%

model_arch1 = arch_model(mdl[''].dropna(), vol='ARCH', p=1)
results_arch1 = model_arch1.fit(disp='off')
print(results_arch1.summary())
# %%
