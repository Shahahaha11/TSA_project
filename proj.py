#%%
# The following environment is selected
# ~/TSA_project/.venv/bin/python
# 
import pandas as pd
import datetime as dt
import yfinance as yf
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

df = yf.download(
    list(tickers.values()),
    start=start,
    end=end,
    progress=False
)['Close']

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
# To save train and test to pickle
file_path = "/Users/shah/TSA_project/TSA_project/train.pkl"
train.to_pickle(file_path)

file_path = "/Users/shah/TSA_project/TSA_project/test.pkl"
test.to_pickle(file_path)

"""
#%%
file_path = "/Users/shah/TSA_project/prices.pkl"
prices=pd.read_pickle(file_path)

cols = ['Company_Stock', 'Crypto', 'FX_Pair', 'Commodity', 'Equity_Index']
returns = prices[cols].pct_change().add_suffix('_ret')
prices = pd.concat([prices, returns], axis=1).dropna()
print(prices.filter(like='_ret').head())
#%%
def portfolio_return(df):
    assets = ['Company_Stock_ret', 'Crypto_ret', 'FX_Pair_ret', 'Commodity_ret', 'Equity_Index_ret']
    weight = 1 / len(assets)
    df['portfolio'] = df[assets].mul(weight).sum(axis=1)
    return df

print(prices.filter(like='_ret').head())
#%%
# Main cell: splitting into train and test
df = portfolio_return(prices)
df.columns
#%%
train = prices.iloc[:-261]  
test  = prices.iloc[-261:] 
#%%
print(train.columns)
#%% 
print(test.tail(2))
#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sharpe_frontier(df, rf=0.0, n_portfolios=5_000, use_full=False):
    """
    Draw efficient frontier and return exact GMV & tangency weights.
    """
    data = df.copy() if use_full else df
    ret_cols = [c for c in data.columns if c.endswith('_ret')]
    R = data[ret_cols].dropna()

    mu_vec   = R.mean().values                   # mean vector
    sig_cap  = R.cov().values                    # covariance matrix (Σ)
    n_assets = len(mu_vec)

    # ----- exact weights --------------------------------------------------
    inv_sig_cap = np.linalg.inv(sig_cap)
    ones_vec    = np.ones(n_assets)

    w_gmv = inv_sig_cap @ ones_vec / (ones_vec @ inv_sig_cap @ ones_vec)
    excess = mu_vec - rf
    w_tan  = inv_sig_cap @ excess / (ones_vec @ inv_sig_cap @ excess)

    # ----- Monte-Carlo frontier ------------------------------------------
    W_rand = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    port_rets = W_rand @ mu_vec
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', W_rand, sig_cap, W_rand))
    sharpe = (port_rets - rf) / port_vols

    # ----- plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(port_vols, port_rets, c=sharpe, cmap='viridis', alpha=0.3)
    ax.scatter(np.sqrt(w_gmv @ sig_cap @ w_gmv), w_gmv @ mu_vec, marker='D', color='blue', s=120, label='GMV')
    ax.scatter(np.sqrt(w_tan @ sig_cap @ w_tan),  w_tan @ mu_vec, marker='*', color='red',  s=160, label='Tangency')
    ax.set_xlabel('Volatility σ')
    ax.set_ylabel('Expected return μ')
    ax.set_title('Efficient Frontier')
    ax.legend()
    plt.show()

    # ----- return weights -------------------------------------------------
    return (pd.Series(w_gmv, index=ret_cols, name='GMV'),
            pd.Series(w_tan,  index=ret_cols, name='Tangency'))

#%%

gmv_weights, tan_weights = plot_sharpe_frontier(train)
print(gmv_weights)   # minimum-variance weights
print(tan_weights)   # max-Sharpe (tangency) weights

#%%

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
adf_test(train['Equity_Index_ret'])
#%%

train.head()

#%%

import numpy as np
from arch import arch_model
from scipy.stats import norm

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

#%%
from scipy.stats import norm  
def fit_garch(train_returns_pct):
    model = arch_model(train_returns_pct, vol='GARCH', p=1, q=1, dist='normal')
    res   = model.fit(disp='off')

    # --- annualized conditional σ (pct units) ---
    ann_sigma_pct = res.conditional_volatility * np.sqrt(252)
    res.ann_cond_std = ann_sigma_pct / 100          # store as decimals for convenience
    return res
#%%

# --- 1. fit EGARCH once to get in-sample annualized σ̂ ----------------------
def fit_egarch(train_returns_pct):
    """
    Fit EGARCH(1,1,1) and store annualized conditional σ (decimals).
    Mirrors fit_garch.
    """
    model = arch_model(train_returns_pct, vol='EGARCH',
                       p=1, o=1, q=1, dist='normal')
    res   = model.fit(disp='off')

    ann_sigma_pct = res.conditional_volatility * np.sqrt(252)
    res.ann_cond_std = ann_sigma_pct / 100      # store as decimals
    return res

#%%
# --- 2. rolling VaR + σ² for EGARCH (thin wrapper) -------------------------
def rolling_egarch_var(full_ret, alpha=0.05, window=None):
    """
    Wrapper that just calls the generic rolling routine with vol='EGARCH'.
    Produces predicted_variance and VaR Series for the test period.
    """
    return rolling_garch_var(full_ret, alpha=alpha, window=window, vol='EGARCH')


#%%
def rolling_garch_var(train_test, alpha=0.05, window=None, vol='GARCH'):
    """
    Rolling one-day VaR and σ² with either GARCH or EGARCH.
    Set vol='GARCH' (default) or vol='EGARCH'.
    """
    test_index = train_test.index[train_test.index > train_end]
    sig2_list, var_list = [], []

    for t in test_index:
        # training slice (expanding or fixed)
        if window:
            train_slice = train_test.loc[:t - pd.Timedelta(days=1)].tail(window)
        else:
            train_slice = train_test.loc[:t - pd.Timedelta(days=1)]

        # fit chosen model
        res = arch_model(train_slice * 100,
                         vol=vol, p=1, o=(1 if vol == 'EGARCH' else 0),
                         q=1, dist='normal').fit(disp='off')

        # 1-step forecast
        sig2_pct = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        sigma    = np.sqrt(sig2_pct) / 100
        VaR_t    = -norm.ppf(alpha) * sigma

        sig2_list.append(sig2_pct / 1e4)   # store as decimal variance
        var_list.append(VaR_t)

    return (pd.Series(sig2_list, index=test_index, name=f'{vol}_σ2'),
            pd.Series(var_list,  index=test_index, name=f'{vol}_VaR'))


#%%
def main(train, test, alpha=0.05, window=None):
    train = train.copy()
    # --- in-sample fits ----------------------------------------------------
    res_g = fit_garch(train['portfolio'] * 100)
    res_e = fit_egarch(train['portfolio'] * 100)
    train['ann_sigma_garch']  = res_g.ann_cond_std
    train['ann_sigma_egarch'] = res_e.ann_cond_std

    # --- rolling out-of-sample -------------------------------------------
    full_ret  = pd.concat([train['portfolio'], test['portfolio']])
    global train_end; train_end = train.index[-1]

    pred_var_g, VaR_g = rolling_garch_var(full_ret, alpha=alpha,
                                          window=window, vol='GARCH')
    pred_var_e, VaR_e = rolling_garch_var(full_ret, alpha=alpha,
                                          window=window, vol='EGARCH')

    # --- quick comparison plot -------------------------------------------
    ax = VaR_g.plot(label='GARCH VaR', figsize=(10,4))
    VaR_e.plot(ax=ax, label='EGARCH VaR')
    test['portfolio'].plot(ax=ax, alpha=0.4, label='Returns')
    ax.set_title('Rolling 1-day VaR: GARCH vs. EGARCH'); ax.legend()
    plt.show()

#%%
if __name__ == "__main__":
    # assume you already sliced your data
    # train  : DataFrame with column 'portfolio', DatetimeIndex
    # test   : same structure, later dates
    main(train, test, alpha=0.05, window=None)   # window=None → expanding
    # e.g. main(train, test, alpha=0.05, window=252)  # 1-year rolling window

#%%
#%% [markdown]
# **Workflow:**
# <br>We estimate GARCH(1,1) and EGARCH(1,1,1) on the training set and record their annualised conditional σ.  
# <br>Each test date is then forecast by refitting the chosen model on data up to t-1 and producing a one-day 95 % VaR.  
# <br>The objects `ann_sigma_garch`, `ann_sigma_egarch`, `GARCH_VaR`, and `EGARCH_VaR` hold the in-sample risk levels and the rolling out-of-sample VaR forecasts used for comparison.


#%%

def compute_single_var_from_garch(train, test_index, alpha=0.05):
    y_pct = train['portfolio'] * 100
    model = arch_model(y_pct, vol='GARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')

    h1_var_pct = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    # Use this for VaR of each forecasted day
    # h1_var_pct = res.forecast(horizon=len(test_index), reindex=False).variance.values[-1]
    sigma_pct = np.sqrt(h1_var_pct)
    z = norm.ppf(alpha)
    VaR = -(z * sigma_pct) / 100

    return VaR

#%%

print(compute_single_var_from_garch(train, test.index))
#%% [markdown] 
# The function compute_single_var_from_garch as it says, outputs a single value (at-risk) for the next day in train
# <br>**VAR Interpretation:** 
# At the 95 % confidence level, tomorrow’s one-day VaR is 1.73 % of portfolio value—i.e., there’s only a 5 % chance the loss will exceed 1.73 % in the next trading day.
#%%