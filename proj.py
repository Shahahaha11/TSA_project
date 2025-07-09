#%%
# The following environment is selected
# ~/TSA_project/.venv/bin/python

#%% 
from IPython.display import Markdown, display
display(Markdown("""
# 1. Data Collection and Portfolio Construction  
**Objective:** To create a diversified, equally weighted portfolio comprising five financial instruments across distinct asset classes, and prepare the dataset for volatility modeling.  
Data was collected for the period from 2020-05-01 to 2025-05-01.  

**Instruments chosen:**  
- S&P 500 Index (Equity Index)  
- Apple Inc. (Stock)  
- EUR/USD (Currency Pair)  
- Gold Futures (Commodity)  
- Ethereum (Cryptocurrency)  

All instruments were sourced using Yahoo Finance (`yfinance` package).  
Daily adjusted closing prices were aligned across all instruments.  
The portfolio was constructed with equal weights (20%) and daily log returns were calculated.
"""))

#%% 

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf, pacf
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from IPython.display import Markdown
from arch import arch_model
from scipy import stats
from curl_cffi import requests
from scipy.stats import norm
from scipy.optimize import minimize
#from arch.unitroot import arch_unitroot_test
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.univariate.base import DataScaleWarning
import warnings

# Suppress only this warning
warnings.filterwarnings("ignore", category=DataScaleWarning)


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
file_path = "prices.pkl"
prices=pd.read_pickle(file_path)
#%%
def adf_test0(series, max_lag=3):
    result = adfuller(series.dropna(), maxlag=max_lag, autolag=None)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")

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
cols = ['Company_Stock', 'Crypto', 'FX_Pair', 'Commodity', 'Equity_Index']
"""returns = prices[cols].pct_change().add_suffix('_ret')"""
returns = np.log(prices[cols] / prices[cols].shift(1)).add_suffix('_ret')
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
# Plot time series of returns
plt.figure(figsize=(12, 4))
plt.plot(train['portfolio'])
plt.title('Time Series of Portfolio Log Returns')
plt.grid(True)
plt.show()
#%%
# Plot squared returns
plt.figure(figsize=(12, 4))
plt.plot(train['portfolio'] ** 2)
plt.title('Time Series of Squared Returns')
plt.grid(True)
plt.show()

#%%
cols = ['Company_Stock', 'Crypto', 'FX_Pair', 'Commodity', 'Equity_Index']

train[cols].div(train[cols].iloc[0]).mul(100).plot(figsize=(10, 6), title="Rebased (t₀ = 100)")
plt.show()

#%%
############# ADF TEST ##############
train.columns

adf_test0(train['Company_Stock_ret'])
adf_test0(train['Crypto_ret'])
adf_test0(train['FX_Pair_ret'])
adf_test0(train['Commodity_ret'])
adf_test0(train['Equity_Index_ret'])

#%% 
adf_test(train['Company_Stock'])
#%%
adf_test(train['Crypto'])
#%%
adf_test(train['FX_Pair'])
#%%
adf_test(train['Commodity'])
#%%
adf_test(train['Equity_Index'])
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
adf_test(train['portfolio'])
#%%
train.head()
# Histogram
plt.figure(figsize=(8, 4))
sns.histplot(train['portfolio'], bins=50, kde=True)
x = np.linspace(train['portfolio'].min(), train['portfolio'].max(), 100)
plt.plot(x, norm.pdf(x, train['portfolio'].mean(), train['portfolio'].std()), 'r-', linewidth=2)
plt.title('Histogram of Portfolio Log Returns')
plt.grid(True)
plt.show()

#%% 
# Autocorrelation function for squared log returns.
version = pacf
acf_values = version((train['portfolio']**2).dropna(), nlags=36)
plt.figure(figsize=(12, 6))
plt.stem(range(len(acf_values)), acf_values)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train['portfolio'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train['portfolio'])), linestyle='--', color='gray')
plt.title(f'{version.__name__} of Log Returns of ')
plt.show()
#%% 
# Autocorrelation function for squared log returns.
version = acf
acf_values = version((train['portfolio']**2).dropna(), nlags=36)
plt.figure(figsize=(12, 6))
plt.stem(range(len(acf_values)), acf_values)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train['portfolio'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train['portfolio'])), linestyle='--', color='gray')
plt.title(f'{version.__name__} of Log Returns of ')
plt.show()
#%%
display(Markdown("""
### PACF Interpretation and Lag Selection Strategy

Looking at the PACF, we can say with 95% confidence that the log price differences from 9, 24, and 26 days ago  
are significantly correlated with today's price change, independent of other lags.  
Since the ACF and PACF patterns look nearly identical, this suggests that both **ARCH** (shock-driven) and **GARCH** (volatility persistence) effects are present and intertwined.  
In other words, both squared shocks (ε²) and past variances (σ²) influence current volatility.  
Therefore, for the **GARCH lag (`q`)**, we keep it minimal (lag 1) to save computation.  
For the **ARCH lag (`p`)**, we test the specific lags: 1, 2, 3, 5, and 25.
"""))
#%%
print("Basic statistics:")
print(train['portfolio'].describe())
print("\nSkewness:", stats.skew(train['portfolio'].dropna()))
print("Kurtosis:", stats.kurtosis(train['portfolio'].dropna()))
#%% [markdown]
display(Markdown("""The returns show a slight left skew, meaning small losses occur a bit more frequently than gains. Additionally, the kurtosis is higher than 3, indicating fat tails—so extreme returns, both positive and negative, happen more often than would be expected in a normal distribution. This suggests the data is not perfectly symmetric and has a higher chance of large shocks."""))
#%%
# Histogram of log returns
plt.figure(figsize=(12, 6))
sns.histplot(train['portfolio'].dropna(), stat="density", bins = 60)
# add normal distribution curve
mu, std = train['portfolio'].mean(), train['portfolio'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(x=0, linestyle='--', color='gray')
plt.axvline(x=train['portfolio'].mean(), linestyle='--', color='red')
plt.axvline(x=train['portfolio'].quantile(0.025), linestyle='--', color='orange')
plt.axvline(x=train['portfolio'].quantile(0.975), linestyle='--', color='orange')
plt.title('Distribution of Log Returns of portfolio')
plt.show()
#%%
# Jarque-Bera test
jb_test = stats.jarque_bera(train['portfolio'].dropna())
print(f"Jarque-Bera test statistic: {jb_test[0]:.2f}")
print(f"p-value: {jb_test[1]:.2e}")

#%%
#QQ- plot

## Q-Q plot of log returns
plt.figure(figsize=(12, 6))
stats.probplot(train['portfolio'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Log Returns of Portfolio')
plt.grid(True)
plt.show()


#%% [markdown]
display(Markdown("""
### **Normality and Distributional Characteristics**

- We observe **high kurtosis** and **fat tails**, which are inconsistent with a normal distribution.  
- The **Jarque-Bera test** strongly rejects the null hypothesis, confirming the sample does not follow normality.  

---

### **Q-Q Plot Observations**

- The **left tail is much thicker**, suggesting negative skewness and potential asymmetry.  
- This skew and fat tail may signal **volatility clustering**.  
- The **probability mass is concentrated around small negative returns**, which may hint at a snowball effect during downturns (bear phase).  
- However, this is **not conclusive from the Q-Q plot alone** — we confirm with the **ARCH LM test** next.
"""))

#%%
# ARCH test (Engle's LM test)
# H0: No ARCH effects
# H1: ARCH effects are present
from statsmodels.stats.diagnostic import het_arch
# Drop NA values from returns for the test
returns_for_arch_test = train['portfolio'].dropna()
# H0: NO ARCH EFFECT
# low p -> reject H0. strong arch effect
arch_test_results = het_arch(returns_for_arch_test, nlags=5)

print(f"LM Statistic: {arch_test_results[0]:.3f}")
print(f"p-value: {arch_test_results[1]:.3f}")
print(f"F-statistic: {arch_test_results[2]:.3f}")
print(f"F p-value: {arch_test_results[3]:.3f}")

#%%

# [markdown]
display(Markdown("""
### ARCH Effect and Portfolio Weighting

- A **strong ARCH effect** is observed in the equally weighted portfolio.  
- For further exploration, we can compare this with a **minimum-variance weighted portfolio**.  
- This helps test whether the ARCH effect can be reduced by positioning along the **efficient frontier**.
"""))

#%%
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
def fit_garch(train_returns_pct, p =1, q=1, dist='ged'):
    model = arch_model(train_returns_pct, vol='GARCH', p=p, q=q, dist=dist)
    res   = model.fit(disp='off')
    alpha = res.params["alpha[1]"]
    beta  = res.params["beta[1]"]
    # --- annualized conditional σ (pct units) ---
    ann_sigma_pct = res.conditional_volatility * np.sqrt(252)
    res.ann_cond_std = ann_sigma_pct / 100          # store as decimals for convenience
    std_resid = res.resid / res.conditional_volatility
    return res, alpha, beta, std_resid

#%%
# --- 1. fit EGARCH once to get in-sample annualized σ̂ ----------------------
def fit_egarch(train_returns_pct, p=1, o=1, q=1, dist = 'ged', start_params=None):

    model = arch_model(train_returns_pct, vol='EGARCH',
                       p=p, o=o, q=q, dist=dist)
    res   = model.fit(disp='off', starting_values=start_params)
    alpha = res.params["alpha[1]"]
    beta  = res.params["beta[1]"]
    ann_sigma_pct = res.conditional_volatility * np.sqrt(252)
    res.ann_cond_std = ann_sigma_pct / 100      # store as decimals
    std_resid = res.resid / res.conditional_volatility
    
    return res, alpha, beta, std_resid

#%%
def rolling_garch_var(train_test, split_idx, alpha=0.05, window=None,
                      p=1, q=1, o=1, vol='GARCH', dist= 'ged', start_params=None):
    
    start_params = start_params
    test_index  = train_test.loc[split_idx:].index[1:]
    sig2_list, var_list = [], []

    for t in test_index:
        train_slice = train_test.loc[:t - pd.Timedelta(days=1)]
        if window:
            train_slice = train_slice.tail(window)  

        if vol == 'GARCH':
            res, arch_alpha, beta, _ = fit_garch(train_slice*100, p=p, q=q, dist = dist)
        else:
            res, arch_alpha, beta, _ = fit_egarch(train_slice*100, p=p, o=o, q=q, dist=dist,start_params=start_params)
        if getattr(res, "converged", getattr(res, "success", True)):
            start_params = res.params.values
        
        sigma  = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / 100
        VaR_t = -norm.ppf(alpha) * sigma
        sig2_list.append(sigma**2)             # store decimal variance directly
        var_list.append(VaR_t)

    return (pd.Series(sig2_list, index=test_index, name=f'{vol}_σ2'),
            pd.Series(var_list,  index=test_index, name=f'{vol}_VaR'))
    
#%%
def rolling_garch_var_1(train_test, split_idx, alpha=0.05, window=None,
                      p=1, q=1, o=1, vol='GARCH',dist = 'ged', start_params=None):
    
    start_params = start_params
    test_index  = train_test.loc[split_idx:].index[1:]
    sig2_list, var_list = [], []

    for t in test_index:
        train_slice = train_test.loc[:t - pd.Timedelta(days=1)]
        if window:
            train_slice = train_slice.tail(window)  

        if vol == 'GARCH':
            res, arch_alpha, beta, _ = fit_garch(train_slice*100, p=p, q=q, dist = dist)
        else:
            res, arch_alpha, beta, _ = fit_egarch(train_slice*100, p=p, o=o, q=q, dist= dist, start_params=start_params)
        if getattr(res, "converged", getattr(res, "success", True)):
            start_params = res.params.values
        
        sigma  = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / 100
        VaR_t = -norm.ppf(alpha) * sigma
        sig2_list.append(sigma**2)             # store decimal variance directly
        var_list.append(VaR_t)

    return (pd.Series(sig2_list, index=test_index, name=f'{vol}_σ2'),
            pd.Series(var_list,  index=test_index, name=f'{vol}_VaR'))
#%%
def violation_ratio(returns, var):
    breaches = returns < -var
    return breaches.sum() / len(breaches)
#%%
# MAIN - BASE GARCH VARIANTS
def main(train, test, alpha=0.05, window= None, p=1, o=1, q=1, dist='skewt', prev_params= None):
    train = train.copy()

    # --- in-sample fits ----------------------------------------------------
    res_g ,alpha_gar, beta_gar, std_resid_gar= fit_garch(train['portfolio'] * 100, p=p, q=q, dist=dist)
    res_e , alpha_egar, beta_egar, std_resid_egar= fit_egarch(train['portfolio']* 100, p=p, o=o, q=q, dist=dist, start_params=prev_params)
    train['ann_sigma_garch']  = res_g.ann_cond_std
    train['ann_sigma_egarch'] = res_e.ann_cond_std

    print('alpha+beta for GARCH:', res_g.params.filter(like='alpha').sum() + res_g.params.filter(like='beta').sum())

    print('beta sum for EGARCH:', res_e.params.filter(like='beta').sum())
    
    
    # --- rolling out-of-sample -------------------------------------------
    full_ret  = pd.concat([train['portfolio'], test['portfolio']])
    split_idx = train.index[-1]       # the cut-off between in-sample and out-of-sample

    _, VaR_g = rolling_garch_var(full_ret, split_idx, alpha=alpha,
                                          window=window, vol='GARCH', p=p, q=q, dist=dist)
    _, VaR_e = rolling_garch_var(full_ret, split_idx, alpha=alpha,
                                          window=window, vol='EGARCH', p=p, o=o, q=q, dist=dist, start_params=prev_params)
     # --- out  of sample breach rates ------------------------------------------------------
    vr_g = violation_ratio(test['portfolio'], VaR_g.loc[test.index])
    vr_e = violation_ratio(test['portfolio'], VaR_e.loc[test.index])
    print(f"GARCH breach rate OoS: {vr_g*100:.2f}%")
    print(f"EGARCH breach rate OoS: {vr_e*100:.2f}%")
    
    print(VaR_g.tail(10))
    VaR_g_is = -norm.ppf(alpha) * res_g.conditional_volatility /100
    VaR_e_is = -norm.ppf(alpha) * res_e.conditional_volatility /100
     # --- in sample breach rates ------------------------------------------------------
    vr_g = violation_ratio(train['portfolio'], VaR_g_is)
    vr_e = violation_ratio(train['portfolio'], VaR_e_is)
    print(f"GARCH breach rate IS: {vr_g*100:.2f}%")
    print(f"EGARCH breach rate IS: {vr_e*100:.2f}%")
    
    # --- plot - In sample -------------------------------------------------------------
    ax = VaR_g_is.plot(label='GARCH VaR', figsize=(10,4))
    VaR_e_is.plot(ax=ax, label='EGARCH VaR')
    train['portfolio'].plot(ax=ax, alpha=0.4, label='Returns')
    ax.set_title('IS VaR: GARCH vs. EGARCH'); ax.legend()
    plt.show()
    # --- plot - Out of Sample -------------------------------------------
    
    ax = VaR_g.plot(label='GARCH VaR', figsize=(10,4))
    VaR_e.plot(ax=ax, label='EGARCH VaR')
    test['portfolio'].plot(ax=ax, alpha=0.4, label='Returns')
    ax.set_title('OOS VaR : GARCH vs. EGARCH'); ax.legend()
    plt.show()
    return res_g, res_e, VaR_g, VaR_e, vr_g, vr_e, std_resid_gar, std_resid_egar
#%%
# MAIN_1 - GARCH VARIANTS WITH HIGHER ORDER TERMS.(copy of main)
def main_1(train, test, alpha=0.05, window= None, p=5, o=1, q=5, dist='skewt', prev_params=None):

    train = train.copy()
    # --- in-sample fits ----------------------------------------------------
    res_g ,alpha_gar, beta_gar, std_resid_gar= fit_garch(train['portfolio'] * 100, p=p, q=q, dist=dist)
    res_e , alpha_egar, beta_egar, std_resid_egar= fit_egarch(train['portfolio']* 100, p=p, o=o, q=q,dist=dist, start_params=prev_params)
    train['ann_sigma_garch']  = res_g.ann_cond_std
    train['ann_sigma_egarch'] = res_e.ann_cond_std


    print('alpha+beta for GARCH:', res_g.params.filter(like='alpha').sum() + res_g.params.filter(like='beta').sum())

    print('beta sum for EGARCH:', res_e.params.filter(like='beta').sum())    
    # --- rolling out-of-sample -------------------------------------------
    full_ret  = pd.concat([train['portfolio'], test['portfolio']])
    split_idx = train.index[-1]       # the cut-off between in-sample and out-of-sample

    _, VaR_g = rolling_garch_var_1(full_ret, split_idx, alpha=alpha,
                                          window=window, vol='GARCH', p=p, q=q, dist=dist)
    _, VaR_e = rolling_garch_var_1(full_ret, split_idx, alpha=alpha,
                                          window=window, vol='EGARCH', p=p, o=o, q=q, dist=dist, start_params=prev_params)
     # --- out  of sample breach rates ------------------------------------------------------
    vr_g = violation_ratio(test['portfolio'], VaR_g.loc[test.index])
    vr_e = violation_ratio(test['portfolio'], VaR_e.loc[test.index])
    print(f"GARCH breach rate OoS: {vr_g*100:.2f}%")
    print(f"EGARCH breach rate OoS: {vr_e*100:.2f}%")
    
    VaR_g_is = -norm.ppf(alpha) * res_g.conditional_volatility /100
    VaR_e_is = -norm.ppf(alpha) * res_e.conditional_volatility /100
     # --- in sample breach rates ------------------------------------------------------
    vr_g = violation_ratio(train['portfolio'], VaR_g_is)
    vr_e = violation_ratio(train['portfolio'], VaR_e_is)
    print(f"GARCH breach rate IS: {vr_g*100:.2f}%")
    print(f"EGARCH breach rate IS: {vr_e*100:.2f}%")
    
    # --- plot - In sample -------------------------------------------------------------
    ax = VaR_g_is.plot(label='GARCH VaR', figsize=(10,4))
    VaR_e_is.plot(ax=ax, label='EGARCH VaR')
    train['portfolio'].plot(ax=ax, alpha=0.4, label='Returns')
    ax.set_title('IS VaR: GARCH vs. EGARCH'); ax.legend()
    plt.show()
    # --- plot - Out of Sample -------------------------------------------
    
    ax = VaR_g.plot(label='GARCH VaR', figsize=(10,4))
    VaR_e.plot(ax=ax, label='EGARCH VaR')
    test['portfolio'].plot(ax=ax, alpha=0.4, label='Returns')
    ax.set_title('OOS VaR : GARCH vs. EGARCH'); ax.legend()
    plt.show()
    return res_g, res_e, VaR_g, VaR_e, vr_g, vr_e, std_resid_gar, std_resid_egar

#%%
prev_params = np.array([0.0, 0.0, 0.05, 0.0, 0.90, 8.0])  # [μ, ω, α1, γ1, β1, ν]
if __name__ == "__main__":

    res_g_0, res_e_0, VaR_g_0, VaR_e_0, vr_g_0, vr_e_0, std_resid_gar_0, std_resid_egar_0  = main(train, test, alpha=0.05, window= None, p=1, o=1, q=1, dist='ged', prev_params=prev_params)
#%%
# Define initial parameters for model_1.
#%%
# Define initial parameters for model_1.
prev_params_121 = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.9, 6.0])  # [μ, ω, α1, α2, α3, α4, α4, γ1, β1, β2, β3, β4, β5, ν]
prev_params_515 = np.array([0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.18, 0.18, 0.18, 0.18, 0.18, 6.0])  # [μ, ω, α1, α2, α3, α4, α4, γ1, β1, β2, β3, β4, β5, ν]
prev_params_513 = np.array([0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.03, 0.03, 0.03, 8.0])  # [μ, ω, α1, α2, α3, α4, α5, γ1, β1, β2, β3, ν]
prev_params_513_skew = np.array([0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.03, 0.03, 0.03, 10.0, 0.0])  # [μ, ω, α1, α2, α3, α4, α5, γ1, β1, β2, β3, ν]
prev_params_315 = np.array([0.0, 0.0, 0.017,0.017,0.017, 0.0, 0.18, 0.18, 0.18, 0.18, 0.18, 8.0])  # [μ, ω, α1, α2, α3, γ1, β1, β2, β3, β4, β5, ν]
prev_params_523 = np.array([0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.03, 0.03, 0.03, 8.0])  # [μ, ω, α1, α2, α3, α4, α5, γ1, β1, β2, β3, ν]
if __name__ == "__main__":
    res_g_1, res_e_1, VaR_g_1, VaR_e_1, vr_g_1, vr_e_1, std_resid_gar_1, std_resid_egar_1  = main_1(train, test, alpha=0.05, window= None, p=5, o=1, q=3, dist = 'ged', prev_params=prev_params_513)
#%%
from statsmodels.stats.diagnostic import acorr_ljungbox
for lag in (10, 15, 20): print(acorr_ljungbox(std_resid_egar_0, lags=[lag], return_df=True))

#%%
# Empirical Density of Standardized Residuals
plt.figure(figsize=(10, 5))
sns.histplot(std_resid_gar, kde=True, stat="density", label='Empirical Density')
# Overlay standard normal for comparison
x_norm = np.linspace(std_resid_gar.min(), std_resid_gar.max(), 100)
plt.plot(x_norm, stats.norm.pdf(x_norm, 0, 1), label='Standard Normal PDF', color='red')
plt.title('Empirical Density of Standardized Residuals')
plt.legend()
plt.show()

#%%
from scipy.stats import jarque_bera
stat, p = jarque_bera(std_resid_egar_0)
print(f"JB stat={stat:.2f}, p-value={p:.3f}")

#%%
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch

# Diagnostics
lb_test = acorr_ljungbox(std_resid_gar_0.dropna(), lags=[10], return_df=True)
lb_test_squared = acorr_ljungbox(std_resid_gar_0.dropna()**2, lags=[10], return_df=True)

lm_arch = het_arch(std_resid_gar_0.dropna())

print("\nLjung-Box Test (Residuals):\n", lb_test)
print("\nLjung-Box Test (Squared Residuals):\n", lb_test_squared)
print("\nARCH LM Test:\n", lm_arch)
#%%
res_g_0.summary()
#%%
from scipy.stats import chisquare
# PIT under fitted skewt
nu = res_e_0.params['nu']
u = stats.t.cdf(std_resid_egar_0.dropna(), df=nu)

# test uniformity over 10 bins
hist, bins = np.histogram(u, bins=10, range=(0,1))
expected = np.full(10, len(u) / 10)
chi_stat, p_val = chisquare(hist, expected)

print(f"Chi-squared stat: {chi_stat:.2f}, p-value: {p_val:.4f}")


#%%


# PIT uniformity χ² tests for standardized residuals of two GARCH & two EGARCH fits
from scipy.stats import chisquare, t

for name, (std_resid, res) in [
    ("GARCH0",  (std_resid_gar_0,  res_g_0)),
    ("EGARCH0", (std_resid_egar_0, res_e_0)),
    ("GARCH1",  (std_resid_gar_1,  res_g_1)),
    ("EGARCH1", (std_resid_egar_1, res_e_1)),
]:
    u    = t.cdf(std_resid.dropna(), df=res.params["nu"])
    hist = np.histogram(u, bins=10, range=(0,1))[0]
    chi, p = chisquare(hist, np.full(10, len(u)/10))
    print(f"{name}: χ²={chi:.2f}, p={p:.4f}")



#%%
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.__future__ import reindexing

def run_diagnostics(train_returns_pct, res, label='GARCH'):
    returns = train_returns_pct.values
    fitted_vol = res.conditional_volatility
    residuals = returns  # mean is assumed 0
    standardized_resid = residuals / fitted_vol

    print(f"\nDiagnostics for {label} model:")
    
    lb1 = acorr_ljungbox(standardized_resid, lags=[10], return_df=True)
    print(f"Ljung-Box on residuals (lag 10): p-value = {lb1['lb_pvalue'].values[0]:.4f}")
    

    lb2 = acorr_ljungbox(standardized_resid**2, lags=[10], return_df=True)
    print(f"Ljung-Box on squared residuals (lag 10): p-value = {lb2['lb_pvalue'].values[0]:.4f}")
    
    print("Standard deviation of residuals:", np.std(standardized_resid))
#%%
from statsmodels.stats.diagnostic import acorr_ljungbox

run_diagnostics(train['portfolio'] * 100, res_g, label='GARCH')
run_diagnostics(train['portfolio'] * 100, res_e, label='EGARCH')

#%%
display(Markdown("""
### Model Comparison: GARCH vs EGARCH

- GARCH: no serial correlation or ARCH left; residual SD ≈ 1.00
- EGARCH: no serial correlation or ARCH left; residual SD ≈ 1.00
- GARCH_c: no serial correlation or ARCH left; residual SD ≈ 0.99
- EGARCH_c: no serial correlation but significant ARCH remains; residual SD ≈ 1.06

---

###  **Conclusion**:
GARCH_c delivers the cleanest fit with perfectly whitened and well-scaled residuals. EGARCH_c performs worst, leaving volatility clustering unaddressed.
- But we might be over fitting, for that we need the tests below
"""))
#%%
display(Markdown("""
### Comparison: Custom GARCH vs. ARCH Package Models

- **GARCH (Custom vs. ARCH package):** Both models perform well, but the custom GARCH shows slightly better residual diagnostics (higher p-values, lower standard deviation).  
- **EGARCH (Custom vs. ARCH package):** The ARCH package EGARCH is more balanced, while the custom EGARCH leaves a noticeable ARCH effect and slightly over-scales residuals (standard deviation > 1).  
- **Overall Conclusion:** Custom GARCH is robust and effective; however, the ARCH package EGARCH provides a cleaner and more reliable specification.
- Why is beta for e_garch with custom lags more smaller?
-  Because we let the model spread its memory across more time-points, the weight on the main lag shrinks
"""))

#%%
# Below works only for base GARCH fitted using arch package

print("GARCH")
print("AIC:", res_g.aic)
print("BIC:", res_g.bic)
print("EGARCH")
print("AIC:", res_e.aic)
print("BIC:", res_e.bic)
#%%

def calculate_aic_bic(log_lik, num_params, num_obs):
    aic = -2 * log_lik + 2 * num_params
    bic = -2 * log_lik + np.log(num_obs) * num_params
    return aic, bic

aic_g, bic_g = calculate_aic_bic(loglik_g, k_g, T_g)
aic_e, bic_e = calculate_aic_bic(loglik_e, k_e, T_e)

print("GARCH_c")
print("AIC:", aic_g) 
print("BIC:", bic_g)
print("EGARCH_c")
print("AIC:", aic_e)
print("BIC:", bic_e)

#%%
display(Markdown("""
### Model Fit Comparison: AIC and BIC
- **GARCH:** solid fit but higher AIC (3441.38) and BIC (3461.17).  
- **EGARCH:** slightly lower AIC (3437.17) but similar BIC (3461.91), diagnostics on par with GARCH.  
- **GARCH_c:** lowest AIC (3410.81) with clean, whitened residuals—**best overall**.  
- **EGARCH_c:** same AIC/BIC as GARCH_c but leaves ARCH effects—overfitting risk.

**Conclusion:** We recommend choosing **GARCH_c**— since it delivers the lowest AIC and perfectly whitened residuals with no remaining ARCH effects to be modelled.
"""))

#%%