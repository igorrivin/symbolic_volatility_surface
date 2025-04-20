# Diagnostics: Return Independence, Degrees of Freedom, Hurst Exponent, Ljung-Box, PACF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, linregress
from statsmodels.tsa.stattools import acf, pacf, adfuller, acovf, q_stat
from hurst import compute_Hc
import seaborn as sns
api_key = "5N9SBTYG0LHC7UWZ"
from alpha_vantage.timeseries import TimeSeries

# Estimate degrees of freedom of a Student-t fit
def fit_student_t(df_returns):
    from scipy.stats import t as t_dist
    from scipy.optimize import minimize

    def neg_log_likelihood(params):
        df_, loc_, scale_ = params
        if df_ <= 2 or scale_ <= 0:
            return np.inf
        return -np.sum(t_dist.logpdf(df_returns, df_, loc=loc_, scale=scale_))

    res = minimize(neg_log_likelihood, x0=[4.0, 0, np.std(df_returns)], method='Nelder-Mead')
    return res.x[0]

# Hurst exponent estimation
def estimate_hurst(ts):
    H, _, _ = compute_Hc(ts, kind='price', simplified=True)
    return H

# Ljung–Box test
def ljung_box_test(returns, lags=10):
    acovs = acovf(returns, fft=False, demean=True)
    q, p = q_stat(acovs[1:lags+1], len(returns))
    return p[-1]  # last p-value

# Full diagnostics pipeline for a ticker
def analyze_ticker(ticker='AAPL', max_lag=10):

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.sort_index()  # Ensure oldest -> newest

    log_prices = np.log(data['4. close'])

    results = []
    for lag in range(1, max_lag + 1):
        returns = log_prices.diff(lag).dropna()
        df_fit = fit_student_t(returns)
        hurst = estimate_hurst(returns.values)
        ac = acf(returns, nlags=1, fft=False)[1]
        pac = pacf(returns, nlags=1, method='yw')[1]
        lb_pval = ljung_box_test(returns)

        results.append({
        'Lag': lag,
        'DF': df_fit,
        'Hurst': hurst,
        'AC(1)': ac,
        'PAC(1)': pac,
        'LB p-val': lb_pval
        })

    df_results = pd.DataFrame(results)
    df_results['Ticker'] = ticker
    df_results.to_csv(f'diagnostics_{ticker}.csv', index=False)
    return df_results

# Run diagnostics on multiple tickers
if __name__ == "__main__":
    tickers = ['QQQ', 'AAPL', 'GOOG']
    dfs = [analyze_ticker(tk) for tk in tickers]
    full_df = pd.concat(dfs)

    sns.set(style="whitegrid")
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for metric, ax in zip(['DF', 'Hurst', 'AC(1)', 'PAC(1)'], axs):
        sns.lineplot(data=full_df, x='Lag', y=metric, hue='Ticker', marker='o', ax=ax)
        ax.set_title(f'{metric} vs. Lag')
        ax.axhline(0.5 if metric == 'Hurst' else None, linestyle='--', color='gray')

    plt.tight_layout()
    plt.show()

    print(full_df.round(3))
