# Adaptive Model Selector with Batch Symbolic Regression Reporting

import pandas as pd
import numpy as np
import pymc as pm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix
from pysr import PySRRegressor
from sympy import log as sym_log, Abs as sym_abs
from pysr.export_sympy import pysr2sympy
from sympy import latex
import yfinance as yf
import os

# Load diagnostics from CSV (generated per ticker)
def load_diagnostics(ticker):
    df = pd.read_csv(f"diagnostics_{ticker}.csv")
    avg_df = df[['DF', 'Hurst', 'PAC(1)', 'LB p-val']].mean()
    return avg_df

# Decide model configuration based on diagnostic profile
def get_model_template_for(ticker):
    stats = load_diagnostics(ticker)

    df = stats['DF']
    hurst = stats['Hurst']
    pac1 = stats['PAC(1)']
    lb_p = stats['LB p-val']

    config = {
        'likelihood': 'normal',
        'include_lags': False,
        'trend_features': False,
        'autoregressive': False,
        'prior_strength': 'medium'
    }

    if df < 4:
        config['likelihood'] = 'student-t'
        config['prior_strength'] = 'low'

    if hurst > 0.55:
        config['trend_features'] = True

    if pac1 > 0.2 or lb_p < 0.05:
        config['include_lags'] = True
        config['autoregressive'] = True

    return config

# Create dynamic features for modeling
def create_features(returns, config, lags=3):
    df = pd.DataFrame({'y': returns})
    if config['include_lags']:
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
    if config['trend_features']:
        df['trend'] = np.arange(len(df))
        df_spline = dmatrix("bs(trend, df=4, include_intercept=False) - 1", df, return_type='dataframe')
        df = pd.concat([df, df_spline], axis=1).drop(columns='trend')
    return df.dropna()

# Build and evaluate model with PyMC
def build_and_fit_model(df, config):
    y = df['y'].values
    X = df.drop(columns='y').values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        mu = intercept + pm.math.dot(X_scaled, beta)
        sigma = pm.HalfNormal("sigma", sigma=1)

        if config['likelihood'] == 'student-t':
            nu = pm.Exponential("nu", 1/10)
            obs = pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)
        else:
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

    return trace, X_scaled, y, df.drop(columns='y').columns

# Score and flag model performance
def evaluate_model(trace, X, y):
    post = trace.posterior
    beta_mean = post['beta'].mean(dim=('chain', 'draw')).values
    intercept_mean = post['intercept'].mean(dim=('chain', 'draw')).values.item()
    y_hat = np.dot(X, beta_mean) + intercept_mean
    r2 = r2_score(y, y_hat)
    flagged = r2 < 0.5
    return r2, flagged, y_hat

# Run PySR and save results
def symbolic_regression(X_df, y, flagged, label, out_path):
    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "exp", "abs"],
        extra_sympy_mappings={"log": sym_log, "abs": sym_abs},
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        verbosity=0
    )
    model.fit(X_df, y)
    eq = model.get_best()
    expr = pysr2sympy(str(eq), feature_names_in=X_df.columns, extra_sympy_mappings={"log": sym_log, "abs": sym_abs})
    latex_expr = latex(expr)

    result = {
        "Ticker": label,
        "Flagged": flagged,
        "Equation": str(eq),
        "LaTeX": latex_expr
    }
    df_out = pd.DataFrame([result])
    df_out.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"✅ Saved symbolic model for {label} (flagged: {flagged})")

# Batch runner
if __name__ == "__main__":
    output_csv = "symbolic_models_summary.csv"
    tickers = ['AAPL', 'GOOG', 'QQQ']

    if os.path.exists(output_csv):
        os.remove(output_csv)

    for ticker in tickers:
        print(f"\n🔍 Processing {ticker}...")
        config = get_model_template_for(ticker)
        data = yf.download(ticker, period='2y', progress=False)['Adj Close']
        returns = np.log(data).diff().dropna()
        df_feat = create_features(returns, config)

        trace, X_scaled, y, feature_names = build_and_fit_model(df_feat, config)
        r2, flagged, y_hat = evaluate_model(trace, X_scaled, y)

        print(f"R^2 score: {r2:.4f}")
        if flagged:
            print("⚠️ Low-performing model.")
        else:
            print("✅ Model passed threshold.")

        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        symbolic_regression(X_df, y, flagged, ticker, output_csv)
