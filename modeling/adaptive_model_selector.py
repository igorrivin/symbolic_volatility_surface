# Adaptive PyMC and PySR Modeling Pipeline

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from patsy import dmatrix
from pysr import PySRRegressor, best
from pysr.export_sympy import pysr2sympy
from sympy import log as sym_log, Abs as sym_abs, latex
import os

np.seterr(all="ignore")  # Ignore harmless numerical warnings

# --- PyMC wrapper ---
def get_model_template_for(X, y):
    with pm.Model() as model:
        coeffs = pm.Normal("coeffs", mu=0, sigma=1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.math.dot(X, coeffs)
        obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    return model

# --- Symbolic regression and reporting ---
def symbolic_regression(X_df, y, flagged, label, out_path):
    model = PySRRegressor(
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "abs", "sqrt"],
        model_selection="best",
        maxsize=30,
        verbosity=0,
        extra_sympy_mappings={"log": sym_log, "abs": sym_abs},
        output_jax_format=False,
        progress=False
    )
    model.fit(X_df, y)

    best_eq = model.get_best()
    equation_str = str(best_eq['equation']).strip().splitlines()[0]  # ✅ Clean, single-line expression
    print("✅ Equation string to parse:", repr(equation_str))

    try:
        expr = pysr2sympy(equation_str, feature_names_in=X_df.columns)
        latex_expr = latex(expr)
    except Exception as e:
        print("Failed to parse:", equation_str)
        print(e)
        expr = ""
        latex_expr = ""

    new_row = pd.DataFrame([{
        "Ticker": label,
        "Equation": equation_str,
        "LaTeX": latex_expr,
        "Flagged": flagged
    }])

    if os.path.exists(out_path):
        pd.concat([pd.read_csv(out_path), new_row]).to_csv(out_path, index=False)
    else:
        new_row.to_csv(out_path, index=False)

    print(f"✅ Saved symbolic model for {label} (flagged: {flagged})")

# --- Main pipeline ---
if __name__ == "__main__":
    output_csv = "symbolic_models_summary.csv"
    tickers = ["QQQ", "AAPL", "GOOG"]

    for ticker in tickers:
        print(f"\n🧪 Modeling: {ticker}")
        df = pd.read_csv(f"data/diagnostics_{ticker}.csv")

        # Simple feature generation: spline on lag
        df_spline = dmatrix("bs(Lag, df=4, include_intercept=False)", df, return_type='dataframe')
        df_spline.columns = [f"spline_{i}" for i in range(df_spline.shape[1])]  # ✅ Fix names

        X = df_spline.values
        y = df["DF"].values
        flagged = df["Flagged"].iloc[0] if "Flagged" in df.columns else False

        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

        # PyMC modeling
        model = get_model_template_for(X_train, y_train)
        with model:
            trace = pm.sample(2000, tune=1000, cores=1, chains=2, progressbar=True)
            posterior_pred = pm.sample_posterior_predictive(trace, model=model)
        y_pred = posterior_pred.posterior_predictive["y"].mean(dim=("chain", "draw")).values
        r2 = r2_score(y_train, y_pred)

        print(f"R^2 score: {r2:.4f}")
        if r2 < 0.1:
            print("⚠️ Low-performing model.")    


        X_df = pd.DataFrame(X_scaled, columns=df_spline.columns)  # Use cleaned column names
        symbolic_regression(X_df, y, flagged, ticker, output_csv)
