# Symbolic Regression of Volatility and Price Models (Dual Track)

import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sympy import log as sym_log, Abs as sym_abs
from pysr.export_sympy import pysr2sympy
from sympy import latex

# Load the dataset from previous stage
features = pd.read_csv("synthetic_dualtrack_dataset.csv")

X = features[['Moneyness', 'Expiry', 'Utilization']]

# Track A: Symbolic model of volatility
print("\n--- Symbolic Regression: Volatility Surface ---\n")
y_vol = features['Volatility']
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_vol, test_size=0.2, random_state=42)

model_vol = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt", "exp", "abs"],
    extra_sympy_mappings={"log": sym_log, "abs": sym_abs},
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    verbosity=1
)

model_vol.fit(X_train_v, y_train_v)
y_pred_vol = model_vol.predict(X_test_v)
print("R^2 (Volatility):", r2_score(y_test_v, y_pred_vol))

# Export best volatility formula
best_eq_vol = model_vol.get_best()
print("Best volatility equation:", best_eq_vol)
expr_vol = pysr2sympy(str(best_eq_vol), feature_names_in=X.columns, extra_sympy_mappings={"log": sym_log, "abs": sym_abs})
print("LaTeX (Volatility):", latex(expr_vol))

# Track B: Symbolic model of option prices
print("\n--- Symbolic Regression: Option Pricing Model ---\n")
y_price = features['Price']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_price, test_size=0.2, random_state=42)

model_price = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt", "exp", "abs"],
    extra_sympy_mappings={"log": sym_log, "abs": sym_abs},
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    verbosity=1
)

model_price.fit(X_train_p, y_train_p)
y_pred_price = model_price.predict(X_test_p)
print("R^2 (Price):", r2_score(y_test_p, y_pred_price))

# Export best price formula
best_eq_price = model_price.get_best()
print("Best price equation:", best_eq_price)
expr_price = pysr2sympy(str(best_eq_price), feature_names_in=X.columns, extra_sympy_mappings={"log": sym_log, "abs": sym_abs})
print("LaTeX (Price):", latex(expr_price))

# Plot predictions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test_v, y_pred_vol, alpha=0.6, label='Volatility')
plt.plot([y_test_v.min(), y_test_v.max()], [y_test_v.min(), y_test_v.max()], 'k--')
plt.xlabel("True Volatility")
plt.ylabel("Predicted Volatility")
plt.title("Volatility Fit")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_p, y_pred_price, alpha=0.6, label='Price', color='darkgreen')
plt.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Price Fit")
plt.grid(True)

plt.tight_layout()
plt.show()
