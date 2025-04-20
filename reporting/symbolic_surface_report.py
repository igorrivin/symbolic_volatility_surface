# Generate Report and Visualize Symbolic Surfaces

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, sympify
from sympy.parsing.sympy_parser import parse_expr
from matplotlib import cm
import os

# Load symbolic summary
def load_summary(path="symbolic_models_summary.csv"):
    return pd.read_csv(path)

# Generate Markdown report
def generate_markdown_report(df, md_path="symbolic_models_report.md"):
    with open(md_path, 'w') as f:
        f.write("# Symbolic Models Summary\n\n")
        for _, row in df.iterrows():
            f.write(f"## {row['Ticker']}\n")
            f.write(f"**Flagged**: {'⚠️ Yes' if row['Flagged'] else '✅ No'}  \n")
            f.write(f"**Equation**: `{row['Equation']}`  \n")
            f.write(f"**LaTeX**: \\({row['LaTeX']}\\)\n\n")
    print(f"📄 Markdown report saved to {md_path}")

# Evaluate symbolic function over 2D slices
def plot_surface(expr_str, vars2=('Moneyness', 'Expiry'), fixed_values={'Utilization': 0.5},
                 ticker='Model', save_dir='surface_plots'):
    os.makedirs(save_dir, exist_ok=True)
    all_vars = ['Moneyness', 'Expiry', 'Utilization']
    sym_vars = symbols(all_vars)
    try:
        expr = parse_expr(expr_str, evaluate=False)
        func = lambdify(sym_vars, expr, 'numpy')
    except Exception as e:
        print(f"Failed to parse: {expr_str}\n{e}")
        return

    x_var, y_var = vars2
    x = np.linspace(0.8, 1.2, 50)
    y = np.linspace(0.01, 1.0, 50)
    X, Y = np.meshgrid(x, y)

    kwargs = {var: fixed_values.get(var, 0.5) for var in all_vars}
    kwargs[x_var] = X
    kwargs[y_var] = Y

    try:
        Z = func(kwargs['Moneyness'], kwargs['Expiry'], kwargs['Utilization'])
    except Exception as e:
        print(f"Eval error for {ticker}: {e}")
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel('Volatility')
    ax.set_title(f"{ticker}: {x_var} vs {y_var}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{ticker}_{x_var}_{y_var}.png"))
    plt.close()
    print(f"📊 Surface plot saved: {ticker} ({x_var}, {y_var})")

# Run full report and plot generation
if __name__ == "__main__":
    df = load_summary()
    generate_markdown_report(df)

    for _, row in df.iterrows():
        expr_str = row['Equation']
        ticker = row['Ticker']
        # Usual volatility surface: Moneyness vs Expiry
        plot_surface(expr_str, vars2=('Moneyness', 'Expiry'), ticker=ticker)
        # Other slices (optional)
        plot_surface(expr_str, vars2=('Utilization', 'Moneyness'), ticker=ticker)
        plot_surface(expr_str, vars2=('Utilization', 'Expiry'), ticker=ticker)
