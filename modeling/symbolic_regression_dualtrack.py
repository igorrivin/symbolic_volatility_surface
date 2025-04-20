# Symbolic Volatility Surface Modeling

This repository contains a complete pipeline for modeling and analyzing volatility surfaces and option pricing formulas derived directly from empirical returns. It leverages probabilistic programming (PyMC), symbolic regression (PySR), and diagnostic tools for financial time series.

## 📦 Repository Structure

```
symbolic-volatility-surface/
├── diagnostics/
│   └── return_dependence_diagnostics.py        # Hurst, DF, PAC, Ljung–Box
├── modeling/
│   └── adaptive_model_selector.py              # Diagnostics-aware PyMC + PySR modeling
│   └── symbolic_regression_dualtrack.py        # Compare volatility vs price surface modeling
├── reporting/
│   └── symbolic_surface_report.py              # Generate plots and Markdown report
│   └── bundle_report_archive.py                # Zip Markdown and 3D plots
│   └── upload_to_drive.py                      # Upload final report bundle to public Google Drive
├── data/
│   └── diagnostics_{ticker}.csv                # Output from diagnostics stage
│   └── symbolic_models_summary.csv             # Symbolic regression outputs
│   └── surface_plots/                          # 3D surface plot PNGs
├── symbolic_models_report.md                   # Markdown report of symbolic models
├── symbolic_model_report_bundle.zip            # Bundled report + plots
└── requirements.txt                            # Dependencies for local or Colab usage
```

## 🚀 Workflow

### 1. **Diagnostics**
Analyze asset return structure using:
- Degrees of freedom (Student-t fit)
- Hurst exponent
- Autocorrelation and Ljung–Box tests

Run:
```bash
python diagnostics/return_dependence_diagnostics.py
```

### 2. **Modeling**
Use diagnostics to select modeling strategy and generate symbolic models for:
- Volatility surfaces
- Direct pricing expressions

Run:
```bash
python modeling/adaptive_model_selector.py
```

### 3. **Reporting**
Generate Markdown summary and 3D surface plots:
```bash
python reporting/symbolic_surface_report.py
```

Bundle results:
```bash
python reporting/bundle_report_archive.py
```

Upload to Google Drive:
```bash
python reporting/upload_to_drive.py
```

## 🔄 Colab Usage
This project can be executed on Google Colab. Upload the repository or clone via:
```python
!git clone https://github.com/YOUR_USER/symbolic-volatility-surface.git
```
Then run scripts step by step via `%run path/to/script.py`

## 📋 Requirements
See `requirements.txt` for the full environment setup.

## 📚 Future Directions
- Comparison with Black-Scholes pricing outputs
- Alternative volatility factorization (e.g., local volatility, stochastic volatility)
- Interactive web visualization
- GitHub Pages for report hosting

## 👤 License & Acknowledgments
MIT License. Built with ❤️ using PyMC, PySR, SymPy, and Matplotlib.


