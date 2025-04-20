# Bundle Report and Surface Plots into ZIP Archive

import os
import zipfile

# Configuration
report_file = "symbolic_models_report.md"
plots_folder = "surface_plots"
output_zip = "symbolic_model_report_bundle.zip"

# Create ZIP bundle
def zip_report_and_plots(report_path, plot_dir, output_path):
    with zipfile.ZipFile(output_path, 'w') as zipf:
        # Add markdown report
        if os.path.exists(report_path):
            zipf.write(report_path, arcname=os.path.basename(report_path))
        # Add surface plots
        if os.path.isdir(plot_dir):
            for fname in os.listdir(plot_dir):
                fpath = os.path.join(plot_dir, fname)
                zipf.write(fpath, arcname=os.path.join(plot_dir, fname))
    print(f"📦 Bundle saved to: {output_path}")

if __name__ == "__main__":
    zip_report_and_plots(report_file, plots_folder, output_zip)
