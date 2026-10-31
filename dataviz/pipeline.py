"""Run the whole analysis from the raw CSV file to charts and reports.

Run:  python -m dataviz.pipeline
      python -m dataviz.pipeline --input my_data.csv --output results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataviz import config
from dataviz.cleaning import clean_sales_data
from dataviz.export import (
    export_clean_data,
    export_json,
    export_markdown_report,
    export_tables,
)
from dataviz.loader import DataLoadError, load_csv
from dataviz.plots.bar import plot_bar, plot_grouped_bar
from dataviz.plots.heatmap import plot_correlation_heatmap
from dataviz.plots.histogram import plot_histogram, plot_histogram_grid
from dataviz.plots.line import plot_monthly_revenue
from dataviz.statistics import (
    correlation_matrix,
    describe_numeric,
    key_metrics,
    revenue_by_category_region,
    revenue_by_month,
    strongest_correlations,
    summarize_by,
)


def run_analysis(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    rolling_window: int = 3,
) -> dict:
    """Load, clean, analyse, draw and export. Returns everything that was produced."""
    out = Path(output_dir) if output_dir else config.OUTPUT_DIR
    figures_dir = out / "figures"

    # 1. Load and clean
    raw = load_csv(input_path)
    clean, report = clean_sales_data(raw)
    if clean.empty:
        raise ValueError("No rows are left after cleaning, so there is nothing to analyse")

    # 2. Statistics
    metrics = key_metrics(clean)
    monthly = revenue_by_month(clean)
    by_category = summarize_by(clean, "category")
    by_region = summarize_by(clean, "region")
    category_region = revenue_by_category_region(clean)
    corr = correlation_matrix(clean)

    tables = {
        "numeric_summary": describe_numeric(clean),
        "revenue_by_category": by_category,
        "revenue_by_region": by_region,
        "revenue_by_category_and_region": category_region,
        "monthly_revenue": monthly.to_frame(),
        "correlations": corr,
        "strongest_correlations": strongest_correlations(corr, n=5),
    }

    # 3. Charts
    figures = {
        "Monthly revenue": figures_dir / "monthly_revenue.png",
        "Revenue by category": figures_dir / "revenue_by_category.png",
        "Revenue by region": figures_dir / "revenue_by_region.png",
        "Revenue by category and region": figures_dir / "revenue_by_category_and_region.png",
        "Customer age": figures_dir / "customer_age_histogram.png",
        "Distributions": figures_dir / "distributions.png",
        "Correlation heatmap": figures_dir / "correlation_heatmap.png",
    }

    plot_monthly_revenue(
        monthly, rolling_window=rolling_window, save_path=figures["Monthly revenue"], close=True
    )
    plot_bar(
        by_category["sum"], title="Revenue by category", ylabel="Revenue ($)",
        save_path=figures["Revenue by category"], close=True,
    )
    plot_bar(
        by_region["sum"], title="Revenue by region", xlabel="Revenue ($)", horizontal=True,
        save_path=figures["Revenue by region"], close=True,
    )
    plot_grouped_bar(
        category_region, title="Revenue by category and region", ylabel="Revenue ($)",
        save_path=figures["Revenue by category and region"], close=True,
    )
    plot_histogram(
        clean["customer_age"], title="Customer age", xlabel="Age (years)",
        save_path=figures["Customer age"], close=True,
    )
    plot_histogram_grid(
        clean, ("customer_age", "unit_price", "revenue", "customer_rating"),
        save_path=figures["Distributions"], close=True,
    )
    plot_correlation_heatmap(corr, save_path=figures["Correlation heatmap"], close=True)

    # 4. Export
    files = {
        "clean_data": export_clean_data(clean, out / "clean_sales.csv"),
        "metrics": export_json(metrics, out / "summary.json"),
        "cleaning_report": export_json(report.as_dict(), out / "cleaning_report.json"),
    }
    files.update({f"table_{name}": path for name, path in export_tables(tables, out / "tables").items()})
    files["report"] = export_markdown_report(
        out / "report.md", metrics, report.as_dict(), tables, figures
    )

    return {
        "metrics": metrics,
        "cleaning": report,
        "tables": tables,
        "figures": figures,
        "files": files,
        "clean_data": clean,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sales analysis.")
    parser.add_argument("--input", default=None, help="CSV file to analyse (default: the sample data)")
    parser.add_argument("--output", default=None, help="folder for the results (default: outputs/)")
    parser.add_argument("--rolling-window", type=int, default=3, help="months in the moving average")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")      # save charts to files, do not open windows

    try:
        result = run_analysis(args.input, args.output, args.rolling_window)
    except (DataLoadError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(result["cleaning"].summary())
    print("\nKey metrics")
    for name, value in result["metrics"].items():
        print(f"  {name:<24} {value}")

    print("\nFiles written:")
    for name, path in result["files"].items():
        print(f"  {name:<34} {path}")
    for name, path in result["figures"].items():
        print(f"  figure: {name:<26} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())