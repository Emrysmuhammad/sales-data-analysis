"""Project-wide settings: folders, column names and the plot style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler

# ---------- folders ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sales_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------- columns of the sales dataset ----------
ID_COLUMN = "order_id"
DATE_COLUMN = "order_date"
CATEGORICAL_COLUMNS = ("region", "category")
NUMERIC_COLUMNS = (
    "units",
    "unit_price",
    "discount",
    "customer_age",
    "customer_rating",
    "marketing_spend",
    "revenue",
)
COLUMNS = (ID_COLUMN, DATE_COLUMN) + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# ---------- chart colors ----------
PALETTE = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#14b8a6"]


def apply_plot_style() -> None:
    """Give every chart the same clean look."""
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)