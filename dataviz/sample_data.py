"""Create the sample sales dataset used throughout the project.

The data is synthetic but realistic: it has a seasonal pattern, prices that
depend on the product category, and a few relationships between the columns.
It is then deliberately made MESSY, like real-world data, so the cleaning step
has something to do: duplicate orders, missing values, prices written as text,
inconsistent spelling, impossible values, broken dates and extreme outliers.

Run:  python -m dataviz.sample_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataviz import config

DEFAULT_SEED = 42
DEFAULT_ROWS = 1200

# weight = share of orders, price = (lowest, highest) unit price,
# extra_units = how many extra units a typical order of this category contains
CATEGORIES = {
    "Electronics": {"weight": 0.22, "price": (150.0, 900.0), "extra_units": 0.2},
    "Clothing": {"weight": 0.26, "price": (20.0, 120.0), "extra_units": 1.0},
    "Home": {"weight": 0.20, "price": (30.0, 250.0), "extra_units": 0.6},
    "Sports": {"weight": 0.17, "price": (25.0, 200.0), "extra_units": 0.5},
    "Books": {"weight": 0.15, "price": (8.0, 40.0), "extra_units": 1.4},
}
REGIONS = ("North", "South", "East", "West")
REGION_WEIGHTS = (0.30, 0.22, 0.26, 0.22)
DISCOUNTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
DISCOUNT_WEIGHTS = (0.40, 0.15, 0.20, 0.12, 0.08, 0.05)

# How busy each month is (January to December): a big peak in November and December
SEASONALITY = np.array([0.8, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.8, 2.2])


def generate_clean_orders(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate perfectly clean orders (the 'truth' before the mess is added)."""
    if rows < 1:
        raise ValueError("rows must be at least 1")

    rng = np.random.default_rng(seed)

    # Orders are spread over two years, with more of them in busy months
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    day_weights = SEASONALITY[days.month.to_numpy() - 1]
    chosen = rng.choice(len(days), size=rows, p=day_weights / day_weights.sum())
    order_dates = days[chosen]
    season = SEASONALITY[order_dates.month.to_numpy() - 1]

    names = list(CATEGORIES)
    category = rng.choice(names, size=rows, p=[CATEGORIES[name]["weight"] for name in names])

    lowest = np.array([CATEGORIES[name]["price"][0] for name in category])
    highest = np.array([CATEGORIES[name]["price"][1] for name in category])
    unit_price = np.round(lowest + (highest - lowest) * rng.beta(2, 5, size=rows), 2)

    extra = np.array([CATEGORIES[name]["extra_units"] for name in category])
    units = 1 + rng.poisson(extra * season)

    discount = rng.choice(DISCOUNTS, size=rows, p=DISCOUNT_WEIGHTS)
    age = np.clip(np.round(rng.normal(36, 11, rows)), 18, 75).astype(int)

    # Customers who got a bigger discount tend to be a little happier
    rating = np.clip(np.round(rng.normal(3.6 + 2.0 * discount, 0.9)), 1, 5).astype(int)

    # More money is spent on marketing in busy months
    marketing = np.round(np.clip(150 + 120 * season + rng.normal(0, 30, rows), 20, None), 2)

    revenue = np.round(units * unit_price * (1 - discount), 2)

    frame = pd.DataFrame(
        {
            "order_id": [f"ORD-{number:05d}" for number in range(1, rows + 1)],
            "order_date": order_dates,
            "region": rng.choice(REGIONS, size=rows, p=REGION_WEIGHTS),
            "category": category,
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "customer_age": age,
            "customer_rating": rating,
            "marketing_spend": marketing,
            "revenue": revenue,
        }
    )
    return frame[list(config.COLUMNS)]


def make_messy(clean: pd.DataFrame, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn clean orders into a messy 'raw export'.

    Returns the messy table and a count of every kind of problem that was added.
    """
    rng = np.random.default_rng(seed + 1)

    messy = clean.copy()
    messy["order_date"] = messy["order_date"].dt.strftime("%Y-%m-%d")
    for column in messy.columns:
        messy[column] = messy[column].astype(object)

    problems: dict[str, int] = {}

    def pick(share: float) -> np.ndarray:
        size = max(1, int(round(len(messy) * share)))
        return rng.choice(messy.index.to_numpy(), size=size, replace=False)

    # 1. Inconsistent spelling: lower case, UPPER CASE or extra spaces
    for column, share in (("category", 0.06), ("region", 0.06)):
        rows = pick(share)
        for row in rows:
            value = str(messy.at[row, column])
            messy.at[row, column] = str(rng.choice([value.lower(), value.upper(), f"  {value} "]))
        problems[f"messy_{column}_text"] = len(rows)

    # 2. Extreme prices (data-entry mistakes: 20 times too high)
    rows = pick(0.005)
    new_price = (clean.loc[rows, "unit_price"] * 20).round(2)
    messy.loc[rows, "unit_price"] = new_price
    messy.loc[rows, "revenue"] = (
        clean.loc[rows, "units"] * new_price * (1 - clean.loc[rows, "discount"])
    ).round(2)
    problems["extreme_unit_prices"] = len(rows)

    # 3. Impossible number of units (returns or typing mistakes)
    rows = pick(0.007)
    messy.loc[rows, "units"] = rng.choice([0, -1], size=len(rows))
    problems["invalid_units"] = len(rows)

    # 4. Impossible ages
    rows = pick(0.007)
    half = len(rows) // 2
    messy.loc[rows[:half], "customer_age"] = 999
    messy.loc[rows[half:], "customer_age"] = 0
    problems["invalid_ages"] = len(rows)

    # 5. Broken dates
    rows = pick(0.005)
    messy.loc[rows, "order_date"] = rng.choice(["not available", "31/02/2024", "N/A"], size=len(rows))
    problems["invalid_dates"] = len(rows)

    # 6. Revenue written as text with a currency sign, like "$1,234.50"
    rows = pick(0.04)
    messy.loc[rows, "revenue"] = [f"${float(messy.at[row, 'revenue']):,.2f}" for row in rows]
    problems["dollar_text_revenue"] = len(rows)

    # 7. Missing values
    for column, share in (
        ("customer_age", 0.04),
        ("customer_rating", 0.03),
        ("region", 0.02),
        ("marketing_spend", 0.015),
        ("revenue", 0.015),
    ):
        rows = pick(share)
        messy.loc[rows, column] = np.nan
        problems[f"missing_{column}"] = len(rows)

    # 8. Duplicate orders (the same row exported twice), then shuffle everything
    duplicates = messy.sample(n=min(20, len(messy)), random_state=seed)
    problems["duplicate_rows"] = len(duplicates)
    messy = pd.concat([messy, duplicates], ignore_index=True)
    messy = messy.sample(frac=1, random_state=seed).reset_index(drop=True)

    return messy, problems


def write_sample_csv(
    path: str | Path | None = None,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, dict[str, int]]:
    """Create the messy sample dataset and save it as a CSV file."""
    target = Path(path) if path else config.RAW_DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    clean = generate_clean_orders(rows, seed)
    messy, problems = make_messy(clean, seed)
    messy.to_csv(target, index=False)
    return target, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the sample sales dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="number of orders")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--output", default=None, help="where to save the CSV file")
    args = parser.parse_args(argv)

    path, problems = write_sample_csv(args.output, args.rows, args.seed)

    print(f"Saved {args.rows} orders (plus duplicates) to {path}\n")
    print("Problems added to the data:")
    for name, count in problems.items():
        print(f"  {name:<26} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())