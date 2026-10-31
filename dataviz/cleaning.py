"""Data cleaning functions.

Every function takes a DataFrame and returns a NEW one, so the raw data is
never changed by accident. clean_sales_data() chains them together in the
right order and returns a report of what it did.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from dataviz import config

# Values outside these ranges are treated as mistakes
VALID_AGE = (16, 100)
VALID_RATING = (1, 5)
MAX_DISCOUNT = 0.9


# ---------- small, single-purpose steps ----------


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """' Order Date ' -> 'order_date'."""
    out = df.copy()
    out.columns = [str(name).strip().lower().replace(" ", "_") for name in out.columns]
    return out


def standardize_text(
    df: pd.DataFrame, columns: tuple[str, ...] = config.CATEGORICAL_COLUMNS
) -> pd.DataFrame:
    """Trim spaces and use one spelling style: '  nORTH ' -> 'North'. Blank text becomes missing."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            text = out[column].astype("string").str.strip().str.title()
            out[column] = text.replace("", pd.NA)
    return out


def parse_dates(
    df: pd.DataFrame,
    column: str = config.DATE_COLUMN,
    date_format: str | None = "%Y-%m-%d",
) -> pd.DataFrame:
    """Turn text into real dates. Anything that is not a valid date becomes NaT (missing).

    Pass date_format=None to let pandas work out several different formats.
    """
    out = df.copy()
    if date_format is None:
        out[column] = pd.to_datetime(out[column], errors="coerce", format="mixed")
    else:
        out[column] = pd.to_datetime(out[column], errors="coerce", format=date_format)
    return out


def drop_missing_dates(df: pd.DataFrame, column: str = config.DATE_COLUMN) -> pd.DataFrame:
    """Remove rows whose date is missing: an order without a date cannot be placed in time."""
    return df[df[column].notna()].reset_index(drop=True)


def to_numeric(
    df: pd.DataFrame, columns: tuple[str, ...] = config.NUMERIC_COLUMNS
) -> pd.DataFrame:
    """Turn text such as '$1,234.50' into the number 1234.5. Anything unreadable becomes NaN."""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        text = out[column].astype(str).str.replace(r"[$,\s]", "", regex=True)
        out[column] = pd.to_numeric(text, errors="coerce").astype("float64")
    return out


def mark_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible values (age 999, rating 7, discount 150%...) with NaN."""
    out = df.copy()
    rules = {
        "customer_age": VALID_AGE,
        "customer_rating": VALID_RATING,
        "discount": (0.0, MAX_DISCOUNT),
    }
    for column, (low, high) in rules.items():
        if column in out.columns:
            invalid = out[column].notna() & ~out[column].between(low, high)
            out.loc[invalid, column] = np.nan
    return out


def drop_duplicate_orders(
    df: pd.DataFrame, subset: tuple[str, ...] = (config.ID_COLUMN,)
) -> pd.DataFrame:
    """Keep the first row of every order id."""
    return df.drop_duplicates(subset=list(subset), keep="first").reset_index(drop=True)


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove orders that cannot be real: no units, no price, or a negative quantity."""
    valid = df["units"].gt(0) & df["unit_price"].gt(0)
    return df[valid].reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the gaps that can be filled sensibly.

    - ages, ratings and marketing spend -> the median (robust against outliers)
    - discount                           -> 0 (no discount recorded)
    - region and category                -> 'Unknown'
    - revenue                            -> units x unit price x (1 - discount)
    """
    out = df.copy()

    for column in ("customer_age", "customer_rating", "marketing_spend"):
        out[column] = out[column].fillna(out[column].median())
    out["discount"] = out["discount"].fillna(0.0)

    for column in config.CATEGORICAL_COLUMNS:
        out[column] = out[column].fillna("Unknown")

    computed = out["units"] * out["unit_price"] * (1 - out["discount"])
    out["revenue"] = out["revenue"].fillna(computed.round(2))
    return out


# ---------- outliers ----------


def _iqr_bounds(values: pd.Series, factor: float) -> tuple[float, float]:
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    return q1 - factor * spread, q3 + factor * spread


def flag_outliers(
    df: pd.DataFrame,
    column: str,
    by: str | None = None,
    factor: float = 3.0,
) -> pd.Series:
    """True for values far outside the 'normal' range (the IQR rule).

    A value is an outlier when it is more than `factor` x IQR beyond the quartiles.
    factor=1.5 is the classic rule; 3.0 only catches extreme values.
    With `by`, the range is worked out separately for each group, which is what
    you need when groups differ a lot (a 900 dollar laptop is normal, a 900 dollar book is not).
    """
    if by is None:
        low, high = _iqr_bounds(df[column], factor)
        return ~df[column].between(low, high)

    def per_group(values: pd.Series) -> pd.Series:
        if len(values) < 4:  # too few rows to judge
            return pd.Series(False, index=values.index)
        low, high = _iqr_bounds(values, factor)
        return ~values.between(low, high)

    return df.groupby(by, observed=True)[column].transform(per_group).astype(bool)


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    by: str | None = None,
    factor: float = 3.0,
) -> pd.DataFrame:
    """Drop the rows flagged by flag_outliers()."""
    mask = flag_outliers(df, column, by=by, factor=factor)
    return df[~mask].reset_index(drop=True)


def finalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Use whole numbers for columns that are counts."""
    out = df.copy()
    out["units"] = out["units"].round().astype(int)
    out["customer_age"] = out["customer_age"].round().astype(int)
    out["customer_rating"] = out["customer_rating"].round().clip(*VALID_RATING).astype(int)
    return out


# ---------- the whole pipeline ----------


@dataclass
class CleaningReport:
    """What the cleaning pipeline did, step by step."""

    rows_in: int = 0
    rows_out: int = 0
    invalid_dates_removed: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    outliers_removed: int = 0
    invalid_values_blanked: dict[str, int] = field(default_factory=dict)
    missing_values_filled: dict[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict:
        data = asdict(self)
        data["rows_removed"] = self.rows_removed
        return data

    def summary(self) -> str:
        lines = [
            f"Rows in:                    {self.rows_in:,}",
            f"  invalid dates removed:    {self.invalid_dates_removed:,}",
            f"  duplicate orders removed: {self.duplicates_removed:,}",
            f"  invalid rows removed:     {self.invalid_rows_removed:,}",
            f"  outliers removed:         {self.outliers_removed:,}",
            f"Rows out:                   {self.rows_out:,}",
        ]
        if self.invalid_values_blanked:
            lines.append("Impossible values blanked: " + _format_counts(self.invalid_values_blanked))
        if self.missing_values_filled:
            lines.append("Missing values filled:     " + _format_counts(self.missing_values_filled))
        return "\n".join(lines)


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} ({count})" for name, count in counts.items())


def clean_sales_data(
    raw: pd.DataFrame, outlier_factor: float = 3.0
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw sales table. Returns the clean table and a CleaningReport."""
    report = CleaningReport(rows_in=len(raw))

    df = standardize_columns(raw)
    df = standardize_text(df)

    # 1. Dates: orders without a usable date are removed
    df = parse_dates(df)
    before = len(df)
    df = drop_missing_dates(df)
    report.invalid_dates_removed = before - len(df)

    # 2. Numbers: "$1,234.50" -> 1234.5, then blank out impossible values
    df = to_numeric(df)
    checked = ["customer_age", "customer_rating", "discount"]
    missing_before = {column: int(df[column].isna().sum()) for column in checked}
    df = mark_invalid_values(df)
    blanked = {column: int(df[column].isna().sum()) - missing_before[column] for column in checked}
    report.invalid_values_blanked = {column: n for column, n in blanked.items() if n > 0}

    # 3. Duplicates and impossible orders
    before = len(df)
    df = drop_duplicate_orders(df)
    report.duplicates_removed = before - len(df)

    before = len(df)
    df = remove_invalid_rows(df)
    report.invalid_rows_removed = before - len(df)

    # 4. Fill the remaining gaps
    fill_columns = [
        "customer_age", "customer_rating", "marketing_spend", "discount", "revenue",
        *config.CATEGORICAL_COLUMNS,
    ]
    missing = {column: int(df[column].isna().sum()) for column in fill_columns}
    report.missing_values_filled = {column: n for column, n in missing.items() if n > 0}
    df = fill_missing(df)

    # 5. Extreme prices (compared with other products of the same category)
    before = len(df)
    df = remove_outliers(df, "unit_price", by="category", factor=outlier_factor)
    report.outliers_removed = before - len(df)

    df = finalize_types(df)
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    report.rows_out = len(df)
    return df, report