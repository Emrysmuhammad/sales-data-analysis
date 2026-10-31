"""Descriptive statistics and summary tables for the cleaned sales data."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from dataviz import config


def describe_numeric(
    df: pd.DataFrame, columns: Iterable[str] = config.NUMERIC_COLUMNS
) -> pd.DataFrame:
    """A summary of every numeric column.

    Columns: count, missing, mean, median, std, min, 25%, 75%, max and skew.
    Skew tells you which way the values lean: above 0 means a long tail of
    high values, below 0 means a long tail of low values, near 0 is balanced.
    """
    present = [column for column in columns if column in df.columns]
    if not present:
        raise ValueError("None of the requested columns exist in the table")

    numbers = df[present]
    table = numbers.describe().T
    table["median"] = numbers.median()
    table["skew"] = numbers.skew()
    table["missing"] = numbers.isna().sum()

    order = ["count", "missing", "mean", "median", "std", "min", "25%", "75%", "max", "skew"]
    return table[order].round(2)


def describe_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """How often each value of a text column appears (count and percent)."""
    counts = df[column].value_counts(dropna=False)
    table = counts.rename("count").to_frame()
    table["percent"] = (counts / counts.sum() * 100).round(1)
    table.index.name = column
    return table


def summarize_by(
    df: pd.DataFrame,
    group_column: str,
    value_column: str = "revenue",
    aggregations: Sequence[str] = ("count", "sum", "mean", "median"),
) -> pd.DataFrame:
    """Summarize one column for every group, biggest total first.

    When 'sum' is included, a share_pct column shows each group's share of the total.
    """
    grouped = df.groupby(group_column, observed=True)[value_column].agg(list(aggregations))

    if "sum" in grouped.columns:
        grouped["share_pct"] = grouped["sum"] / grouped["sum"].sum() * 100
        grouped = grouped.sort_values("sum", ascending=False)
    else:
        grouped = grouped.sort_values(grouped.columns[0], ascending=False)

    return grouped.round(2)


def revenue_by_category_region(df: pd.DataFrame, value_column: str = "revenue") -> pd.DataFrame:
    """Total revenue for every category (rows) and region (columns)."""
    table = df.pivot_table(
        index="category",
        columns="region",
        values=value_column,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    return table.round(2)


def revenue_by_month(
    df: pd.DataFrame,
    date_column: str = config.DATE_COLUMN,
    value_column: str = "revenue",
) -> pd.Series:
    """Total revenue per calendar month. Months without orders show 0."""
    months = df[date_column].dt.to_period("M")
    monthly = df.groupby(months)[value_column].sum()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index().asfreq("MS", fill_value=0)
    monthly.index.name = "month"
    monthly.name = value_column
    return monthly


def percent_change(series: pd.Series) -> pd.Series:
    """Change from the previous value in percent. The first value (and changes from 0) are NaN."""
    previous = series.shift(1).replace(0, np.nan)
    return ((series - previous) / previous * 100).round(1)


def correlation_matrix(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """How strongly every pair of numeric columns moves together (-1 to +1).

    +1 means they rise together, -1 means one rises as the other falls and 0 means no
    straight-line relationship. method can be 'pearson', 'spearman' or 'kendall'.
    """
    if columns is None:
        columns = [column for column in config.NUMERIC_COLUMNS if column in df.columns]
    return df[list(columns)].corr(method=method).round(3)


def strongest_correlations(corr: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n strongest relationships (positive or negative), each pair listed once."""
    names = list(corr.columns)
    pairs = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = corr.loc[first, second]
            if not np.isnan(value):
                pairs.append((first, second, float(value)))

    pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)
    return pd.DataFrame(pairs[:n], columns=["variable_1", "variable_2", "correlation"])


def key_metrics(df: pd.DataFrame, date_column: str = config.DATE_COLUMN) -> dict:
    """The headline numbers of the whole dataset, as plain Python values."""
    if df.empty:
        raise ValueError("There is no data to summarize")

    monthly = revenue_by_month(df, date_column)
    by_category = df.groupby("category", observed=True)["revenue"].sum()
    by_region = df.groupby("region", observed=True)["revenue"].sum()

    return {
        "orders": int(len(df)),
        "units_sold": int(df["units"].sum()),
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "average_order_value": round(float(df["revenue"].mean()), 2),
        "median_order_value": round(float(df["revenue"].median()), 2),
        "average_discount_pct": round(float(df["discount"].mean() * 100), 1),
        "average_rating": round(float(df["customer_rating"].mean()), 2),
        "average_customer_age": round(float(df["customer_age"].mean()), 1),
        "top_category": str(by_category.idxmax()),
        "top_category_revenue": round(float(by_category.max()), 2),
        "top_region": str(by_region.idxmax()),
        "top_region_revenue": round(float(by_region.max()), 2),
        "best_month": monthly.idxmax().strftime("%Y-%m"),
        "best_month_revenue": round(float(monthly.max()), 2),
        "worst_month": monthly.idxmin().strftime("%Y-%m"),
        "worst_month_revenue": round(float(monthly.min()), 2),
        "first_order_date": df[date_column].min().strftime("%Y-%m-%d"),
        "last_order_date": df[date_column].max().strftime("%Y-%m-%d"),
    }