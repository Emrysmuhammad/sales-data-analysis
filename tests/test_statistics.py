import numpy as np
import pandas as pd
import pytest

from dataviz.statistics import (
    correlation_matrix,
    describe_categorical,
    describe_numeric,
    key_metrics,
    percent_change,
    revenue_by_category_region,
    revenue_by_month,
    strongest_correlations,
    summarize_by,
)


@pytest.fixture
def orders():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-03-10", "2024-03-11"]
            ),
            "category": ["Books", "Home", "Books", "Home"],
            "region": ["North", "North", "South", "South"],
            "units": [1, 2, 3, 4],
            "unit_price": [10.0, 20.0, 30.0, 40.0],
            "discount": [0.0, 0.1, 0.0, 0.2],
            "customer_age": [20, 30, 40, 50],
            "customer_rating": [3, 4, 5, 4],
            "marketing_spend": [100.0, 200.0, 300.0, 400.0],
            "revenue": [10.0, 36.0, 90.0, 128.0],
        }
    )


def test_numeric_summary_has_the_expected_numbers(orders):
    table = describe_numeric(orders, columns=("units", "revenue"))

    assert list(table.index) == ["units", "revenue"]
    assert table.loc["units", "count"] == 4
    assert table.loc["units", "mean"] == 2.5
    assert table.loc["units", "median"] == 2.5
    assert table.loc["units", "min"] == 1
    assert table.loc["units", "max"] == 4
    assert table.loc["revenue", "mean"] == pytest.approx(66.0)
    assert table.loc["units", "missing"] == 0


def test_numeric_summary_counts_missing_values(orders):
    orders.loc[0, "units"] = np.nan
    table = describe_numeric(orders, columns=("units",))
    assert table.loc["units", "missing"] == 1
    assert table.loc["units", "count"] == 3


def test_numeric_summary_ignores_unknown_columns_but_needs_one(orders):
    assert list(describe_numeric(orders, columns=("units", "nope")).index) == ["units"]
    with pytest.raises(ValueError):
        describe_numeric(orders, columns=("nope",))


def test_categorical_summary_counts_and_percentages(orders):
    table = describe_categorical(orders, "category")
    assert table.loc["Books", "count"] == 2
    assert table.loc["Books", "percent"] == 50.0
    assert table.index.name == "category"


def test_summary_by_group_is_sorted_by_total(orders):
    table = summarize_by(orders, "category")

    assert list(table.index) == ["Home", "Books"]          # 164 vs 100
    assert table.loc["Home", "sum"] == 164.0
    assert table.loc["Books", "count"] == 2
    assert table.loc["Home", "share_pct"] == pytest.approx(164 / 264 * 100, abs=0.01)


def test_summary_by_group_without_a_sum(orders):
    table = summarize_by(orders, "category", aggregations=("count", "mean"))
    assert "share_pct" not in table.columns
    assert set(table.index) == {"Books", "Home"}


def test_category_by_region_table(orders):
    table = revenue_by_category_region(orders)
    assert table.loc["Books", "North"] == 10.0
    assert table.loc["Books", "South"] == 90.0
    assert table.loc["Home", "North"] == 36.0
    assert table.loc["Home", "South"] == 128.0


def test_monthly_revenue_fills_the_months_without_orders(orders):
    monthly = revenue_by_month(orders)

    assert list(monthly.index) == list(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    assert monthly.tolist() == [46.0, 0.0, 218.0]
    assert monthly.name == "revenue"


def test_percent_change():
    series = pd.Series([100.0, 150.0, 0.0, 50.0])
    result = percent_change(series)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 50.0
    assert result.iloc[2] == -100.0
    assert np.isnan(result.iloc[3])           # a change from 0 cannot be expressed in percent


def test_correlation_matrix_is_square_and_symmetric(orders):
    corr = correlation_matrix(orders, columns=("units", "unit_price", "customer_age"))

    assert corr.shape == (3, 3)
    assert (np.diag(corr.to_numpy()) == 1.0).all()
    assert corr.loc["units", "unit_price"] == corr.loc["unit_price", "units"]
    assert corr.loc["units", "customer_age"] == pytest.approx(1.0)   # perfectly in step


def test_correlation_detects_opposite_movement():
    table = pd.DataFrame({"a": [1, 2, 3, 4], "b": [8, 6, 4, 2]})
    assert correlation_matrix(table, columns=("a", "b")).loc["a", "b"] == pytest.approx(-1.0)


def test_correlation_rejects_an_unknown_method(orders):
    with pytest.raises(ValueError):
        correlation_matrix(orders, columns=("units", "revenue"), method="magic")


def test_strongest_correlations_lists_each_pair_once():
    corr = pd.DataFrame(
        [[1.0, 0.9, -0.95], [0.9, 1.0, 0.1], [-0.95, 0.1, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    top = strongest_correlations(corr, n=2)

    assert len(top) == 2
    assert (top.loc[0, "variable_1"], top.loc[0, "variable_2"]) == ("a", "c")   # |-0.95| is biggest
    assert top.loc[0, "correlation"] == -0.95
    assert (top.loc[1, "variable_1"], top.loc[1, "variable_2"]) == ("a", "b")


def test_key_metrics(orders):
    metrics = key_metrics(orders)

    assert metrics["orders"] == 4
    assert metrics["units_sold"] == 10
    assert metrics["total_revenue"] == 264.0
    assert metrics["average_order_value"] == 66.0
    assert metrics["top_category"] == "Home"
    assert metrics["top_region"] == "South"
    assert metrics["best_month"] == "2024-03"
    assert metrics["worst_month"] == "2024-02"
    assert metrics["first_order_date"] == "2024-01-05"
    assert metrics["last_order_date"] == "2024-03-11"
    assert isinstance(metrics["orders"], int)
    assert isinstance(metrics["total_revenue"], float)


def test_key_metrics_needs_data(orders):
    with pytest.raises(ValueError):
        key_metrics(orders.iloc[0:0])