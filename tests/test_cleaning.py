import numpy as np
import pandas as pd
import pytest

from dataviz import config
from dataviz.cleaning import (
    CleaningReport,
    clean_sales_data,
    drop_duplicate_orders,
    drop_missing_dates,
    fill_missing,
    finalize_types,
    flag_outliers,
    mark_invalid_values,
    parse_dates,
    remove_invalid_rows,
    remove_outliers,
    standardize_columns,
    standardize_text,
    to_numeric,
)
from dataviz.loader import load_csv
from dataviz.sample_data import CATEGORIES, REGIONS, write_sample_csv


# ---------- single steps ----------


def test_columns_are_standardized():
    table = pd.DataFrame(columns=[" Order Date ", "REGION"])
    assert list(standardize_columns(table).columns) == ["order_date", "region"]


def test_text_is_trimmed_and_given_one_spelling():
    table = pd.DataFrame({"region": ["  north ", "SOUTH", "", None], "category": ["books"] * 4})
    result = standardize_text(table)

    assert result["region"].isna().tolist() == [False, False, True, True]
    assert result["region"].dropna().tolist() == ["North", "South"]
    assert result["category"].tolist() == ["Books"] * 4


def test_dates_are_parsed_and_bad_ones_become_missing():
    table = pd.DataFrame({"order_date": ["2024-03-05", "31/02/2024", "not available", None]})
    result = parse_dates(table)

    assert pd.api.types.is_datetime64_any_dtype(result["order_date"])
    assert result["order_date"].isna().tolist() == [False, True, True, True]
    assert result["order_date"].iloc[0] == pd.Timestamp("2024-03-05")


def test_dates_can_use_another_format():
    table = pd.DataFrame({"order_date": ["05/03/2024"]})
    result = parse_dates(table, date_format="%d/%m/%Y")
    assert result["order_date"].iloc[0] == pd.Timestamp("2024-03-05")


def test_rows_without_a_date_are_dropped():
    table = pd.DataFrame({"order_date": pd.to_datetime(["2024-01-01", None]), "x": [1, 2]})
    result = drop_missing_dates(table)
    assert len(result) == 1
    assert result["x"].tolist() == [1]


def test_numbers_are_read_from_messy_text():
    table = pd.DataFrame({"revenue": ["$1,234.50", "12", "abc", None, "  7 "]})
    result = to_numeric(table, columns=("revenue",))

    assert result["revenue"].dtype == "float64"
    np.testing.assert_allclose(
        result["revenue"].to_numpy(), [1234.5, 12, np.nan, np.nan, 7], equal_nan=True
    )


def test_impossible_values_become_missing():
    table = pd.DataFrame(
        {
            "customer_age": [25.0, 999.0, 0.0, np.nan],
            "customer_rating": [5.0, 6.0, 0.0, 3.0],
            "discount": [0.1, 1.5, -0.2, 0.0],
        }
    )
    result = mark_invalid_values(table)

    assert result["customer_age"].isna().tolist() == [False, True, True, True]
    assert result["customer_rating"].isna().tolist() == [False, True, True, False]
    assert result["discount"].isna().tolist() == [False, True, True, False]


def test_duplicate_orders_keep_the_first_row():
    table = pd.DataFrame({"order_id": ["A", "B", "A"], "units": [1, 2, 99]})
    result = drop_duplicate_orders(table)
    assert result["order_id"].tolist() == ["A", "B"]
    assert result["units"].tolist() == [1, 2]


def test_orders_without_units_or_price_are_removed():
    table = pd.DataFrame({"units": [1.0, 0.0, -1.0, np.nan, 2.0], "unit_price": [5.0, 5.0, 5.0, 5.0, np.nan]})
    result = remove_invalid_rows(table)
    assert len(result) == 1
    assert result["units"].tolist() == [1.0]


def test_missing_values_are_filled_sensibly():
    table = pd.DataFrame(
        {
            "customer_age": [20.0, 30.0, np.nan],
            "customer_rating": [4.0, np.nan, 2.0],
            "marketing_spend": [100.0, np.nan, 300.0],
            "discount": [0.1, np.nan, 0.0],
            "region": pd.array(["North", None, "South"], dtype="string"),
            "category": pd.array(["Books", "Books", None], dtype="string"),
            "units": [2.0, 1.0, 1.0],
            "unit_price": [10.0, 10.0, 10.0],
            "revenue": [np.nan, 10.0, 10.0],
        }
    )
    result = fill_missing(table)

    assert result["customer_age"].tolist() == [20.0, 30.0, 25.0]
    assert result["customer_rating"].tolist() == [4.0, 3.0, 2.0]
    assert result["marketing_spend"].tolist() == [100.0, 200.0, 300.0]
    assert result["discount"].tolist() == [0.1, 0.0, 0.0]
    assert result["region"].tolist() == ["North", "Unknown", "South"]
    assert result["category"].tolist() == ["Books", "Books", "Unknown"]
    assert result["revenue"].iloc[0] == pytest.approx(18.0)   # 2 x 10 x (1 - 0.1)


def test_whole_number_columns_become_integers():
    table = pd.DataFrame(
        {"units": [1.0, 2.0], "customer_age": [30.4, 41.6], "customer_rating": [3.5, 4.0]}
    )
    result = finalize_types(table)
    assert result["units"].dtype.kind == "i"
    assert result["customer_age"].tolist() == [30, 42]
    assert result["customer_rating"].dtype.kind == "i"


# ---------- outliers ----------


def _prices():
    return pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 4,
            "price": [10, 11, 12, 13, 14, 1000, 100, 101, 102, 103],
        }
    )


def test_outliers_are_found_inside_each_group():
    flags = flag_outliers(_prices(), "price", by="group", factor=1.5)
    assert flags.tolist() == [False] * 5 + [True] + [False] * 4


def test_outliers_can_be_found_without_groups():
    flags = flag_outliers(_prices(), "price", factor=1.5)
    assert flags.iloc[5]          # the 1000 stands out
    assert not flags.iloc[0]


def test_outliers_are_removed():
    result = remove_outliers(_prices(), "price", by="group", factor=1.5)
    assert len(result) == 9
    assert 1000 not in result["price"].tolist()


def test_small_groups_are_never_judged():
    table = pd.DataFrame({"group": ["A", "A", "A"], "price": [1, 2, 1000]})
    assert not flag_outliers(table, "price", by="group").any()


# ---------- the whole pipeline ----------


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    path, _ = write_sample_csv(tmp_path_factory.mktemp("data") / "sales.csv", rows=400, seed=5)
    raw = load_csv(path)
    clean, report = clean_sales_data(raw)
    return raw, clean, report


def test_the_clean_table_has_no_missing_values(cleaned):
    _, clean, _ = cleaned
    assert clean[list(config.COLUMNS)].isna().sum().sum() == 0


def test_the_clean_table_has_one_row_per_order(cleaned):
    _, clean, _ = cleaned
    assert clean["order_id"].is_unique


def test_the_clean_table_has_proper_types(cleaned):
    _, clean, _ = cleaned
    assert pd.api.types.is_datetime64_any_dtype(clean["order_date"])
    for column in config.NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(clean[column])
    assert clean["order_date"].is_monotonic_increasing


def test_the_clean_values_are_realistic(cleaned):
    _, clean, _ = cleaned
    assert (clean["units"] > 0).all()
    assert clean["customer_age"].between(18, 75).all()
    assert clean["customer_rating"].between(1, 5).all()
    assert set(clean["category"]) <= set(CATEGORIES)
    assert set(clean["region"]) <= set(REGIONS) | {"Unknown"}


def test_extreme_prices_are_gone(cleaned):
    _, clean, _ = cleaned
    for category, settings in CATEGORIES.items():
        prices = clean.loc[clean["category"] == category, "unit_price"]
        if not prices.empty:
            assert prices.max() <= settings["price"][1]


def test_the_report_adds_up(cleaned):
    raw, clean, report = cleaned
    assert report.rows_in == len(raw)
    assert report.rows_out == len(clean)
    removed = (
        report.invalid_dates_removed
        + report.duplicates_removed
        + report.invalid_rows_removed
        + report.outliers_removed
    )
    assert report.rows_removed == removed
    assert 0 < report.duplicates_removed <= 20
    assert report.missing_values_filled      # the sample data has gaps


def test_the_raw_table_is_left_untouched(cleaned):
    raw, _, _ = cleaned
    assert raw["order_id"].duplicated().any()
    assert raw["revenue"].astype(str).str.startswith("$").any()


def test_the_report_can_be_printed_and_saved():
    report = CleaningReport(rows_in=10, rows_out=7, duplicates_removed=3)
    assert report.as_dict()["rows_removed"] == 3
    assert "Rows in" in report.summary()
    assert "Rows out" in report.summary()