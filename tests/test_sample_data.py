import re

import numpy as np
import pandas as pd
import pytest

from dataviz import config
from dataviz.sample_data import (
    CATEGORIES,
    generate_clean_orders,
    make_messy,
    write_sample_csv,
)


def test_clean_orders_have_the_right_shape():
    orders = generate_clean_orders(rows=300, seed=1)
    assert len(orders) == 300
    assert list(orders.columns) == list(config.COLUMNS)
    assert orders["order_id"].is_unique


def test_clean_orders_stay_within_realistic_ranges():
    orders = generate_clean_orders(rows=500, seed=2)
    assert orders["customer_age"].between(18, 75).all()
    assert orders["customer_rating"].between(1, 5).all()
    assert (orders["units"] >= 1).all()
    assert set(orders["category"]) <= set(CATEGORIES)

    for category, settings in CATEGORIES.items():
        prices = orders.loc[orders["category"] == category, "unit_price"]
        if not prices.empty:
            low, high = settings["price"]
            assert prices.between(low, high).all()


def test_revenue_follows_units_price_and_discount():
    orders = generate_clean_orders(rows=200, seed=3)
    expected = orders["units"] * orders["unit_price"] * (1 - orders["discount"])
    assert np.allclose(orders["revenue"], expected, atol=0.01)


def test_november_and_december_are_the_busiest_months():
    orders = generate_clean_orders(rows=1200, seed=4)
    share = orders["order_date"].dt.month.isin([11, 12]).mean()
    assert share > 0.22  # a perfectly even year would give about 0.17


def test_same_seed_gives_the_same_data():
    first = generate_clean_orders(rows=100, seed=5)
    second = generate_clean_orders(rows=100, seed=5)
    other = generate_clean_orders(rows=100, seed=6)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_zero_rows_is_rejected():
    with pytest.raises(ValueError):
        generate_clean_orders(rows=0)


def test_messy_data_contains_every_kind_of_problem():
    clean = generate_clean_orders(rows=300, seed=7)
    messy, problems = make_messy(clean, seed=7)

    assert len(messy) == 300 + problems["duplicate_rows"]
    assert all(count >= 1 for count in problems.values())

    assert messy["order_id"].duplicated().sum() == problems["duplicate_rows"]
    assert messy["customer_age"].isna().any()
    assert messy["region"].isna().any()
    assert messy["revenue"].astype(str).str.startswith("$").any()
    assert (messy["customer_age"] == 999).any()
    assert set(messy["category"].dropna()) - set(CATEGORIES)  # odd spellings exist

    iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert not messy["order_date"].astype(str).str.match(iso).all()  # broken dates exist


def test_messy_data_is_repeatable():
    clean = generate_clean_orders(rows=150, seed=8)
    first, _ = make_messy(clean, seed=8)
    second, _ = make_messy(clean, seed=8)
    pd.testing.assert_frame_equal(first, second)


def test_the_csv_file_is_written(tmp_path):
    path, problems = write_sample_csv(tmp_path / "data" / "sales.csv", rows=120, seed=9)

    assert path.exists()
    saved = pd.read_csv(path, dtype=str)
    assert list(saved.columns) == list(config.COLUMNS)
    assert len(saved) == 120 + problems["duplicate_rows"]