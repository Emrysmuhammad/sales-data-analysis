import pandas as pd
import pytest

from dataviz import config
from dataviz.loader import DataLoadError, describe_load, load_csv, normalize_column_name
from dataviz.sample_data import write_sample_csv


def test_the_sample_file_loads_with_every_column_as_text(tmp_path):
    path, problems = write_sample_csv(tmp_path / "sales.csv", rows=200, seed=3)
    table = load_csv(path)

    assert len(table) == 200 + problems["duplicate_rows"]
    assert set(config.COLUMNS) <= set(table.columns)
    assert all(pd.api.types.is_string_dtype(table[column]) for column in table.columns)


def test_dollar_values_are_kept_as_written(tmp_path):
    path, _ = write_sample_csv(tmp_path / "sales.csv", rows=300, seed=4)
    table = load_csv(path)
    assert table["revenue"].astype(str).str.startswith("$").any()


def test_a_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_csv(tmp_path / "nothing.csv")


def test_a_file_that_is_not_a_csv_is_rejected(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    with pytest.raises(DataLoadError, match="csv"):
        load_csv(notes)


def test_an_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError, match="empty"):
        load_csv(empty)


def test_missing_columns_are_listed(tmp_path):
    partial = tmp_path / "partial.csv"
    partial.write_text("order_id,region\nA1,North\n")
    with pytest.raises(DataLoadError) as info:
        load_csv(partial)
    assert "order_date" in str(info.value)
    assert "revenue" in str(info.value)


def test_column_names_are_normalized(tmp_path):
    headers = ",".join(name.replace("_", " ").title() for name in config.COLUMNS)
    row = ",".join(["x"] * len(config.COLUMNS))
    path = tmp_path / "titles.csv"
    path.write_text(f"{headers}\n{row}\n")

    table = load_csv(path)
    assert list(table.columns) == list(config.COLUMNS)


def test_normalize_column_name():
    assert normalize_column_name("  Order Date ") == "order_date"
    assert normalize_column_name("REVENUE") == "revenue"


def test_describe_load_counts_problems():
    table = pd.DataFrame(
        {"a": ["1", "2", "2", None], "b": ["x", "y", "y", "z"]}
    )
    summary = describe_load(table)

    assert summary.rows == 4
    assert summary.columns == 2
    assert summary.missing_cells == 1
    assert summary.duplicate_rows == 1
    assert summary.memory_kb > 0
    assert "Rows:" in str(summary)