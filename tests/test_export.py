import json

import numpy as np
import pandas as pd
import pytest

from dataviz.export import (
    build_markdown_report,
    dataframe_to_markdown,
    export_clean_data,
    export_json,
    export_markdown_report,
    export_tables,
)


@pytest.fixture
def table():
    return pd.DataFrame(
        {"count": [3, 1_200], "sum": [1234.5, 99.0]},
        index=pd.Index(["Books", "Home"], name="category"),
    )


def test_clean_data_is_saved_with_iso_dates(tmp_path):
    frame = pd.DataFrame({"order_date": pd.to_datetime(["2024-03-05"]), "revenue": [10.5]})
    path = export_clean_data(frame, tmp_path / "deep" / "folder" / "clean.csv")

    saved = pd.read_csv(path)
    assert saved["order_date"].tolist() == ["2024-03-05"]
    assert saved["revenue"].tolist() == [10.5]
    assert list(saved.columns) == ["order_date", "revenue"]       # no index column


def test_every_table_gets_its_own_file(tmp_path, table):
    written = export_tables({"by_category": table, "plain": pd.DataFrame({"a": [1, 2]})}, tmp_path / "tables")

    assert set(written) == {"by_category", "plain"}
    assert all(path.exists() for path in written.values())
    assert list(pd.read_csv(written["by_category"]).columns) == ["category", "count", "sum"]
    assert list(pd.read_csv(written["plain"]).columns) == ["a"]            # plain index is not saved


def test_json_handles_numpy_numbers_and_dates(tmp_path):
    data = {
        "count": np.int64(5),
        "average": np.float64(2.5),
        "flag": np.bool_(True),
        "values": np.array([1, 2]),
        "when": pd.Timestamp("2024-01-31"),
    }
    path = export_json(data, tmp_path / "out" / "data.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == {
        "count": 5, "average": 2.5, "flag": True, "values": [1, 2], "when": "2024-01-31T00:00:00",
    }


def test_markdown_tables_are_formatted(table):
    markdown = dataframe_to_markdown(table)
    lines = markdown.splitlines()

    assert lines[0] == "| category | count | sum |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Books | 3 | 1,234.50 |"
    assert lines[3] == "| Home | 1,200 | 99.00 |"


def test_markdown_tables_can_skip_the_index():
    markdown = dataframe_to_markdown(pd.DataFrame({"a": [1]}), index=False)
    assert markdown.splitlines()[0] == "| a |"


def test_markdown_shows_dates_and_blanks():
    frame = pd.DataFrame({"month": pd.to_datetime(["2024-01-01"]), "value": [np.nan]})
    row = dataframe_to_markdown(frame, index=False).splitlines()[2]
    assert row == "| 2024-01-01 |  |"


def test_the_report_contains_every_section(table):
    report = build_markdown_report(
        metrics={"total_revenue": 1234.5, "top_category": "Home"},
        cleaning={"rows_in": 1220, "missing_values_filled": {"customer_age": 40}, "invalid_values_blanked": {}},
        tables={"revenue_by_category": table},
        figures={"Monthly revenue": "figures/monthly.png"},
    )

    assert report.startswith("# Sales analysis report")
    assert "## Key metrics" in report and "## Data cleaning" in report
    assert "## Charts" in report and "## Tables" in report
    assert "**Total revenue:** 1,234.5" in report
    assert "**Top category:** Home" in report
    assert "customer_age (40)" in report
    assert "**Invalid values blanked:** none" in report
    assert "![Monthly revenue](figures/monthly.png)" in report
    assert "### Revenue by category" in report


def test_long_tables_are_shortened_in_the_report():
    long_table = pd.DataFrame({"x": range(40)})
    report = build_markdown_report({}, {}, {"long": long_table}, {}, max_table_rows=5)
    assert "Showing the first 5 of 40 rows" in report


def test_the_report_links_to_images_relative_to_itself(tmp_path, table):
    images = {"Chart": tmp_path / "figures" / "chart.png"}
    path = export_markdown_report(tmp_path / "report.md", {}, {}, {"t": table}, images)

    text = path.read_text(encoding="utf-8")
    assert "![Chart](figures/chart.png)" in text