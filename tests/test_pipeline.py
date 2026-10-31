import json

import pandas as pd
import pytest

from dataviz.loader import DataLoadError
from dataviz.pipeline import main, run_analysis
from dataviz.sample_data import write_sample_csv


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    folder = tmp_path_factory.mktemp("analysis")
    source, _ = write_sample_csv(folder / "raw.csv", rows=400, seed=11)
    result = run_analysis(source, folder / "results")
    return folder / "results", result


def test_every_chart_is_created(finished):
    _, result = finished
    assert len(result["figures"]) == 7
    for path in result["figures"].values():
        assert path.exists() and path.stat().st_size > 0


def test_every_file_is_created(finished):
    folder, result = finished
    for path in result["files"].values():
        assert path.exists() and path.stat().st_size > 0
    assert (folder / "clean_sales.csv").exists()
    assert (folder / "summary.json").exists()
    assert (folder / "report.md").exists()


def test_the_exported_clean_data_is_really_clean(finished):
    folder, _ = finished
    saved = pd.read_csv(folder / "clean_sales.csv", parse_dates=["order_date"])

    assert saved["order_id"].is_unique
    assert saved.isna().sum().sum() == 0
    assert (saved["units"] > 0).all()


def test_the_summary_file_matches_the_returned_metrics(finished):
    folder, result = finished
    saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert saved == result["metrics"]
    assert saved["orders"] > 0


def test_the_report_links_to_the_charts(finished):
    folder, _ = finished
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "![Monthly revenue](figures/monthly_revenue.png)" in report
    assert "![Correlation heatmap](figures/correlation_heatmap.png)" in report
    assert "## Key metrics" in report


def test_the_cleaning_report_is_exported(finished):
    folder, result = finished
    saved = json.loads((folder / "cleaning_report.json").read_text(encoding="utf-8"))
    assert saved["rows_in"] == result["cleaning"].rows_in
    assert saved["rows_removed"] == result["cleaning"].rows_removed


def test_a_missing_input_file_raises_a_clear_error(tmp_path):
    with pytest.raises(DataLoadError):
        run_analysis(tmp_path / "missing.csv", tmp_path / "results")


def test_the_command_line_runs_and_reports_errors(tmp_path, capsys):
    source, _ = write_sample_csv(tmp_path / "raw.csv", rows=150, seed=12)

    assert main(["--input", str(source), "--output", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert "Key metrics" in printed
    assert "Files written" in printed

    assert main(["--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "out2")]) == 1
    assert "Error" in capsys.readouterr().err