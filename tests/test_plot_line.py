import numpy as np
import pandas as pd
import pytest

from dataviz.plots.line import plot_monthly_revenue


@pytest.fixture
def monthly():
    return pd.Series(
        [100.0, 200.0, 150.0, 400.0, 300.0, 250.0],
        index=pd.date_range("2024-01-01", periods=6, freq="MS"),
        name="revenue",
    )


def test_the_data_is_drawn_as_a_line(monthly):
    fig = plot_monthly_revenue(monthly, close=True)
    ax = fig.axes[0]

    np.testing.assert_allclose(ax.lines[0].get_ydata(), monthly.to_numpy())
    assert ax.get_title() == "Monthly revenue"
    assert ax.get_ylabel() == "Revenue ($)"


def test_a_moving_average_is_added_by_default(monthly):
    ax = plot_monthly_revenue(monthly, close=True).axes[0]
    assert len(ax.lines) == 2
    assert "3-month average" in [text.get_text() for text in ax.get_legend().get_texts()]


def test_the_moving_average_can_be_turned_off(monthly):
    assert len(plot_monthly_revenue(monthly, rolling_window=None, close=True).axes[0].lines) == 1
    assert len(plot_monthly_revenue(monthly, rolling_window=1, close=True).axes[0].lines) == 1


def test_a_window_longer_than_the_data_is_skipped(monthly):
    assert len(plot_monthly_revenue(monthly, rolling_window=12, close=True).axes[0].lines) == 1


def test_the_peak_month_is_labelled(monthly):
    ax = plot_monthly_revenue(monthly, close=True).axes[0]
    labels = [text.get_text() for text in ax.texts]
    assert "Peak: $400" in labels


def test_the_peak_label_can_be_turned_off(monthly):
    ax = plot_monthly_revenue(monthly, highlight_peak=False, close=True).axes[0]
    assert ax.texts == []


def test_the_title_can_be_changed(monthly):
    ax = plot_monthly_revenue(monthly, title="Sales 2024", close=True).axes[0]
    assert ax.get_title() == "Sales 2024"


def test_the_chart_is_saved_to_a_file(monthly, tmp_path):
    target = tmp_path / "charts" / "line.png"
    plot_monthly_revenue(monthly, save_path=target, close=True)
    assert target.exists()
    assert target.stat().st_size > 0


def test_empty_data_is_rejected():
    with pytest.raises(ValueError):
        plot_monthly_revenue(pd.Series(dtype=float))