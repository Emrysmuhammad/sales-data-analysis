import numpy as np
import pandas as pd
import pytest

from dataviz.plots.bar import plot_bar, plot_grouped_bar


@pytest.fixture
def totals():
    return pd.Series({"Books": 100.0, "Home": 300.0, "Sports": 200.0})


def heights(ax):
    return [patch.get_height() for patch in ax.patches]


def widths(ax):
    return [patch.get_width() for patch in ax.patches]


def test_bars_are_sorted_from_biggest_to_smallest(totals):
    ax = plot_bar(totals, close=True).axes[0]
    assert heights(ax) == [300.0, 200.0, 100.0]


def test_sorting_can_be_turned_off(totals):
    ax = plot_bar(totals, sort=False, close=True).axes[0]
    assert heights(ax) == [100.0, 300.0, 200.0]


def test_horizontal_bars_put_the_biggest_on_top(totals):
    ax = plot_bar(totals, horizontal=True, close=True).axes[0]
    assert widths(ax) == [100.0, 200.0, 300.0]    # drawn bottom to top


def test_only_the_biggest_groups_can_be_shown(totals):
    ax = plot_bar(totals, top_n=2, close=True).axes[0]
    assert heights(ax) == [300.0, 200.0]


def test_every_bar_gets_a_value_label(totals):
    ax = plot_bar(totals, close=True).axes[0]
    assert {text.get_text() for text in ax.texts} == {"$100", "$200", "$300"}


def test_labels_can_be_turned_off_or_reformatted(totals):
    assert plot_bar(totals, show_labels=False, close=True).axes[0].texts == []
    ax = plot_bar(totals, value_format="{:.1f} pts", close=True).axes[0]
    assert "300.0 pts" in {text.get_text() for text in ax.texts}


def test_titles_and_labels(totals):
    ax = plot_bar(totals, title="Revenue", xlabel="Category", ylabel="Dollars", close=True).axes[0]
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Revenue", "Category", "Dollars")


def test_missing_values_are_ignored():
    series = pd.Series({"A": 5.0, "B": np.nan, "C": 3.0})
    assert len(plot_bar(series, close=True).axes[0].patches) == 2


def test_the_bar_chart_is_saved(totals, tmp_path):
    target = tmp_path / "out" / "bars.png"
    plot_bar(totals, save_path=target, close=True)
    assert target.exists() and target.stat().st_size > 0


def test_an_empty_bar_chart_is_rejected():
    with pytest.raises(ValueError):
        plot_bar(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        plot_bar(pd.Series({"A": np.nan}))


# ---------- grouped bars ----------


@pytest.fixture
def table():
    frame = pd.DataFrame(
        {"North": [10.0, 20.0, 30.0], "South": [15.0, 25.0, 35.0]},
        index=["Books", "Home", "Sports"],
    )
    frame.columns.name = "region"
    return frame


def test_grouped_bars_draw_one_bar_per_cell(table):
    ax = plot_grouped_bar(table, close=True).axes[0]
    assert len(ax.patches) == 6
    assert len(ax.containers) == 2


def test_grouped_bars_have_the_right_heights(table):
    ax = plot_grouped_bar(table, close=True).axes[0]
    assert [patch.get_height() for patch in ax.containers[0]] == [10.0, 20.0, 30.0]
    assert [patch.get_height() for patch in ax.containers[1]] == [15.0, 25.0, 35.0]


def test_grouped_bars_have_a_legend_and_names(table):
    ax = plot_grouped_bar(table, title="Revenue", close=True).axes[0]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["North", "South"]
    assert ax.get_legend().get_title().get_text() == "region"
    assert ax.get_title() == "Revenue"


def test_grouped_bars_are_saved(table, tmp_path):
    target = tmp_path / "grouped.png"
    plot_grouped_bar(table, save_path=target, close=True)
    assert target.exists()


def test_empty_grouped_bars_are_rejected():
    with pytest.raises(ValueError):
        plot_grouped_bar(pd.DataFrame())