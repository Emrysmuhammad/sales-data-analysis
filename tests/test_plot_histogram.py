import numpy as np
import pandas as pd
import pytest

from dataviz.plots.histogram import plot_histogram, plot_histogram_grid


@pytest.fixture
def ages():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(35, 10, 500))


def test_the_requested_number_of_bins_is_drawn(ages):
    ax = plot_histogram(ages, bins=10, close=True).axes[0]
    assert len(ax.patches) == 10


def test_every_value_is_counted_exactly_once(ages):
    ax = plot_histogram(ages, bins=12, close=True).axes[0]
    assert sum(patch.get_height() for patch in ax.patches) == 500


def test_missing_values_are_ignored(ages):
    with_gaps = pd.concat([ages, pd.Series([np.nan] * 25)], ignore_index=True)
    ax = plot_histogram(with_gaps, bins=10, close=True).axes[0]
    assert sum(patch.get_height() for patch in ax.patches) == 500


def test_mean_and_median_lines_are_drawn(ages):
    ax = plot_histogram(ages, close=True).axes[0]
    assert len(ax.lines) == 2

    positions = sorted(line.get_xdata()[0] for line in ax.lines)
    assert positions[0] == pytest.approx(min(ages.mean(), ages.median()))
    assert positions[1] == pytest.approx(max(ages.mean(), ages.median()))

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any(label.startswith("Mean") for label in labels)
    assert any(label.startswith("Median") for label in labels)


def test_the_statistics_lines_can_be_turned_off(ages):
    ax = plot_histogram(ages, show_stats=False, close=True).axes[0]
    assert ax.lines == []
    assert ax.get_legend() is None


def test_titles_and_labels(ages):
    ax = plot_histogram(ages, title="Ages", xlabel="Years", ylabel="People", close=True).axes[0]
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("Ages", "Years", "People")


def test_the_histogram_is_saved(ages, tmp_path):
    target = tmp_path / "plots" / "hist.png"
    plot_histogram(ages, save_path=target, close=True)
    assert target.exists() and target.stat().st_size > 0


def test_empty_data_is_rejected():
    with pytest.raises(ValueError):
        plot_histogram(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        plot_histogram(pd.Series([np.nan, np.nan]))


# ---------- the grid of histograms ----------


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "customer_age": rng.normal(35, 10, 200),
            "unit_price": rng.gamma(2, 50, 200),
            "revenue": rng.gamma(2, 80, 200),
        }
    )


def test_the_grid_has_one_visible_chart_per_column(frame):
    fig = plot_histogram_grid(frame, ["customer_age", "unit_price", "revenue"], ncols=2, close=True)
    assert len(fig.axes) == 4
    assert sum(ax.get_visible() for ax in fig.axes) == 3


def test_the_grid_titles_are_readable(frame):
    fig = plot_histogram_grid(frame, ["customer_age", "revenue"], close=True)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert titles == ["Customer Age", "Revenue"]


def test_unknown_columns_are_skipped_but_one_is_needed(frame):
    fig = plot_histogram_grid(frame, ["revenue", "nope"], close=True)
    assert sum(ax.get_visible() for ax in fig.axes) == 1
    with pytest.raises(ValueError):
        plot_histogram_grid(frame, ["nope"])


def test_the_grid_is_saved(frame, tmp_path):
    target = tmp_path / "grid.png"
    plot_histogram_grid(frame, ["customer_age", "revenue"], save_path=target, close=True)
    assert target.exists()