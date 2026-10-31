import numpy as np
import pandas as pd
import pytest

from dataviz.plots.heatmap import plot_correlation_heatmap


@pytest.fixture
def corr():
    return pd.DataFrame(
        [[1.0, 0.8, -0.5], [0.8, 1.0, 0.1], [-0.5, 0.1, 1.0]],
        index=["unit_price", "revenue", "customer_age"],
        columns=["unit_price", "revenue", "customer_age"],
    )


def test_the_colors_come_from_the_correlation_values(corr):
    ax = plot_correlation_heatmap(corr, close=True).axes[0]
    drawn = np.ma.getdata(ax.images[0].get_array()).astype(float)
    np.testing.assert_allclose(drawn, corr.to_numpy())


def test_the_color_scale_is_fixed_from_minus_one_to_one(corr):
    image = plot_correlation_heatmap(corr, close=True).axes[0].images[0]
    assert image.get_clim() == (-1, 1)


def test_there_is_a_colorbar(corr):
    fig = plot_correlation_heatmap(corr, close=True)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Correlation"


def test_every_cell_is_annotated_with_its_value(corr):
    ax = plot_correlation_heatmap(corr, close=True).axes[0]
    texts = [text.get_text() for text in ax.texts]
    assert len(texts) == 9
    assert texts.count("1.00") == 3
    assert "-0.50" in texts and "0.80" in texts


def test_annotations_can_be_turned_off(corr):
    ax = plot_correlation_heatmap(corr, annotate=False, close=True).axes[0]
    assert ax.texts == []


def test_labels_are_readable_and_in_order(corr):
    fig = plot_correlation_heatmap(corr, close=True)
    fig.canvas.draw()
    ax = fig.axes[0]
    expected = ["Unit Price", "Revenue", "Customer Age"]
    assert [label.get_text() for label in ax.get_xticklabels()] == expected
    assert [label.get_text() for label in ax.get_yticklabels()] == expected


def test_empty_values_are_left_blank():
    corr = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    ax = plot_correlation_heatmap(corr, close=True).axes[0]
    assert len(ax.texts) == 2


def test_the_title_can_be_changed(corr):
    ax = plot_correlation_heatmap(corr, title="Relationships", close=True).axes[0]
    assert ax.get_title() == "Relationships"


def test_the_heatmap_is_saved(corr, tmp_path):
    target = tmp_path / "plots" / "heatmap.png"
    plot_correlation_heatmap(corr, save_path=target, close=True)
    assert target.exists() and target.stat().st_size > 0


def test_a_table_that_is_not_square_is_rejected():
    with pytest.raises(ValueError):
        plot_correlation_heatmap(pd.DataFrame(np.ones((2, 3))))
    with pytest.raises(ValueError):
        plot_correlation_heatmap(pd.DataFrame())