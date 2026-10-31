"""Heatmap: a colored table showing how strongly columns move together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dataviz.plots.common import finish_figure, new_figure, pretty_label


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    title: str = "Correlation between columns",
    annotate: bool = True,
    cmap: str = "coolwarm",
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Draw a correlation matrix (see statistics.correlation_matrix) as colored squares.

    Red = the two columns rise together (positive correlation).
    Blue = one rises while the other falls (negative correlation).
    White = no straight-line relationship.
    The diagonal is always dark red: every column is perfectly correlated with itself.
    """
    if corr.empty or corr.shape[0] != corr.shape[1]:
        raise ValueError("The correlation table must be square and not empty")

    values = corr.to_numpy(dtype=float)
    size = len(values)
    labels = [pretty_label(name) for name in corr.columns]

    fig, ax = new_figure(figsize=(max(6.0, 0.9 * size + 3), max(5.0, 0.8 * size + 2.5)))

    # The scale is fixed to -1..+1 so colors always mean the same thing, in every chart
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(False)

    if annotate:
        for row in range(size):
            for column in range(size):
                value = values[row, column]
                if np.isnan(value):
                    continue
                # White text on dark squares, black text on light ones
                ax.text(
                    column, row, f"{value:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.6 else "black",
                )

    fig.colorbar(image, ax=ax, label="Correlation", shrink=0.85)
    ax.set_title(title)
    return finish_figure(fig, save_path=save_path, show=show, close=close)