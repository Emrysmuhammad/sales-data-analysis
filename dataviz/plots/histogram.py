"""Histograms: how the values of one column are spread out."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
from matplotlib.figure import Figure

from dataviz import config
from dataviz.plots.common import finish_figure, new_figure, pretty_label


def _draw_histogram(ax, data: pd.Series, *, bins, color: str, show_stats: bool) -> None:
    """Draw the bars and, if wanted, vertical lines for the mean and the median."""
    ax.hist(data.to_numpy(), bins=bins, color=color, edgecolor="white", alpha=0.9)

    if show_stats:
        mean = float(data.mean())
        median = float(data.median())
        ax.axvline(mean, color=config.PALETTE[1], linestyle="--", linewidth=2, label=f"Mean {mean:,.1f}")
        ax.axvline(median, color=config.PALETTE[2], linestyle=":", linewidth=2, label=f"Median {median:,.1f}")
        ax.legend()


def plot_histogram(
    values: pd.Series,
    *,
    bins: int | str | Sequence[float] = "auto",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Number of orders",
    show_stats: bool = True,
    color: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Count how many values fall into each range ("bin").

    A histogram shows the SHAPE of the data: where most values are, how wide the
    spread is, and whether a few extreme values stretch one side. When the mean
    is far from the median, the data is lopsided (skewed).
    """
    data = values.dropna()
    if data.empty:
        raise ValueError("There is no data to plot")

    fig, ax = new_figure()
    _draw_histogram(ax, data, bins=bins, color=color or config.PALETTE[0], show_stats=show_stats)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return finish_figure(fig, save_path=save_path, show=show, close=close)


def plot_histogram_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    bins: int | str = "auto",
    ncols: int = 2,
    show_stats: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Several histograms side by side, one per column."""
    present = [column for column in columns if column in df.columns]
    if not present:
        raise ValueError("None of the requested columns exist in the table")

    nrows = math.ceil(len(present) / ncols)
    fig, axes = new_figure(figsize=(5.5 * ncols, 3.8 * nrows), nrows=nrows, ncols=ncols, squeeze=False)
    flat = axes.ravel()

    for index, (ax, column) in enumerate(zip(flat, present)):
        data = df[column].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        _draw_histogram(
            ax, data, bins=bins, color=config.PALETTE[index % len(config.PALETTE)], show_stats=show_stats
        )
        ax.set_title(pretty_label(column))
        ax.set_xlabel(pretty_label(column))
        ax.set_ylabel("Number of orders")

    for ax in flat[len(present):]:      # hide the unused squares of the grid
        ax.set_visible(False)

    return finish_figure(fig, save_path=save_path, show=show, close=close)