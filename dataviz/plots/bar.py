"""Bar charts: compare totals between groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dataviz import config
from dataviz.plots.common import finish_figure, new_figure, number_formatter


def plot_bar(
    values: pd.Series,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    sort: bool = True,
    top_n: int | None = None,
    value_format: str = "${:,.0f}",
    show_labels: bool = True,
    color: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """One bar per group.

    values        a Series whose index holds the group names
    horizontal    draw horizontal bars (best for long names); the biggest ends up on top
    sort          order the bars from biggest to smallest
    top_n         only draw the n biggest groups
    value_format  how the numbers on the bars and on the axis are written
    """
    data = values.dropna()
    if data.empty:
        raise ValueError("There is no data to plot")

    if top_n is not None:
        data = data.nlargest(top_n)
    if sort:
        # Horizontal bars are drawn bottom-to-top, so they need the opposite order
        data = data.sort_values(ascending=horizontal)

    names = [str(name) for name in data.index]
    fig, ax = new_figure()
    fill = color or config.PALETTE[0]

    if horizontal:
        bars = ax.barh(names, data.to_numpy(), color=fill)
        ax.xaxis.set_major_formatter(number_formatter(value_format))
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.14)
    else:
        bars = ax.bar(names, data.to_numpy(), color=fill)
        ax.yaxis.set_major_formatter(number_formatter(value_format))
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.14)

    if show_labels:
        ax.bar_label(bars, labels=[value_format.format(v) for v in data.to_numpy()], padding=3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return finish_figure(fig, save_path=save_path, show=show, close=close)


def plot_grouped_bar(
    table: pd.DataFrame,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    value_format: str = "${:,.0f}",
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Side-by-side bars: one cluster per row of the table, one bar per column.

    For example, a table of revenue with categories as rows and regions as
    columns shows how every region did inside every category.
    """
    if table.empty:
        raise ValueError("There is no data to plot")

    fig, ax = new_figure()

    series_count = len(table.columns)
    width = 0.8 / series_count
    positions = np.arange(len(table.index))

    for index, column in enumerate(table.columns):
        offset = (index - (series_count - 1) / 2) * width
        ax.bar(positions + offset, table[column].to_numpy(dtype=float), width, label=str(column))

    ax.set_xticks(positions)
    ax.set_xticklabels([str(name) for name in table.index])
    ax.yaxis.set_major_formatter(number_formatter(value_format))
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=str(table.columns.name or ""))
    return finish_figure(fig, save_path=save_path, show=show, close=close)