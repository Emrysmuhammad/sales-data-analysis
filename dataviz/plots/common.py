"""Helpers shared by all the charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from dataviz import config


def new_figure(figsize: tuple[float, float] | None = None, **kwargs):
    """Apply the project style and create a figure. Extra arguments go to plt.subplots()."""
    config.apply_plot_style()
    return plt.subplots(figsize=figsize, **kwargs)


def finish_figure(
    fig: Figure,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Tidy the layout, then optionally save, show and close the figure."""
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def number_formatter(pattern: str = "${:,.0f}") -> FuncFormatter:
    """An axis formatter such as $1,200."""
    return FuncFormatter(lambda value, _position: pattern.format(value))


def pretty_label(name: str) -> str:
    """'customer_age' -> 'Customer Age'."""
    return str(name).replace("_", " ").title()