"""Line chart: how something changes over time."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

from dataviz import config
from dataviz.plots.common import finish_figure, new_figure, number_formatter


def plot_monthly_revenue(
    monthly: pd.Series,
    *,
    title: str = "Monthly revenue",
    ylabel: str = "Revenue ($)",
    rolling_window: int | None = 3,
    highlight_peak: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Draw revenue per month as a line.

    monthly         a Series indexed by month (see statistics.revenue_by_month)
    rolling_window  also draw a dashed moving average over this many months
                    (it smooths out month-to-month noise); None or 1 turns it off
    highlight_peak  circle and label the best month
    """
    if monthly.empty:
        raise ValueError("There is no data to plot")

    fig, ax = new_figure()
    ax.plot(monthly.index, monthly.to_numpy(), marker="o", linewidth=2, label="Monthly revenue")

    if rolling_window and rolling_window > 1 and len(monthly) >= rolling_window:
        trend = monthly.rolling(rolling_window, min_periods=rolling_window).mean()
        ax.plot(
            trend.index,
            trend.to_numpy(),
            linestyle="--",
            linewidth=2,
            label=f"{rolling_window}-month average",
        )

    if highlight_peak:
        peak_month = monthly.idxmax()
        peak_value = float(monthly.max())
        ax.scatter([peak_month], [peak_value], s=140, color=config.PALETTE[4], zorder=5)
        ax.annotate(
            f"Peak: ${peak_value:,.0f}",
            xy=(peak_month, peak_value),
            xytext=(-15, 18),
            textcoords="offset points",
            ha="right",
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": config.PALETTE[4]},
        )

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(number_formatter("${:,.0f}"))
    ax.margins(y=0.15)

    if isinstance(monthly.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()

    ax.legend()
    return finish_figure(fig, save_path=save_path, show=show, close=close)