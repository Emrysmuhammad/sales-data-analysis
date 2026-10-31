"""Save results: cleaned data, tables, numbers as JSON and a readable report."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _prepare(path: str | Path) -> Path:
    """Make sure the folder of a file exists."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def export_clean_data(df: pd.DataFrame, path: str | Path) -> Path:
    """Save the cleaned table as a CSV file (dates as YYYY-MM-DD)."""
    target = _prepare(path)
    df.to_csv(target, index=False, date_format="%Y-%m-%d")
    return target


def export_tables(tables: Mapping[str, pd.DataFrame], directory: str | Path) -> dict[str, Path]:
    """Save every table as <name>.csv inside a folder. Returns the files written."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, table in tables.items():
        target = folder / f"{name}.csv"
        # Keep the index when it carries information (like category names), skip a plain 0, 1, 2...
        table.to_csv(target, index=not isinstance(table.index, pd.RangeIndex))
        written[name] = target
    return written


def _json_default(value: Any) -> Any:
    """Teach json how to write numpy numbers, dates and similar objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def export_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Save a dictionary as a nicely indented JSON file."""
    target = _prepare(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default, ensure_ascii=False)
    return target


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Turn a table into a Markdown table (no extra packages needed)."""
    frame = df.reset_index() if index else df
    headers = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return "" if pd.isna(value) else f"{value:,.2f}"
        return "" if pd.isna(value) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()


def build_markdown_report(
    metrics: Mapping[str, Any],
    cleaning: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, str],
    max_table_rows: int = 15,
) -> str:
    """Write the whole analysis as one readable Markdown document.

    `figures` maps a chart title to the path of its image file, relative to the report.
    """
    lines = [
        "# Sales analysis report",
        "",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Key metrics",
        "",
    ]
    lines += [f"- **{_title(name)}:** {value:,}" if isinstance(value, (int, float)) else f"- **{_title(name)}:** {value}"
              for name, value in metrics.items()]

    lines += ["", "## Data cleaning", ""]
    for name, value in cleaning.items():
        if isinstance(value, dict):
            detail = ", ".join(f"{key} ({count})" for key, count in value.items()) or "none"
            lines.append(f"- **{_title(name)}:** {detail}")
        else:
            lines.append(f"- **{_title(name)}:** {value:,}")

    lines += ["", "## Charts", ""]
    for title, path in figures.items():
        lines += [f"### {title}", "", f"![{title}]({path})", ""]

    lines += ["## Tables", ""]
    for name, table in tables.items():
        shown = table.head(max_table_rows)
        index = not isinstance(table.index, pd.RangeIndex)
        lines += [f"### {_title(name)}", "", dataframe_to_markdown(shown, index=index), ""]
        if len(table) > max_table_rows:
            lines += [f"_Showing the first {max_table_rows} of {len(table)} rows._", ""]

    return "\n".join(lines)


def export_markdown_report(
    path: str | Path,
    metrics: Mapping[str, Any],
    cleaning: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    figure_paths: Mapping[str, str | Path],
) -> Path:
    """Save the Markdown report. Image links are made relative to the report file."""
    target = _prepare(path)
    relative = {
        title: Path(os.path.relpath(Path(image), target.parent)).as_posix()
        for title, image in figure_paths.items()
    }
    target.write_text(build_markdown_report(metrics, cleaning, tables, relative), encoding="utf-8")
    return target