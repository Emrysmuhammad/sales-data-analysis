"""Load the raw sales CSV file into a pandas DataFrame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataviz import config


class DataLoadError(Exception):
    """The data file is missing, empty or does not look like sales data."""


@dataclass(frozen=True)
class LoadSummary:
    """A quick health check of a freshly loaded table."""

    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    memory_kb: float

    def __str__(self) -> str:
        return (
            f"Rows:            {self.rows:,}\n"
            f"Columns:         {self.columns}\n"
            f"Missing cells:   {self.missing_cells:,}\n"
            f"Duplicate rows:  {self.duplicate_rows:,}\n"
            f"Memory:          {self.memory_kb:,.1f} KB"
        )


def normalize_column_name(name: object) -> str:
    """' Order Date ' -> 'order_date'."""
    return str(name).strip().lower().replace(" ", "_")


def load_csv(
    path: str | Path | None = None,
    required_columns: Iterable[str] = config.COLUMNS,
) -> pd.DataFrame:
    """Read a sales CSV file.

    Every column is read as TEXT on purpose. Real files contain surprises such as
    "$1,234.50" in a number column, and reading them as text means nothing is
    silently changed or lost. The cleaning step converts the types properly.

    Raises DataLoadError with a clear message when something is wrong.
    """
    file_path = Path(path) if path else config.RAW_DATA_PATH

    if not file_path.exists():
        raise DataLoadError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise DataLoadError(f"Expected a .csv file, got: {file_path.name}")

    try:
        table = pd.read_csv(file_path, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise DataLoadError(f"The file is empty: {file_path}") from error
    except pd.errors.ParserError as error:
        raise DataLoadError(f"The file could not be read as CSV: {error}") from error

    table.columns = [normalize_column_name(column) for column in table.columns]

    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise DataLoadError("Missing required columns: " + ", ".join(missing))

    return table


def describe_load(table: pd.DataFrame) -> LoadSummary:
    """Summarize a loaded table: size, missing cells and duplicate rows."""
    return LoadSummary(
        rows=len(table),
        columns=table.shape[1],
        missing_cells=int(table.isna().sum().sum()),
        duplicate_rows=int(table.duplicated().sum()),
        memory_kb=float(table.memory_usage(deep=True).sum() / 1024),
    )