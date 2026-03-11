"""
src/autoresearch/tracker.py

Experiment result logging. Initializes and appends to the results log
defined in the experiment manifest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from autoresearch.manifest import LogConfig, LogFormat


def _separator(fmt: LogFormat) -> str:
    if fmt == LogFormat.tsv:
        return "\t"
    if fmt == LogFormat.csv:
        return ","
    return ""  # jsonl has no tabular separator


def init_log(config: LogConfig, directory: Path | None = None) -> Path:
    """Create the results log file with header row. Returns the log path."""
    path = Path(directory / config.path) if directory else Path(config.path)

    if config.format == LogFormat.jsonl:
        path.touch(exist_ok=True)
        return path

    sep = _separator(config.format)
    header = sep.join(config.columns)
    path.write_text(header + "\n")
    return path


def append_result(
    config: LogConfig,
    values: dict[str, str | float],
    directory: Path | None = None,
) -> None:
    """Append a single result row to the log file."""
    path = Path(directory / config.path) if directory else Path(config.path)

    if config.format == LogFormat.jsonl:
        with path.open("a") as f:
            f.write(json.dumps(values) + "\n")
        return

    sep = _separator(config.format)
    quoting = csv.QUOTE_MINIMAL if config.format == LogFormat.csv else csv.QUOTE_NONE
    row = [str(values.get(col, "")) for col in config.columns]

    with path.open("a", newline="") as f:
        writer = csv.writer(f, delimiter=sep, quoting=quoting)
        writer.writerow(row)
