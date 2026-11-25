"""Data loading utilities for Box, Box Formula 1 race analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_AUS_GP_DIR = Path("data/2026/Australian Grand Prix")


def resolve_race_dir(base_dir: Path | str = DEFAULT_AUS_GP_DIR) -> Path:
    """Resolve and validate the local directory containing race data files.

    Args:
        base_dir: Path to the local race directory (defaults to the
            2026 Australian Grand Prix folder inside ``data/2026``).

    Returns:
        Resolved ``Path`` object pointing to an existing directory.

    Raises:
        FileNotFoundError: If ``base_dir`` does not exist.
        NotADirectoryError: If ``base_dir`` exists but is not a directory.
    """
    path = Path(base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Race data directory not found at '{path}'. "
            "Run the sparse-checkout command in README.md to populate data/2026."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory at '{path}', got a file.")
    return path


def list_race_files(
    base_dir: Path | str = DEFAULT_AUS_GP_DIR,
    pattern: str = "*.json",
) -> list[Path]:
    """List matching data files recursively inside the race directory.

    Args:
        base_dir: Root directory of the local race snapshot.
        pattern: Glob pattern used to filter files (e.g., ``"*.json"`` or ``"*.csv"``).

    Returns:
        Sorted list of matching file ``Path`` objects.
    """
    race_dir = resolve_race_dir(base_dir)
    return sorted(p for p in race_dir.rglob(pattern) if p.is_file())


def load_json_file(file_path: Path | str) -> Any:
    """Load and parse a JSON file safely with clear error context.

    Args:
        file_path: Path to the target ``.json`` file.

    Returns:
        Parsed JSON content (typically a ``dict`` or ``list``).

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the file cannot be decoded as UTF-8 or valid JSON.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: '{path}'")

    try:
        raw_text = path.read_text(encoding="utf-8")
        return json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON from '{path}': {exc}") from exc


def load_json_as_dataframe(file_path: Path | str) -> pd.DataFrame:
    """Load a JSON file and normalize it into a pandas ``DataFrame``.

    Handles top-level lists of records, dicts containing a list under common
    keys (such as ``"data"`` or ``"laps"``), or single-object records.

    Args:
        file_path: Path to the target ``.json`` file.

    Returns:
        A pandas ``DataFrame`` containing the normalized records.

    Raises:
        ValueError: If the JSON structure cannot be converted into a table.
    """
    payload = load_json_file(file_path)

    if isinstance(payload, list):
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        for key in ("data", "results", "laps", "records", "items"):
            if key in payload and isinstance(payload[key], list):
                return pd.json_normalize(payload[key])
        return pd.json_normalize([payload])

    raise ValueError(
        f"Unsupported JSON root type '{type(payload).__name__}' in '{file_path}'."
    )


def load_csv_file(file_path: Path | str) -> pd.DataFrame:
    """Load a CSV file into a pandas ``DataFrame`` with validation.

    Args:
        file_path: Path to the target ``.csv`` file.

    Returns:
        Loaded pandas ``DataFrame``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the CSV is empty or malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: '{path}'")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV file '{path}' contains no rows.")
    return df


if __name__ == "__main__":
    try:
        files = list_race_files()
        print(f"Found {len(files)} JSON files under '{DEFAULT_AUS_GP_DIR}':")
        for f in files[:10]:
            print(f" - {f.relative_to(DEFAULT_AUS_GP_DIR)}")
    except FileNotFoundError as err:
        print(f"Loader check: {err}")