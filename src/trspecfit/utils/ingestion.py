"""Utilities for ingesting simple XY spectra into trspecfit projects.

These helpers support the first step of a non-time-resolved workflow:
discovering .xy files, parsing them into numeric arrays, and grouping them by
metadata such as core level and depth.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | Path


def _match_metadata_pattern(path: Path, pattern: str | re.Pattern[str]) -> re.Match[str] | None:
    """Match a metadata pattern against both the full filename and the stem."""

    for candidate in (path.name, path.stem):
        match = re.match(pattern, candidate)
        if match is not None:
            return match
    return None


def parse_xy_spectrum(
    filepath: PathLike,
    *,
    delimiter: str | None = None,
    x_column: int = 0,
    y_column: int = 1,
    skip_rows: int = 0,
) -> dict[str, np.ndarray]:
    """Parse a simple two-column .xy spectrum file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .xy file.
    delimiter : str, optional
        Column delimiter used by the file. If omitted, whitespace splitting is
        used.
    x_column : int, default=0
        Column index for the x values.
    y_column : int, default=1
        Column index for the y values.
    skip_rows : int, default=0
        Number of leading rows to skip before parsing data.

    Returns
    -------
    dict
        Mapping with ``x`` and ``y`` arrays.
    """

    path = Path(filepath)
    values: list[tuple[float, float]] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle):
            if line_number < skip_rows:
                continue
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "ColumnLabels:" in line:
                continue
            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) <= max(x_column, y_column):
                continue
            try:
                x_value = float(parts[x_column])
                y_value = float(parts[y_column])
            except ValueError:
                continue
            values.append((x_value, y_value))

    if not values:
        raise ValueError(f"No numeric data found in {path}")

    x_values = np.array([value[0] for value in values], dtype=float)
    y_values = np.array([value[1] for value in values], dtype=float)
    return {"x": x_values, "y": y_values}


def discover_xy_files(
    path: PathLike,
    *,
    metadata_pattern: str | re.Pattern[str] | None = None,
    metadata_names: tuple[str, str] = ("core_level", "depth"),
) -> list[Path]:
    """Find .xy files under ``path`` and return them in sorted order."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    paths = sorted(root.rglob("*.xy"))
    if metadata_pattern is None:
        return paths

    for candidate in paths:
        match = _match_metadata_pattern(candidate, metadata_pattern)
        if match is None:
            raise ValueError(
                f"Filename {candidate.name} does not match metadata pattern "
                f"{metadata_pattern}."
            )

    return paths


def group_xy_files(
    paths: list[PathLike],
    *,
    metadata_pattern: str | re.Pattern[str] | None = None,
    metadata_names: tuple[str, str] = ("core_level", "depth"),
) -> dict[tuple[str, str], list[Path]]:
    """Group .xy paths by metadata extracted from the filename stem."""

    regex = None
    if metadata_pattern is not None:
        regex = re.compile(metadata_pattern)

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        if regex is None:
            key = (path.stem, "")
        else:
            match = _match_metadata_pattern(path, regex.pattern)
            if match is None:
                raise ValueError(
                    f"Filename {path.name} does not match metadata pattern "
                    f"{metadata_pattern}."
                )
            metadata = match.groupdict()
            if len(metadata_names) != 2:
                raise ValueError("metadata_names must contain exactly two entries")
            key = (
                str(metadata.get(metadata_names[0], path.stem)),
                str(metadata.get(metadata_names[1], "")),
            )
        grouped[key].append(path)

    return {key: sorted(value) for key, value in grouped.items()}


def load_xy_project(
    path: PathLike,
    *,
    project_path: PathLike | None = None,
    project_name: str = "xy_project",
    metadata_pattern: str | re.Pattern[str] | None = None,
    metadata_names: tuple[str, str] = ("core_level", "depth"),
    delimiter: str | None = None,
    x_column: int = 0,
    y_column: int = 1,
    skip_rows: int = 0,
) -> tuple[Any, dict[tuple[str, str], Any]]:
    """Load grouped .xy files into a trspecfit Project and one File per group.

    Each resulting File contains the spectra from all .xy files that share the
    same extracted metadata values (e.g. one core level and one depth).
    """

    from trspecfit.trspecfit import File, Project

    data_dir = Path(path)
    if project_path is None:
        project_path = data_dir
    project = Project(path=project_path, name=project_name)
    paths = discover_xy_files(
        data_dir,
        metadata_pattern=metadata_pattern,
        metadata_names=metadata_names,
    )
    grouped = group_xy_files(
        paths,
        metadata_pattern=metadata_pattern,
        metadata_names=metadata_names,
    )

    files: dict[tuple[str, str], File] = {}
    for (core_level, depth), group_paths in grouped.items():
        spectra = [
            parse_xy_spectrum(
                group_path,
                delimiter=delimiter,
                x_column=x_column,
                y_column=y_column,
                skip_rows=skip_rows,
            )
            for group_path in group_paths
        ]
        x_axis = spectra[0]["x"]
        y_arrays = [spectrum["y"] for spectrum in spectra]
        if len(y_arrays) == 1:
            data = np.asarray(y_arrays[0], dtype=float)
            time_axis = np.array([0.0])
        else:
            data = np.vstack(y_arrays)
            time_axis = np.arange(len(y_arrays), dtype=float)

        file_name = f"{core_level}_{depth}".replace(" ", "_")
        files[(core_level, depth)] = File(
            parent_project=project,
            path=file_name,
            data=data,
            energy=x_axis,
            time=time_axis,
        )

    return project, files


__all__ = [
    "discover_xy_files",
    "group_xy_files",
    "load_xy_project",
    "parse_xy_spectrum",
]
