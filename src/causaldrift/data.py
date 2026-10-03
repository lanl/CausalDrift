"""Small, explicit NetCDF data contract used by every benchmark component."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


SERIES = "time_series"
ADJACENCY = "adjacency"


def make_dataset(
    series: np.ndarray,
    adjacency: np.ndarray,
    *,
    node_names: list[str] | None = None,
    alpha: np.ndarray | None = None,
    attrs: dict[str, object] | None = None,
    extra_variables: dict[str, tuple[tuple[str, ...], np.ndarray]] | None = None,
) -> xr.Dataset:
    """Create a benchmark sample.

    ``series`` is ``[time, node]`` (or ``[time, node, feature]``), and
    ``adjacency`` is a time-varying, source-to-target score in ``[0, 1]``.
    """

    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError(f"series must have shape [time, node, feature], got {values.shape}")

    truth = np.asarray(adjacency, dtype=np.float32)
    if truth.ndim == 2:
        truth = np.repeat(truth[None, ...], values.shape[0], axis=0)
    expected = (values.shape[0], values.shape[1], values.shape[1])
    if truth.shape != expected:
        raise ValueError(f"adjacency must have shape {expected}, got {truth.shape}")

    names = (
        np.asarray([f"X{i}" for i in range(values.shape[1])], dtype=str)
        if node_names is None
        else np.asarray(node_names, dtype=str)
    )
    if len(names) != values.shape[1]:
        raise ValueError("node_names must have one item per node")

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        SERIES: (("time", "node", "feature"), values),
        ADJACENCY: (("time", "source", "target"), np.clip(truth, 0.0, 1.0)),
    }
    if alpha is not None:
        alpha_values = np.asarray(alpha, dtype=np.float32)
        if alpha_values.shape != (values.shape[0],):
            raise ValueError("alpha must have one value per time step")
        data_vars["alpha"] = (("time",), alpha_values)
    if extra_variables:
        data_vars.update(extra_variables)

    safe_attrs = {
        key: value
        for key, value in (attrs or {}).items()
        if value is not None and isinstance(value, (str, int, float, np.integer, np.floating))
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.arange(values.shape[0]),
            "node": names,
            "source": names,
            "target": names,
            "feature": np.arange(values.shape[2]),
        },
        attrs=safe_attrs,
    )


def save_dataset(dataset: xr.Dataset, path: str | Path) -> Path:
    """Save one benchmark sample atomically enough for normal local runs."""

    target = Path(path)
    if target.suffix != ".nc":
        raise ValueError(f"Dataset path must end in .nc, got {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(target)
    return target


def open_dataset(path: str | Path) -> xr.Dataset:
    """Load a sample into memory so callers can close files immediately."""

    target = Path(path)
    if target.suffix != ".nc":
        raise ValueError(f"Dataset path must end in .nc, got {target}")
    with xr.open_dataset(target) as dataset:
        return dataset.load()


def series_matrix(dataset: xr.Dataset, variable: str = SERIES) -> np.ndarray:
    """Return the raw ``[time, node]`` series."""

    if variable not in dataset:
        raise KeyError(f"Dataset has no {variable!r} variable")
    values = np.asarray(dataset[variable].to_numpy(), dtype=np.float32)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{variable} must be [time, node] or [time, node, feature]")
    return values.astype(np.float32)


def standardize_window(values: np.ndarray) -> np.ndarray:
    """Impute and z-score one already selected model-input window columnwise."""

    window = np.asarray(values, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError(f"values must be [time, node], got {window.shape}")
    finite = np.isfinite(window)
    counts = finite.sum(axis=0, keepdims=True)
    sums = np.where(finite, window, 0.0).sum(axis=0, keepdims=True)
    column_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    if not finite.all():
        window = window.copy()
        missing_rows, missing_columns = np.where(~finite)
        window[missing_rows, missing_columns] = column_means[0, missing_columns]
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    return ((window - mean) / np.where(std > 1e-6, std, 1.0)).astype(np.float32)
