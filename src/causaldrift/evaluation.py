"""The single public evaluation protocol: ten random half-length windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ADJACENCY, open_dataset, series_matrix, standardize_window
from .models import ModelConfig, create_model, normalize_models


METRICS = ("auroc", "auprc", "norm_shd", "norm_whd")
TRUTH_SUPPORT_THRESHOLD = 1e-8


@dataclass(frozen=True)
class EvaluationWindow:
    index: int
    start: int
    stop: int


def random_windows(
    num_timesteps: int,
    *,
    count: int = 10,
    window_size: int | None = None,
    seed: int = 0,
) -> list[EvaluationWindow]:
    """Sample fixed-size random windows."""

    if num_timesteps < 2:
        raise ValueError("At least two time steps are required for random-window evaluation")
    if count < 1:
        raise ValueError("count must be positive")
    size = max(1, min(num_timesteps, num_timesteps // 2 if window_size is None else int(window_size)))
    starts = np.random.default_rng(seed).integers(0, num_timesteps - size + 1, size=count)
    return [EvaluationWindow(index=index, start=int(start), stop=int(start) + size) for index, start in enumerate(starts)]


def structural_hamming_distance(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float = 0.5,
    truth_support_threshold: float = TRUTH_SUPPORT_THRESHOLD,
    include_diagonal: bool = False,
) -> float:
    """Directed SHD; a reversal costs one edit."""

    predicted = (np.asarray(prediction) >= threshold).astype(int)
    actual = (np.asarray(truth) > truth_support_threshold).astype(int)
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("prediction and truth must be same-shaped 2-D matrices")
    edits = 0
    nodes = predicted.shape[0]
    if include_diagonal:
        edits += int(np.abs(np.diag(predicted) - np.diag(actual)).sum())
    for left in range(nodes):
        for right in range(left + 1, nodes):
            p_lr, p_rl = predicted[left, right], predicted[right, left]
            t_lr, t_rl = actual[left, right], actual[right, left]
            if (p_lr, p_rl) == (t_lr, t_rl):
                continue
            if (p_lr, p_rl, t_lr, t_rl) in {(1, 0, 0, 1), (0, 1, 1, 0)}:
                edits += 1
            else:
                edits += abs(p_lr - t_lr) + abs(p_rl - t_rl)
    return float(edits)


def weighted_hamming_distance(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    include_diagonal: bool = False,
) -> float:
    """Continuous absolute edge-strength error used by normalized WHD."""

    predicted = np.nan_to_num(np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0), nan=0.0)
    actual = np.nan_to_num(np.clip(np.asarray(truth, dtype=float), 0.0, 1.0), nan=0.0)
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("prediction and truth must be same-shaped 2-D matrices")
    if not include_diagonal:
        predicted = predicted.copy()
        actual = actual.copy()
        np.fill_diagonal(predicted, 0.0)
        np.fill_diagonal(actual, 0.0)
    return float(np.abs(predicted - actual).sum())


def score_graph(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float = 0.5,
    truth_support_threshold: float = TRUTH_SUPPORT_THRESHOLD,
    include_diagonal: bool = False,
) -> dict[str, float]:
    """Compute AUROC, AUPRC, normalized SHD, and normalized WHD."""

    prediction = np.nan_to_num(np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0), nan=0.0)
    truth = np.nan_to_num(np.clip(np.asarray(truth, dtype=float), 0.0, 1.0), nan=0.0)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be same-shaped 2-D matrices")
    if not include_diagonal:
        prediction = prediction.copy()
        truth = truth.copy()
        np.fill_diagonal(prediction, 0.0)
        np.fill_diagonal(truth, 0.0)
        mask = ~np.eye(prediction.shape[0], dtype=bool)
    else:
        mask = np.ones(prediction.shape, dtype=bool)
    possible = max(1, int(mask.sum()))
    labels = (truth[mask] > truth_support_threshold).astype(int)
    scores = prediction[mask]
    shd = structural_hamming_distance(
        prediction,
        truth,
        threshold=threshold,
        truth_support_threshold=truth_support_threshold,
        include_diagonal=include_diagonal,
    )
    whd = weighted_hamming_distance(prediction, truth, include_diagonal=include_diagonal)
    result = {
        "norm_shd": shd / possible,
        "norm_whd": whd / possible,
        "shd": shd,
        "weighted_hamming": whd,
        "possible_edges": float(possible),
        "truth_support_threshold": float(truth_support_threshold),
    }
    if len(np.unique(labels)) < 2:
        result["auroc"] = np.nan
        result["auprc"] = np.nan
    else:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["auprc"] = float(average_precision_score(labels, scores))
    return result


def _prediction_dataset(
    dataset: xr.Dataset,
    models: tuple[str, ...],
    windows: list[EvaluationWindow],
    predictions: np.ndarray,
    truth: np.ndarray,
) -> xr.Dataset:
    model_specs = [create_model(model).spec for model in models]
    names = np.asarray(dataset.coords["node"].to_numpy(), dtype=str)
    return xr.Dataset(
        data_vars={
            "prediction": (("model", "window", "source", "target"), predictions.astype(np.float32)),
            "truth": (("window", "source", "target"), truth.astype(np.float32)),
            "truth_support": (("window", "source", "target"), (truth > TRUTH_SUPPORT_THRESHOLD).astype(np.int8)),
        },
        coords={
            "model": np.asarray(models, dtype=str),
            "model_label": ("model", [spec.label for spec in model_specs]),
            "output_kind": ("model", [spec.output_kind for spec in model_specs]),
            "threshold": ("model", [spec.threshold for spec in model_specs]),
            "window": np.arange(len(windows)),
            "start": ("window", [window.start for window in windows]),
            "stop": ("window", [window.stop for window in windows]),
            "truth_time": ("window", [window.stop - 1 for window in windows]),
            "source": names,
            "target": names,
        },
        attrs={
            "protocol": "10 random fixed windows; ground-truth adjacency is taken at each window endpoint (stop - 1)",
            "include_diagonal": 0,
            "truth_support_threshold": TRUTH_SUPPORT_THRESHOLD,
            "truth_semantics": "AUROC/AUPRC/SHD use positive directed-edge support; WHD uses continuous edge strength",
            **{key: value for key, value in dataset.attrs.items() if isinstance(value, (str, int, float, np.integer, np.floating))},
        },
    )


def _metadata(dataset: xr.Dataset, dataset_name: str) -> dict[str, object]:
    keep = ("dataset_family", "dataset_type", "setting", "scenario", "drift_type", "temporal_pattern", "seed")
    return {"dataset": dataset_name, **{key: dataset.attrs.get(key, "") for key in keep}}


def evaluate_dataset(
    dataset: xr.Dataset,
    *,
    dataset_name: str = "dataset",
    models: tuple[str, ...] | list[str] | str = "all",
    num_windows: int = 10,
    window_size: int | None = None,
    seed: int = 0,
    model_config: ModelConfig | None = None,
    continue_on_error: bool = False,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Evaluate requested models on a single sample under the fixed protocol."""

    requested = normalize_models(models)
    if ADJACENCY not in dataset:
        raise KeyError(f"Dataset has no {ADJACENCY!r} variable")
    series = series_matrix(dataset)
    adjacency = np.asarray(dataset[ADJACENCY].to_numpy(), dtype=np.float32)
    if adjacency.shape != (series.shape[0], series.shape[1], series.shape[1]):
        raise ValueError("adjacency must be [time, node, node] and match time_series")
    windows = random_windows(series.shape[0], count=num_windows, window_size=window_size, seed=seed)
    inputs = np.stack([standardize_window(series[window.start : window.stop]) for window in windows])
    truth = np.stack([adjacency[window.stop - 1] for window in windows])
    predictions = np.full((len(requested), len(windows), series.shape[1], series.shape[1]), np.nan, dtype=np.float32)
    records: list[dict[str, object]] = []
    config = model_config or ModelConfig(seed=seed)
    metadata = _metadata(dataset, dataset_name)

    for model_index, model_name in enumerate(requested):
        model = create_model(model_name, config)
        warning = ""
        try:
            values = np.asarray(model.predict_many(inputs), dtype=np.float32)
            expected = (len(windows), series.shape[1], series.shape[1])
            if values.shape != expected:
                raise ValueError(f"{model_name} returned {values.shape}, expected {expected}")
            predictions[model_index] = np.clip(values, 0.0, 1.0)
        except Exception as error:
            if not continue_on_error:
                raise RuntimeError(f"{model.spec.label} failed on {dataset_name}: {error}") from error
            warning = f"{type(error).__name__}: {error}"
        for window_index, window in enumerate(windows):
            row: dict[str, object] = {
                **metadata,
                "model": model_name,
                "model_label": model.spec.label,
                "output_kind": model.spec.output_kind,
                "prediction_threshold": model.spec.threshold,
                "truth_support_threshold": TRUTH_SUPPORT_THRESHOLD,
                "window": window.index,
                "start": window.start,
                "stop": window.stop,
                "warning": warning,
            }
            if warning:
                row.update({metric: np.nan for metric in METRICS})
            else:
                row.update(score_graph(predictions[model_index, window_index], truth[window_index], threshold=model.spec.threshold))
            records.append(row)
    return _prediction_dataset(dataset, requested, windows, predictions, truth), pd.DataFrame.from_records(records)


def _stable_seed(base_seed: int, relative_path: Path) -> int:
    return int(base_seed + zlib.crc32(relative_path.as_posix().encode("utf-8")))


def summarize_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-dataset 10-window mean/std and model-level aggregates."""

    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["dataset", "model", "model_label", "dataset_family", "dataset_type", "setting", "scenario", "drift_type", "temporal_pattern"]
    grouped = metrics.groupby(keys, dropna=False, sort=False)
    rows: list[dict[str, object]] = []
    for values, group in grouped:
        row = dict(zip(keys, values))
        row["num_windows"] = int(len(group))
        row["successful_windows"] = int(group["warning"].astype(str).eq("").sum())
        for metric in METRICS:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(numeric.mean()) if numeric.notna().any() else np.nan
            row[f"{metric}_std"] = float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    overall_rows: list[dict[str, object]] = []
    for (model, label), group in summary.groupby(["model", "model_label"], sort=False):
        row = {"model": model, "model_label": label, "dataset_count": int(len(group))}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(pd.to_numeric(group[f"{metric}_mean"], errors="coerce").mean())
            row[f"{metric}_std"] = float(pd.to_numeric(group[f"{metric}_std"], errors="coerce").mean())
        overall_rows.append(row)
    return summary, pd.DataFrame.from_records(overall_rows)


def evaluate_directory(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    models: tuple[str, ...] | list[str] | str = "all",
    num_windows: int = 10,
    window_size: int | None = None,
    seed: int = 202_606,
    third_party_dir: str | Path | None = None,
    device: str | None = None,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate NetCDF files and write predictions and metric CSVs."""

    source = Path(data_dir)
    destination = Path(out_dir)
    files = sorted(source.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {source}")
    destination.mkdir(parents=True, exist_ok=True)
    prediction_dir = destination / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    all_records: list[pd.DataFrame] = []
    requested = normalize_models(models)
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source)
        name = relative.with_suffix("").as_posix()
        print(f"[{index}/{len(files)}] {name}", flush=True)
        dataset = open_dataset(path)
        dataset_seed = _stable_seed(seed, relative)
        config = ModelConfig(
            third_party_dir=None if third_party_dir is None else Path(third_party_dir),
            device=device,
            seed=dataset_seed,
        )
        prediction, metrics = evaluate_dataset(
            dataset,
            dataset_name=name,
            models=requested,
            num_windows=num_windows,
            window_size=window_size,
            seed=dataset_seed,
            model_config=config,
            continue_on_error=continue_on_error,
        )
        prediction_path = prediction_dir / relative.with_suffix(".nc")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_netcdf(prediction_path)
        all_records.append(metrics)
    raw = pd.concat(all_records, ignore_index=True)
    summary, overall = summarize_metrics(raw)
    raw.to_csv(destination / "metrics.csv", index=False)
    summary.to_csv(destination / "summary_metrics.csv", index=False)
    overall.to_csv(destination / "overall_metrics.csv", index=False)
    return raw, summary, overall
