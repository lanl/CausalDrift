"""Synthetic CausalDrift generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import make_dataset, save_dataset
from .graphs import (
    DRIFT_TYPES,
    TEMPORAL_PATTERNS,
    absent_edge,
    drift_curve,
    existing_edge,
    root_mask,
    sample_dag,
    schedule_metadata,
)


@dataclass(frozen=True)
class DriftConfig:
    """One synthetic sample, with fixed defaults rather than a tuning grid."""

    drift_type: str = "edge"
    temporal_pattern: str = "abrupt_appearance"
    num_timesteps: int = 1_000
    num_nodes: int = 5
    seed: int = 0
    extra_edge_probability: float = 0.10
    observation_noise: float = 0.05
    max_lag: int = 20

    def validate(self) -> None:
        if self.drift_type not in DRIFT_TYPES:
            raise ValueError(f"Unknown drift type {self.drift_type!r}; choose {DRIFT_TYPES}")
        if self.temporal_pattern not in TEMPORAL_PATTERNS:
            raise ValueError(
                f"Unknown temporal pattern {self.temporal_pattern!r}; choose {TEMPORAL_PATTERNS}"
            )
        if self.num_timesteps < 20:
            raise ValueError("num_timesteps must be at least 20")
        if self.num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        if not 0.0 <= self.extra_edge_probability <= 1.0:
            raise ValueError("extra_edge_probability must be in [0, 1]")
        if self.observation_noise < 0.0:
            raise ValueError("observation_noise must be non-negative")
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")


def _root_drivers(
    num_timesteps: int,
    roots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate nonlinear root-node signals."""

    num_nodes = len(roots)
    time = np.linspace(0.0, 1.0, num_timesteps, dtype=np.float32)
    values = np.zeros((num_timesteps, num_nodes), dtype=np.float32)
    for node in np.where(roots)[0]:
        frequency = rng.uniform(1.0, 5.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        driver = 0.8 * np.sin(2.0 * np.pi * frequency * time + phase)
        driver += 0.3 * np.sin(2.0 * np.pi * (frequency * 0.37) * time + 2.0 * phase)
        innovation = rng.normal(0.0, 0.10, size=num_timesteps)
        for index in range(1, num_timesteps):
            innovation[index] += 0.65 * innovation[index - 1]
        values[:, node] = driver + innovation
    return values


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return ((values - mean) / np.where(std > 1e-6, std, 1.0)).astype(np.float32)


def _select_node_with_children(adjacency: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.where(np.asarray(adjacency).sum(axis=1) > 0)[0]
    if len(candidates) == 0:
        return int(adjacency.shape[0] - 1)
    return int(rng.choice(candidates))


def _initial_graph(config: DriftConfig, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reserve an initially absent node for appearance-style node drift."""

    if config.drift_type == "node" and alpha[0] < 0.5:
        base = sample_dag(config.num_nodes - 1, rng, config.extra_edge_probability)
        graph = np.zeros((config.num_nodes, config.num_nodes), dtype=np.float32)
        graph[:-1, :-1] = base
        return graph
    return sample_dag(config.num_nodes, rng, config.extra_edge_probability)


def _apply_drift(
    config: DriftConfig,
    base_graph: np.ndarray,
    alpha: np.ndarray,
    roots: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    """Build graph, lag, observation, and mechanism schedules for one sample."""

    length, nodes = len(alpha), base_graph.shape[0]
    adjacency = np.repeat(base_graph[None, ...], length, axis=0).astype(np.float32)
    weights = adjacency.copy()
    # Static SCM edges are contemporaneous; only the explicit ``lag`` drift
    # changes this matrix.  High-to-low graph order keeps zero-lag simulation
    # acyclic.
    lag = np.zeros((length, nodes, nodes), dtype=np.int16)
    observed = np.ones((length, nodes), dtype=np.float32)
    mechanism = np.zeros((length, nodes, nodes), dtype=np.float32)
    noise = np.full((length, nodes), config.observation_noise, dtype=np.float32)
    latent_gate = np.zeros(length, dtype=np.float32)
    metadata = {
        "drift_source": -1,
        "drift_target": -1,
        "drift_node": -1,
        "drift_lag": 0,
        "latent_confounder_target_1": -1,
        "latent_confounder_target_2": -1,
    }

    if config.drift_type == "edge":
        if alpha[0] < 0.5:
            source, target = absent_edge(base_graph, rng)
        else:
            source, target = existing_edge(base_graph, rng)
        adjacency[:, source, target] = alpha
        weights[:, source, target] = alpha
        metadata.update(drift_source=source, drift_target=target)

    elif config.drift_type == "strength":
        source, target = existing_edge(base_graph, rng)
        strength = 0.25 + 0.75 * alpha
        adjacency[:, source, target] = strength
        weights[:, source, target] = strength
        metadata.update(drift_source=source, drift_target=target)

    elif config.drift_type == "mechanism":
        source, target = existing_edge(base_graph, rng)
        mechanism[:, source, target] = alpha
        metadata.update(drift_source=source, drift_target=target)

    elif config.drift_type == "node":
        if alpha[0] < 0.5:
            node = nodes - 1
            targets = np.where(~roots[:node])[0]
            if len(targets) == 0:
                targets = np.arange(node)
            target = int(rng.choice(targets))
            adjacency[:, node, target] = alpha
            weights[:, node, target] = alpha
        else:
            node = _select_node_with_children(base_graph, rng)
            target = int(np.where(base_graph[node] > 0)[0][0])
            adjacency[:, node, :] *= alpha[:, None]
            weights[:, node, :] *= alpha[:, None]
        observed[:, node] = (alpha > 1e-6).astype(np.float32)
        metadata.update(drift_source=node, drift_target=target, drift_node=node)

    elif config.drift_type == "driver":
        node = int(rng.choice(np.where(roots)[0]))
        metadata.update(drift_node=node)

    elif config.drift_type == "confounder":
        candidates = np.where(~roots)[0]
        if len(candidates) < 2:
            candidates = np.arange(nodes)
        source, target = (int(item) for item in rng.choice(candidates, size=2, replace=False))
        latent_gate = alpha.copy()
        # Record the selected observed targets.
        metadata.update(
            latent_confounder_target_1=source,
            latent_confounder_target_2=target,
        )

    elif config.drift_type == "lag":
        source, target = existing_edge(base_graph, rng)
        max_lag = int(rng.integers(2, config.max_lag + 1))
        lag[:, source, target] = np.rint(alpha * max_lag).astype(np.int16)
        metadata.update(drift_source=source, drift_target=target, drift_lag=max_lag)

    elif config.drift_type == "noise":
        node = int(rng.integers(nodes))
        noise[:, node] = config.observation_noise * (0.5 + 3.0 * alpha)
        metadata.update(drift_node=node)

    return adjacency, weights, lag, observed, mechanism, noise, latent_gate, metadata


def _simulate(
    *,
    adjacency: np.ndarray,
    weights: np.ndarray,
    lag: np.ndarray,
    roots: np.ndarray,
    observed: np.ndarray,
    mechanism: np.ndarray,
    noise: np.ndarray,
    latent_gate: np.ndarray,
    config: DriftConfig,
    alpha: np.ndarray,
    metadata: dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a nonlinear SCM in the DAG's high-to-low order."""

    length, nodes = observed.shape
    result = np.zeros((length, nodes), dtype=np.float32)
    drivers = _root_drivers(length, roots, rng)
    coefficients = rng.uniform(0.25, 0.85, size=(nodes, nodes)).astype(np.float32)
    coefficients *= rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(nodes, nodes))
    alternate_drivers = _root_drivers(length, roots, rng)
    latent = np.sin(np.linspace(0.0, 12.0 * np.pi, length, dtype=np.float32))
    confounded_targets = np.asarray(
        [metadata["latent_confounder_target_1"], metadata["latent_confounder_target_2"]], dtype=int
    )

    for time_index in range(length):
        for target in range(nodes - 1, -1, -1):
            active_parents = np.where(adjacency[time_index, :, target] > 0.0)[0]
            if roots[target] and len(active_parents) == 0:
                value = drivers[time_index, target]
                if config.drift_type == "driver" and target == metadata["drift_node"]:
                    value = (1.0 - alpha[time_index]) * value + alpha[time_index] * alternate_drivers[time_index, target]
                result[time_index, target] = value
                continue

            value = 0.0
            for source in active_parents:
                lag_steps = int(lag[time_index, source, target])
                source_value = result[max(0, time_index - lag_steps), source]
                primary = np.tanh(source_value)
                alternate = np.sin(source_value)
                blend = mechanism[time_index, source, target]
                value += coefficients[source, target] * weights[time_index, source, target] * (
                    (1.0 - blend) * primary + blend * alternate
                )
            if time_index > 0:
                value += 0.15 * result[time_index - 1, target]
            result[time_index, target] = value

        if config.drift_type == "confounder":
            result[time_index, confounded_targets] += 0.7 * latent_gate[time_index] * latent[time_index]

    result += rng.normal(0.0, noise, size=result.shape).astype(np.float32)
    result = _standardize(result)
    result[observed == 0.0] = np.nan
    return result


def generate_synthetic_dataset(config: DriftConfig) -> xr.Dataset:
    """Generate one independent synthetic CausalDrift sample."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    alpha = drift_curve(config.num_timesteps, config.temporal_pattern)
    base_graph = _initial_graph(config, alpha, rng)
    roots = root_mask(base_graph)
    adjacency, weights, lag, observed, mechanism, noise, latent_gate, metadata = _apply_drift(
        config, base_graph, alpha, roots, rng
    )
    series = _simulate(
        adjacency=adjacency,
        weights=weights,
        lag=lag,
        roots=roots,
        observed=observed,
        mechanism=mechanism,
        noise=noise,
        latent_gate=latent_gate,
        config=config,
        alpha=alpha,
        metadata=metadata,
        rng=rng,
    )
    attrs: dict[str, object] = {
        "benchmark": "CausalDrift",
        "dataset_family": "synthetic",
        "drift_type": config.drift_type,
        "temporal_pattern": config.temporal_pattern,
        "seed": config.seed,
        "num_nodes": config.num_nodes,
        "max_lag": config.max_lag,
        "extra_edge_probability": config.extra_edge_probability,
        "truth_scope": "observed_directed_edges",
        "drift_semantics": (
            "latent_confounding_without_directed_edge_change"
            if config.drift_type == "confounder"
            else "directed_edge_strength_or_data_mechanism_change"
        ),
        **metadata,
        **schedule_metadata(config.num_timesteps, config.temporal_pattern),
    }
    return make_dataset(
        series,
        adjacency,
        alpha=alpha,
        attrs=attrs,
        extra_variables={
            "edge_strength": (("time", "source", "target"), weights),
            "lag_matrix": (("time", "source", "target"), lag),
            "observed_mask": (("time", "node"), observed),
            "mechanism_blend": (("time", "source", "target"), mechanism),
            "noise_sigma": (("time", "node"), noise),
            "root_mask": (("node",), roots.astype(np.int8)),
        },
    )


def generate_synthetic_directory(
    out_dir: str | Path,
    *,
    instances: int = 10,
    num_timesteps: int = 1_000,
    num_nodes: int = 5,
    seed: int = 202_600,
    drift_types: tuple[str, ...] = DRIFT_TYPES,
    temporal_patterns: tuple[str, ...] = TEMPORAL_PATTERNS,
) -> pd.DataFrame:
    """Write synthetic benchmark files."""

    destination = Path(out_dir)
    records: list[dict[str, object]] = []
    counter = 0
    for drift_type in drift_types:
        for temporal_pattern in temporal_patterns:
            for instance in range(instances):
                sample_seed = seed + counter
                config = DriftConfig(
                    drift_type=drift_type,
                    temporal_pattern=temporal_pattern,
                    num_timesteps=num_timesteps,
                    num_nodes=num_nodes,
                    seed=sample_seed,
                )
                filename = f"{drift_type}_{temporal_pattern}_seed{sample_seed}.nc"
                path = destination / drift_type / temporal_pattern / filename
                save_dataset(generate_synthetic_dataset(config), path)
                records.append(
                    {
                        "path": path.relative_to(destination).as_posix(),
                        "drift_type": drift_type,
                        "temporal_pattern": temporal_pattern,
                        "instance": instance,
                        "seed": sample_seed,
                        "num_timesteps": num_timesteps,
                        "num_nodes": num_nodes,
                    }
                )
                counter += 1
    manifest = pd.DataFrame.from_records(records)
    destination.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination / "manifest.csv", index=False)
    return manifest
