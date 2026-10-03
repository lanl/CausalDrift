"""Graph and temporal-drift primitives shared by the two generators."""

from __future__ import annotations

import numpy as np


DRIFT_TYPES = (
    "edge",
    "mechanism",
    "strength",
    "node",
    "driver",
    "confounder",
    "lag",
    "noise",
)
TEMPORAL_PATTERNS = (
    "abrupt_appearance",
    "abrupt_disappearance",
    "gradual_increase",
    "gradual_decrease",
)


def drift_curve(num_timesteps: int, pattern: str) -> np.ndarray:
    """Return the benchmark's [0, 1] abrupt or gradual drift schedule."""

    if pattern not in TEMPORAL_PATTERNS:
        raise ValueError(f"Unknown temporal pattern: {pattern}")
    if num_timesteps < 3:
        raise ValueError("num_timesteps must be at least 3")
    alpha = np.zeros(num_timesteps, dtype=np.float32)
    if pattern == "abrupt_appearance":
        alpha[num_timesteps // 2 :] = 1.0
    elif pattern == "abrupt_disappearance":
        alpha[: num_timesteps // 2] = 1.0
    elif pattern == "gradual_increase":
        start, end = num_timesteps // 3, 2 * num_timesteps // 3
        alpha[end:] = 1.0
        alpha[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False)
    else:  # gradual_decrease
        start, end = num_timesteps // 3, 2 * num_timesteps // 3
        alpha[:start] = 1.0
        alpha[start:end] = np.linspace(1.0, 0.0, end - start, endpoint=False)
    return alpha


def schedule_metadata(num_timesteps: int, pattern: str) -> dict[str, int]:
    """Expose change boundaries in NetCDF attributes without model-side hints."""

    if pattern.startswith("abrupt"):
        return {
            "change_point": num_timesteps // 2,
            "ramp_start": -1,
            "ramp_end": -1,
        }
    return {
        "change_point": -1,
        "ramp_start": num_timesteps // 3,
        "ramp_end": 2 * num_timesteps // 3,
    }


def sample_dag(
    num_nodes: int,
    rng: np.random.Generator,
    extra_edge_probability: float = 0.10,
) -> np.ndarray:
    """Sample a small source-to-target DAG with high-to-low topological order."""

    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    attractiveness = np.ones(num_nodes, dtype=float)
    for source in range(1, num_nodes):
        choices = np.arange(source)
        target = int(rng.choice(choices, p=attractiveness[choices] / attractiveness[choices].sum()))
        adjacency[source, target] = 1.0
        attractiveness[target] += 1.0
        for extra_target in choices:
            if extra_target != target and rng.random() < extra_edge_probability:
                adjacency[source, extra_target] = 1.0
    return adjacency


def root_mask(adjacency: np.ndarray) -> np.ndarray:
    return np.asarray(adjacency).sum(axis=0) == 0


def existing_edge(adjacency: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    candidates = np.argwhere(np.asarray(adjacency) > 0)
    if len(candidates) == 0:
        raise ValueError("Graph has no edge")
    source, target = candidates[int(rng.integers(len(candidates)))]
    return int(source), int(target)


def absent_edge(adjacency: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    candidates = [
        (source, target)
        for source in range(adjacency.shape[0])
        for target in range(source)
        if adjacency[source, target] == 0
    ]
    if not candidates:
        # A fully saturated DAG is rare for the defaults; remove one edge to
        # preserve a valid appearance-drift task rather than silently failing.
        source, target = existing_edge(adjacency, rng)
        adjacency[source, target] = 0.0
        return source, target
    return candidates[int(rng.integers(len(candidates)))]
