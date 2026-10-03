"""Pseudo-realistic CausalDrift generators: ENSO-inspired, SEIR, and SEIHRDV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import make_dataset, save_dataset
from .graphs import drift_curve, schedule_metadata


PSEUDO_TYPES = ("enso", "seir", "seihrdv")


@dataclass(frozen=True)
class PseudoScenario:
    key: str
    family: str
    drift_type: str
    temporal_pattern: str
    description: str
    source: str | None = None
    target: str | None = None


ENSO_SCENARIOS = (
    PseudoScenario(
        "edge_wwv_to_nino34_increase",
        "enso",
        "edge",
        "gradual_increase",
        "Gradual appearance of the warm-water-volume influence on Nino34.",
        "WWV",
        "Nino34",
    ),
    PseudoScenario(
        "edge_wwv_to_nino34_decrease",
        "enso",
        "edge",
        "gradual_decrease",
        "Gradual disappearance of the warm-water-volume influence on Nino34.",
        "WWV",
        "Nino34",
    ),
    PseudoScenario(
        "strength_iod_to_nino34_increase",
        "enso",
        "strength",
        "gradual_increase",
        "Gradual strengthening of the Indian-Ocean teleconnection.",
        "IOD",
        "Nino34",
    ),
    PseudoScenario(
        "strength_iod_to_nino34_decrease",
        "enso",
        "strength",
        "gradual_decrease",
        "Gradual weakening of the Indian-Ocean teleconnection.",
        "IOD",
        "Nino34",
    ),
    PseudoScenario(
        "noise_nino34_increase",
        "enso",
        "noise",
        "gradual_increase",
        "Increasing stochastic forcing around Nino34.",
    ),
    PseudoScenario(
        "noise_nino34_decrease",
        "enso",
        "noise",
        "gradual_decrease",
        "Decreasing stochastic forcing around Nino34.",
    ),
)

SEIR_SCENARIOS = (
    PseudoScenario(
        "abrupt_lockdown",
        "seir",
        "graph",
        "abrupt_disappearance",
        "Abrupt reduction of transmission under a lockdown.",
        "I",
        "S",
    ),
    PseudoScenario(
        "gradual_variant_spread",
        "seir",
        "graph",
        "gradual_increase",
        "Gradual increase of transmission during variant spread.",
        "I",
        "S",
    ),
    PseudoScenario(
        "abrupt_reopening",
        "seir",
        "graph",
        "abrupt_appearance",
        "Abrupt increase of transmission after reopening.",
        "I",
        "S",
    ),
    PseudoScenario(
        "gradual_behavior_change",
        "seir",
        "graph",
        "gradual_decrease",
        "Gradual decline in transmission from behavioral change.",
        "I",
        "S",
    ),
)

SEIHRDV_SCENARIOS = (
    PseudoScenario(
        "vaccination_rollout",
        "seihrdv",
        "graph",
        "gradual_increase",
        "Gradual vaccination rollout increases S-to-V and reduces breakthrough infection.",
        "S",
        "V",
    ),
    PseudoScenario(
        "contact_restrictions",
        "seihrdv",
        "graph",
        "gradual_decrease",
        "Gradual contact restrictions weaken infection pathways.",
        "I",
        "S",
    ),
    PseudoScenario(
        "treatment_improvement",
        "seihrdv",
        "graph",
        "gradual_decrease",
        "Improved treatment weakens the hospital-to-death pathway.",
        "H",
        "D",
    ),
    PseudoScenario(
        "waning_immunity",
        "seihrdv",
        "graph",
        "gradual_increase",
        "Increasing waning immunity strengthens R-to-S and V-to-S pathways.",
        "R",
        "S",
    ),
)

SCENARIOS = {
    "enso": ENSO_SCENARIOS,
    "seir": SEIR_SCENARIOS,
    "seihrdv": SEIHRDV_SCENARIOS,
}


def available_scenarios(family: str) -> tuple[PseudoScenario, ...]:
    try:
        return SCENARIOS[family]
    except KeyError as error:
        raise ValueError(f"Unknown pseudo-realistic family {family!r}; choose {PSEUDO_TYPES}") from error


def _scenario(family: str, scenario_key: str | None) -> PseudoScenario | None:
    if scenario_key in {None, "static"}:
        return None
    for candidate in available_scenarios(family):
        if candidate.key == scenario_key:
            return candidate
    choices = [item.key for item in available_scenarios(family)]
    raise ValueError(f"Unknown {family} scenario {scenario_key!r}; choose {choices} or 'static'")


def _scenario_schedule(length: int, scenario: PseudoScenario | None) -> tuple[np.ndarray, dict[str, int]]:
    if scenario is None:
        return np.zeros(length, dtype=np.float32), {"change_point": -1, "ramp_start": -1, "ramp_end": -1}
    return drift_curve(length, scenario.temporal_pattern), schedule_metadata(length, scenario.temporal_pattern)


def _attrs(
    family: str,
    scenario: PseudoScenario | None,
    seed: int,
    schedule: dict[str, int],
) -> dict[str, object]:
    return {
        "benchmark": "CausalDrift",
        "dataset_family": "pseudo_realistic",
        "dataset_type": family,
        "setting": "static" if scenario is None else "causal_drift",
        "scenario": "static" if scenario is None else scenario.key,
        "scenario_description": "Stationary baseline." if scenario is None else scenario.description,
        "drift_type": "none" if scenario is None else scenario.drift_type,
        "temporal_pattern": "static" if scenario is None else scenario.temporal_pattern,
        "drift_source_name": None if scenario is None else scenario.source,
        "drift_target_name": None if scenario is None else scenario.target,
        "seed": seed,
        **schedule,
    }


def _graph(names: list[str], edges: list[tuple[str, str, float]]) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    graph = np.zeros((len(names), len(names)), dtype=np.float32)
    for source, target, value in edges:
        graph[index[source], index[target]] = value
    return graph


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.where(std > 1e-6, std, 1.0)).astype(np.float32)


def generate_enso_sample(
    *,
    num_timesteps: int = 1_000,
    seed: int = 0,
    noise: float = 0.05,
    scenario_key: str | None = None,
) -> xr.Dataset:
    """Generate an ENSO-inspired, nonlinear coupled climate-index system.

    It is self-contained rather than bundling/redistributing a climate asset;
    source metadata explicitly identifies it as an ENSO-inspired simulator.
    """

    scenario = _scenario("enso", scenario_key)
    alpha, schedule = _scenario_schedule(num_timesteps, scenario)
    rng = np.random.default_rng(seed)
    names = ["Nino34", "WWV", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"]
    graph = _graph(
        names,
        [
            ("WWV", "Nino34", 0.85), ("NPMM", "Nino34", 0.45),
            ("SPMM", "Nino34", 0.40), ("IOD", "Nino34", 0.70),
            ("TNA", "Nino34", 0.40), ("Nino34", "WWV", 0.65),
            ("Nino34", "IOB", 0.55), ("Nino34", "IOD", 0.45),
            ("Nino34", "ATL3", 0.40), ("Nino34", "SIOD", 0.30),
            ("IOD", "IOB", 0.35), ("ATL3", "SASD", 0.30),
        ],
    )
    adjacency = np.repeat(graph[None, ...], num_timesteps, axis=0)
    noise_sigma = np.full((num_timesteps, len(names)), noise, dtype=np.float32)
    if scenario is not None and scenario.drift_type in {"edge", "strength"}:
        source, target = names.index(str(scenario.source)), names.index(str(scenario.target))
        if scenario.drift_type == "edge":
            adjacency[:, source, target] = alpha
        else:
            adjacency[:, source, target] = 0.20 + 0.80 * alpha
    if scenario is not None and scenario.drift_type == "noise":
        noise_sigma[:, names.index("Nino34")] = noise * (0.5 + 3.0 * alpha)

    series = np.zeros((num_timesteps, len(names)), dtype=np.float32)
    seasonal = np.sin(2.0 * np.pi * np.arange(num_timesteps) / 12.0)
    series[0] = rng.normal(0.0, 0.2, len(names))
    signed_weights = adjacency.copy()
    signed_weights[:, names.index("Nino34"), names.index("WWV")] *= -0.55
    for time_index in range(1, num_timesteps):
        previous = series[time_index - 1]
        forcing = signed_weights[time_index].T @ np.tanh(previous)
        series[time_index] = 0.55 * previous + 0.18 * forcing
        series[time_index, names.index("Nino34")] += 0.20 * seasonal[time_index]
        series[time_index] += rng.normal(0.0, noise_sigma[time_index])
    return make_dataset(
        _standardize(series),
        adjacency,
        node_names=names,
        alpha=alpha,
        attrs=_attrs("enso", scenario, seed, schedule) | {"source": "ENSO-inspired coupled climate-index simulator"},
        extra_variables={"noise_sigma": (("time", "node"), noise_sigma), "edge_strength": (("time", "source", "target"), adjacency)},
    )


def _seir_parameters(alpha: np.ndarray, scenario: PseudoScenario | None) -> np.ndarray:
    beta_low, beta_high, beta_default = 0.20, 0.46, 0.40
    if scenario is None:
        return np.full(len(alpha), beta_default, dtype=np.float32)
    if scenario.key == "abrupt_lockdown":
        return beta_low + alpha * (beta_high - beta_low)
    if scenario.key == "gradual_variant_spread":
        return beta_default + alpha * (beta_high - beta_default)
    if scenario.key == "abrupt_reopening":
        return beta_low + alpha * (beta_high - beta_low)
    if scenario.key == "gradual_behavior_change":
        return beta_low + alpha * (beta_high - beta_low)
    raise ValueError(f"Unsupported SEIR scenario {scenario.key}")


def _simulate_seir(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = ["S", "E", "I", "R"]
    length = len(beta)
    values = np.zeros((length, len(names)), dtype=np.float32)
    values[0] = [0.985, 0.006, 0.004, 0.005]
    adjacency = np.zeros((length, len(names), len(names)), dtype=np.float32)
    S, E, I, R = range(4)
    for time_index in range(length - 1):
        susceptible, exposed, infectious, recovered = values[time_index]
        infected = beta[time_index] * susceptible * infectious
        to_infectious = exposed / 4.5
        recovered_from_infectious = infectious / 7.0
        delta = np.asarray(
            [-infected, infected - to_infectious, to_infectious - recovered_from_infectious, recovered_from_infectious]
        )
        values[time_index + 1] = np.clip(values[time_index] + delta, 0.0, None)
        values[time_index + 1] /= values[time_index + 1].sum()
        transmission = float(np.clip(beta[time_index] / 0.46, 0.0, 1.0))
        adjacency[time_index, I, S] = transmission
        adjacency[time_index, S, E] = transmission
        adjacency[time_index, I, E] = transmission
        adjacency[time_index, E, I] = 0.80
        adjacency[time_index, I, R] = 0.80
    adjacency[-1] = adjacency[-2]
    return values, adjacency


def generate_seir_sample(
    *,
    num_timesteps: int = 1_000,
    seed: int = 0,
    noise: float = 0.01,
    scenario_key: str | None = None,
) -> xr.Dataset:
    scenario = _scenario("seir", scenario_key)
    alpha, schedule = _scenario_schedule(num_timesteps, scenario)
    rng = np.random.default_rng(seed)
    clean, adjacency = _simulate_seir(_seir_parameters(alpha, scenario))
    observed = np.clip(clean + rng.normal(0.0, noise, clean.shape), 0.0, None)
    return make_dataset(
        observed,
        adjacency,
        node_names=["S", "E", "I", "R"],
        alpha=alpha,
        attrs=_attrs("seir", scenario, seed, schedule) | {"source": "SEIR compartmental simulator"},
        extra_variables={"edge_strength": (("time", "source", "target"), adjacency)},
    )


def _seihrdv_parameters(alpha: np.ndarray, scenario: PseudoScenario | None) -> dict[str, np.ndarray]:
    length = len(alpha)
    values = {
        "beta": np.full(length, 0.44, dtype=np.float32),
        "dose": np.full(length, 0.006, dtype=np.float32),
        "breakthrough": np.full(length, 0.38, dtype=np.float32),
        "fatality": np.full(length, 0.010, dtype=np.float32),
        "waning": np.full(length, 1.0 / 260.0, dtype=np.float32),
    }
    if scenario is None:
        return values
    if scenario.key == "vaccination_rollout":
        values["dose"] += alpha * 0.010
        values["breakthrough"] -= alpha * 0.14
    elif scenario.key == "contact_restrictions":
        values["beta"] = 0.22 + alpha * (0.44 - 0.22)
    elif scenario.key == "treatment_improvement":
        values["fatality"] = 0.004 + alpha * (0.010 - 0.004)
    elif scenario.key == "waning_immunity":
        values["waning"] = (1.0 / 360.0) + alpha * ((1.0 / 150.0) - (1.0 / 360.0))
    else:
        raise ValueError(f"Unsupported SEIHRDV scenario {scenario.key}")
    return values


def _simulate_seihrdv(parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    names = ["S", "E", "I", "H", "R", "D", "V"]
    length = len(parameters["beta"])
    values = np.zeros((length, len(names)), dtype=np.float32)
    values[0] = [0.976, 0.006, 0.004, 0.001, 0.013, 0.0, 0.0]
    adjacency = np.zeros((length, len(names), len(names)), dtype=np.float32)
    S, E, I, H, R, D, V = range(7)
    for time_index in range(length - 1):
        susceptible, exposed, infectious, hospitalized, recovered, deceased, vaccinated = values[time_index]
        beta = float(parameters["beta"][time_index])
        dose = float(parameters["dose"][time_index])
        breakthrough = float(parameters["breakthrough"][time_index])
        fatality = float(parameters["fatality"][time_index])
        waning = float(parameters["waning"][time_index])
        infection_s = beta * susceptible * infectious
        infection_v = breakthrough * beta * vaccinated * infectious
        vaccination = min(dose * susceptible, 0.08 * susceptible)
        to_infectious = exposed / 4.5
        leave_infectious = infectious / 7.0
        to_hospital = 0.08 * leave_infectious
        recovered_i = 0.92 * leave_infectious
        leave_hospital = hospitalized / 10.0
        recovered_h = (1.0 - fatality) * leave_hospital
        deaths = fatality * leave_hospital
        delta = np.asarray(
            [
                -infection_s - vaccination + waning * recovered + waning * vaccinated,
                infection_s + infection_v - to_infectious,
                to_infectious - leave_infectious,
                to_hospital - leave_hospital,
                recovered_i + recovered_h - waning * recovered,
                deaths,
                vaccination - infection_v - waning * vaccinated,
            ]
        )
        values[time_index + 1] = np.clip(values[time_index] + delta, 0.0, None)
        values[time_index + 1] /= values[time_index + 1].sum()
        transmission = np.clip(beta / 0.62, 0.0, 1.0)
        adjacency[time_index, I, S] = transmission
        adjacency[time_index, S, E] = transmission
        adjacency[time_index, I, E] = transmission
        adjacency[time_index, V, E] = np.clip(breakthrough * beta / 0.62, 0.0, 1.0)
        adjacency[time_index, I, V] = np.clip(breakthrough * beta / 0.62, 0.0, 1.0)
        adjacency[time_index, E, I] = 0.80
        adjacency[time_index, I, H] = 0.50
        adjacency[time_index, I, R] = 0.70
        adjacency[time_index, H, R] = 0.70
        adjacency[time_index, H, D] = np.clip(fatality / 0.022, 0.0, 1.0)
        adjacency[time_index, R, S] = np.clip(waning / (1.0 / 150.0), 0.0, 1.0)
        adjacency[time_index, V, S] = np.clip(waning / (1.0 / 150.0), 0.0, 1.0)
        adjacency[time_index, S, V] = np.clip(dose / 0.018, 0.0, 1.0)
    adjacency[-1] = adjacency[-2]
    return values, adjacency


def generate_seihrdv_sample(
    *,
    num_timesteps: int = 1_000,
    seed: int = 0,
    noise: float = 0.01,
    scenario_key: str | None = None,
) -> xr.Dataset:
    scenario = _scenario("seihrdv", scenario_key)
    alpha, schedule = _scenario_schedule(num_timesteps, scenario)
    rng = np.random.default_rng(seed)
    clean, adjacency = _simulate_seihrdv(_seihrdv_parameters(alpha, scenario))
    observed = np.clip(clean + rng.normal(0.0, noise, clean.shape), 0.0, None)
    return make_dataset(
        observed,
        adjacency,
        node_names=["S", "E", "I", "H", "R", "D", "V"],
        alpha=alpha,
        attrs=_attrs("seihrdv", scenario, seed, schedule) | {"source": "SEIHRDV compartmental simulator"},
        extra_variables={"edge_strength": (("time", "source", "target"), adjacency)},
    )


def generate_pseudo_dataset(
    family: str,
    *,
    num_timesteps: int = 1_000,
    seed: int = 0,
    noise: float | None = None,
    scenario_key: str | None = None,
) -> xr.Dataset:
    """Generate a pseudo-realistic sample using one public, fixed simulator."""

    if family == "enso":
        return generate_enso_sample(
            num_timesteps=num_timesteps,
            seed=seed,
            noise=0.05 if noise is None else noise,
            scenario_key=scenario_key,
        )
    if family == "seir":
        return generate_seir_sample(
            num_timesteps=num_timesteps,
            seed=seed,
            noise=0.01 if noise is None else noise,
            scenario_key=scenario_key,
        )
    if family == "seihrdv":
        return generate_seihrdv_sample(
            num_timesteps=num_timesteps,
            seed=seed,
            noise=0.01 if noise is None else noise,
            scenario_key=scenario_key,
        )
    raise ValueError(f"Unknown pseudo-realistic family {family!r}; choose {PSEUDO_TYPES}")


def generate_pseudo_directory(
    out_dir: str | Path,
    *,
    families: tuple[str, ...] = PSEUDO_TYPES,
    settings: tuple[str, ...] = ("static", "causal_drift"),
    instances: int = 10,
    num_timesteps: int = 1_000,
    seed: int = 371_000,
) -> pd.DataFrame:
    """Write all fixed pseudo-realistic scenarios and a minimal manifest."""

    destination = Path(out_dir)
    records: list[dict[str, object]] = []
    counter = 0
    for family in families:
        if family not in PSEUDO_TYPES:
            raise ValueError(f"Unknown family {family!r}")
        scenario_keys: list[str | None] = []
        if "static" in settings:
            scenario_keys.append(None)
        if "causal_drift" in settings or "drift" in settings:
            scenario_keys.extend(item.key for item in available_scenarios(family))
        for scenario_key in scenario_keys:
            for instance in range(instances):
                sample_seed = seed + counter
                dataset = generate_pseudo_dataset(
                    family,
                    num_timesteps=num_timesteps,
                    seed=sample_seed,
                    scenario_key=scenario_key,
                )
                setting = str(dataset.attrs["setting"])
                scenario_name = str(dataset.attrs["scenario"])
                path = destination / family / setting / f"{family}_{scenario_name}_seed{sample_seed}.nc"
                save_dataset(dataset, path)
                records.append(
                    {
                        "path": path.relative_to(destination).as_posix(),
                        "family": family,
                        "setting": setting,
                        "scenario": scenario_name,
                        "instance": instance,
                        "seed": sample_seed,
                        "num_timesteps": num_timesteps,
                    }
                )
                counter += 1
    manifest = pd.DataFrame.from_records(records)
    destination.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination / "manifest.csv", index=False)
    return manifest
