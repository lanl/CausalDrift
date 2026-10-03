"""Fixed benchmark configurations for each model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import os
import sys
import types

import numpy as np


DEFAULT_MODELS = (
    "acd",
    "cuts",
    "grasp",
    "ngc",
    "uncle",
    "varlingam",
    "cdans",
    "fpcmci",
    "kausal",
    "pcmci+",
)

ALIASES = {"pcmci+": "pcmci+", "pcmciplus": "pcmci+", "f-pcmci": "fpcmci"}


class ModelUnavailableError(RuntimeError):
    """Raised with an actionable setup message instead of a zero prediction."""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    output_kind: str
    threshold: float
    description: str
    dependency: str


MODEL_SPECS = {
    "pcmci+": ModelSpec("PCMCI+", "score", 0.5, "Absolute partial-correlation test statistic at lag 1.", "tigramite"),
    "fpcmci": ModelSpec("F-PCMCI", "binary", 0.5, "Correlation-selector F-PCMCI support at lag 1.", "fpcmci + tigramite"),
    "varlingam": ModelSpec("VARLiNGAM", "score", 0.5, "Normalized absolute lag-1 VARLiNGAM coefficient, transposed to source-to-target.", "lingam"),
    "grasp": ModelSpec("GRaSP", "binary", 0.5, "Definite directed BIC-covariance GRaSP support, depth 3.", "causal-learn"),
    "ngc": ModelSpec("NGC", "score", 0.5, "Internal LSTM Neural Granger Causality score (100 iterations).", "torch"),
    "kausal": ModelSpec("Kausal", "score", 0.5, "MLP observable, one epoch, 30 bootstrap samples.", "third_party/kausal"),
    "cuts": ModelSpec("CUTS", "score", 0.5, "Delayed-supervision CUTS, 100 epochs.", "third_party/UNN/CUTS"),
    "cdans": ModelSpec("CDANs", "binary", 0.5, "Fisher-Z CDANs with tau_max=2.", "third_party/CDANs"),
    "acd": ModelSpec("ACD", "score", 0.5, "Amortized ACD posterior fitted independently per window (100 epochs).", "third_party/AmortizedCausalDiscovery"),
    "uncle": ModelSpec("UnCLe", "score", 0.5, "Amortized UnCLe fitted independently per window (50 + 100 epochs).", "third_party/uncle-causal-discovery"),
}


@dataclass(frozen=True)
class ModelConfig:
    """Shared, non-search execution settings."""

    third_party_dir: Path | None = None
    device: str | None = None
    seed: int = 0


def normalize_models(names: list[str] | tuple[str, ...] | str) -> tuple[str, ...]:
    """Normalize aliases and expand the single useful shorthand, ``all``."""

    items = [names] if isinstance(names, str) else list(names)
    if len(items) == 1 and items[0].lower() == "all":
        return DEFAULT_MODELS
    result = tuple(ALIASES.get(item.lower(), item.lower()) for item in items)
    unknown = sorted(set(result) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose {DEFAULT_MODELS} or 'all'")
    if len(set(result)) != len(result):
        raise ValueError("A model was requested more than once")
    return result


def _unit_scale(values: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(scores.max()) if scores.size else 0.0
    return np.clip(scores / maximum if maximum > 0.0 else scores, 0.0, 1.0).astype(np.float32)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.where(std > 1e-6, std, 1.0)


def _source_to_target_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Transpose a ``B[target, source]`` coefficient matrix."""

    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square coefficient matrix, got {values.shape}")
    return values.T


def _grasp_directed_support(endpoint_matrix: np.ndarray) -> np.ndarray:
    """Return definite source-to-target edges from causal-learn endpoints."""

    endpoints = np.asarray(endpoint_matrix, dtype=int)
    if endpoints.ndim != 2 or endpoints.shape[0] != endpoints.shape[1]:
        raise ValueError(f"Expected a square causal-learn endpoint matrix, got {endpoints.shape}")
    return ((endpoints == -1) & (endpoints.T == 1)).astype(np.float32)


def _third_party_dir(config: ModelConfig) -> Path:
    if config.third_party_dir is not None:
        return Path(config.third_party_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "third_party"


def _prepend(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _require_import(model: str, action: str, error: Exception) -> ModelUnavailableError:
    return ModelUnavailableError(
        f"{MODEL_SPECS[model].label} is unavailable ({type(error).__name__}: {error}). "
        f"Install {MODEL_SPECS[model].dependency} as documented in third_party/README.md, "
        f"then retry {action}."
    )


class DiscoveryModel:
    """Minimal model interface; batch models override ``predict_many``."""

    name: str

    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def spec(self) -> ModelSpec:
        return MODEL_SPECS[self.name]

    def predict(self, series: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_many(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(window) for window in windows], axis=0)


class PCMCIPlus(DiscoveryModel):
    name = "pcmci+"

    def predict(self, series: np.ndarray) -> np.ndarray:
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        result = PCMCI(
            dataframe=pp.DataFrame(series),
            cond_ind_test=ParCorr(significance="analytic"),
            verbosity=0,
        ).run_pcmciplus(tau_min=1, tau_max=1, pc_alpha=0.05)
        return _unit_scale(result["val_matrix"][..., -1])


class FPCMCI(DiscoveryModel):
    name = "fpcmci"

    def predict(self, series: np.ndarray) -> np.ndarray:
        try:
            from fpcmci.CPrinter import CPLevel
            from fpcmci.FPCMCI import FPCMCI as FPCMCIClass
            from fpcmci.preprocessing.data import Data
            from fpcmci.selection_methods.Corr import Corr
            from tigramite.independence_tests.parcorr import ParCorr
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        result = FPCMCIClass(
            Data(series),
            f_alpha=0.01,
            pcmci_alpha=0.01,
            min_lag=1,
            max_lag=1,
            sel_method=Corr(),
            val_condtest=ParCorr(significance="analytic"),
            verbosity=CPLevel.NONE,
        ).run()[1]
        if result is None:
            return np.zeros((series.shape[1], series.shape[1]), dtype=np.float32)
        skeleton = np.asarray(result.get_skeleton()[-1], dtype=np.float32)
        output = np.zeros((series.shape[1], series.shape[1]), dtype=np.float32)
        features = list(getattr(result, "features", []))
        indices = []
        for position, feature in enumerate(features or [f"X_{index}" for index in range(skeleton.shape[0])]):
            try:
                indices.append(int(str(feature).rsplit("_", 1)[-1]))
            except ValueError:
                indices.append(position)
        for source_position, source in enumerate(indices):
            for target_position, target in enumerate(indices):
                if source < output.shape[0] and target < output.shape[1]:
                    output[source, target] = skeleton[source_position, target_position]
        return np.clip(output, 0.0, 1.0)


class VARLiNGAM(DiscoveryModel):
    name = "varlingam"

    def predict(self, series: np.ndarray) -> np.ndarray:
        try:
            import lingam
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        estimator = lingam.VARLiNGAM(lags=1)
        try:
            estimator.fit(series)
        except np.linalg.LinAlgError:
            estimator.fit(series + np.random.default_rng(self.config.seed).normal(0.0, 1e-6, series.shape))
        # lingam: B[target, source]
        return _unit_scale(_source_to_target_coefficients(estimator.adjacency_matrices_[1]))


class GRaSP(DiscoveryModel):
    name = "grasp"

    def predict(self, series: np.ndarray) -> np.ndarray:
        try:
            from causallearn.search.PermutationBased.GRaSP import grasp
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        graph = grasp(
            series,
            score_func="local_score_BIC_from_cov",
            depth=3,
            parameters=None,
            verbose=False,
        )
        return _grasp_directed_support(graph.graph)


class NGC(DiscoveryModel):
    """Small in-tree Neural Granger Causality implementation, fixed at 100 steps."""

    name = "ngc"

    def predict(self, series: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torch.nn as nn
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        data = torch.tensor(np.asarray(series, dtype=np.float32), device=device)
        if data.shape[0] < 3:
            raise ValueError("NGC needs at least three time steps")
        inputs = data[:-1].unsqueeze(0)
        targets = data[1:]
        hidden_dim = 16
        networks = nn.ModuleList([nn.LSTM(data.shape[1], hidden_dim, batch_first=True) for _ in range(data.shape[1])]).to(device)
        readouts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(data.shape[1])]).to(device)
        optimizer = torch.optim.Adam(list(networks.parameters()) + list(readouts.parameters()), lr=1e-3)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            group_penalty = torch.tensor(0.0, device=device)
            for target, (network, readout) in enumerate(zip(networks, readouts)):
                prediction, _ = network(inputs)
                loss = loss + nn.functional.mse_loss(readout(prediction)[0, :, 0], targets[:, target])
                group_penalty = group_penalty + torch.norm(network.weight_ih_l0, dim=0).sum()
            objective = loss / data.shape[1] + 0.01 * group_penalty / data.shape[1]
            objective.backward()
            optimizer.step()
        raw = torch.stack([torch.norm(network.weight_ih_l0, dim=0) for network in networks]).T
        return _unit_scale(raw.detach().cpu().numpy())


class Kausal(DiscoveryModel):
    name = "kausal"

    def predict(self, series: np.ndarray) -> np.ndarray:
        root = _third_party_dir(self.config)
        _prepend(root / "kausal")
        try:
            import torch
            from kausal import Graph
            from kausal.observables import MLPFeatures
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        values = torch.tensor(np.asarray(series, dtype=np.float32).T[:, None, :])
        graph = Graph(
            marginal_observable=MLPFeatures(in_channels=1, hidden_channels=[8, 16], out_channels=1),
            joint_observable=MLPFeatures(in_channels=2, hidden_channels=[8, 16], out_channels=1),
        )
        train_length = int(values.shape[-1])
        graph.infer(
            X=values,
            time_shift=min(100, max(1, train_length - 1)),
            fit_kwargs={"n_train": train_length, "batch_size": train_length, "epochs": 1, "lr": 1e-3},
            bootstrap_kwargs={"bootstrap_ratio": 0.9, "bootstrap_nums": 30},
        )
        output = np.zeros((values.shape[0], values.shape[0]), dtype=np.float32)
        for (source, target), stats in graph.causal_graph.items():
            output[int(source), int(target)] = max(0.0, float(stats["mean"])) * (1.0 - float(stats["pval"]))
        return _unit_scale(output)


class CUTS(DiscoveryModel):
    name = "cuts"

    def predict(self, series: np.ndarray) -> np.ndarray:
        root = _third_party_dir(self.config)
        cuts_root = root / "UNN" / "CUTS"
        _prepend(cuts_root)
        # CUTS uses a top-level, namespace-package ``utils`` directory.  Build
        # that package explicitly before stubbing only its side-effectful
        # logger; otherwise an unrelated installed ``utils`` module wins.
        for module_name in [name for name in sys.modules if name == "utils" or name.startswith("utils.")]:
            del sys.modules[module_name]
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = [str(cuts_root / "utils")]
        sys.modules["utils"] = utils_package
        logger_stub = types.ModuleType("utils.logger")
        logger_stub.MyLogger = _NoOpLogger
        sys.modules["utils.logger"] = logger_stub
        try:
            import torch
            from cuts_main import CUTS as CUTSRunner
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        values = _standardize(series)[:, :, None].astype(np.float32)
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = types.SimpleNamespace(
            n_nodes=values.shape[1], input_step=1, batch_size=min(512, max(1, values.shape[0] - 1)), data_dim=1,
            total_epoch=100, supervision_policy="masked_before_50", fill_policy="rate_0.1_after_10",
            show_graph_every=101, causal_thres="value_0.5",
            data_pred=types.SimpleNamespace(model="multi_mlp", pred_step=1, mlp_hid=128, mlp_layers=3, lr_data_start=1e-4, lr_data_end=1e-5, weight_decay=1e-3, prob=True),
            graph_discov=types.SimpleNamespace(lambda_s_start=0.1, lambda_s_end=0.1, lr_graph_start=1e-2, lr_graph_end=1e-3, start_tau=1.0, end_tau=0.1),
        )
        runner = CUTSRunner(args, _NoOpLogger(), device=device)
        # Upstream CUTS emits one progress bar per window.  The benchmark's
        # own concise dataset progress is sufficient, so keep its training
        # chatter out of notebooks/CI logs.
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                runner.train(values, np.ones_like(values), values.copy(), true_cm=None)
        scores = torch.sigmoid(runner.graph).detach().cpu().numpy()
        return np.clip(np.max(scores, axis=2).T, 0.0, 1.0)


class _NoOpLogger:
    def log_metrics(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        return None

    def log_figures(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        return None

    def log_npz(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        return None

    def log_opt(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        return None

    def close(self) -> None:
        return None


class CDANs(DiscoveryModel):
    name = "cdans"

    def predict(self, series: np.ndarray) -> np.ndarray:
        _prepend(_third_party_dir(self.config) / "CDANs" / "src")
        try:
            from cdans import CDANs as CDANsRunner
        except Exception as error:  # pragma: no cover - dependency specific
            raise _require_import(self.name, "the benchmark", error) from error
        result = CDANsRunner(
            tau_max=2,
            alpha=0.05,
            pc_alpha=0.2,
            ci_test="fisherz",
            max_extra_conds=2,
            use_independent_change=True,
            independent_change_width="auto",
            verbose=False,
        ).fit(_standardize(series))
        output = np.asarray(result.graph.contemp_adj, dtype=np.float32).copy()
        for source, target, _lag in result.graph.lagged_edges:
            output[int(source), int(target)] = 1.0
        np.fill_diagonal(output, 0.0)
        return np.clip(output, 0.0, 1.0)


class ACD(DiscoveryModel):
    name = "acd"

    def predict(self, series: np.ndarray) -> np.ndarray:
        from .amortized import ACDRunner

        return ACDRunner(self.config).fit_predict(np.asarray(series, dtype=np.float32)[None, ...])[0]


class UnCLe(DiscoveryModel):
    name = "uncle"

    def predict(self, series: np.ndarray) -> np.ndarray:
        from .amortized import UnCLeRunner

        return UnCLeRunner(self.config).fit_predict(np.asarray(series, dtype=np.float32)[None, ...])[0]


MODEL_CLASSES = {
    "pcmci+": PCMCIPlus,
    "fpcmci": FPCMCI,
    "varlingam": VARLiNGAM,
    "grasp": GRaSP,
    "ngc": NGC,
    "kausal": Kausal,
    "cuts": CUTS,
    "cdans": CDANs,
    "acd": ACD,
    "uncle": UnCLe,
}


def create_model(name: str, config: ModelConfig | None = None) -> DiscoveryModel:
    canonical = normalize_models(name)[0]
    return MODEL_CLASSES[canonical](config or ModelConfig())


def model_table() -> list[dict[str, str | float]]:
    """Rows suitable for the ``models`` CLI command and README verification."""

    return [
        {
            "id": name,
            "label": spec.label,
            "output_kind": spec.output_kind,
            "threshold": spec.threshold,
            "default": spec.description,
            "dependency": spec.dependency,
        }
        for name, spec in MODEL_SPECS.items()
    ]
