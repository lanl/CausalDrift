"""Command-line entry point for the lean public benchmark workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_directory
from .graphs import DRIFT_TYPES, TEMPORAL_PATTERNS
from .models import model_table
from .plotting import make_plots
from .pseudo import PSEUDO_TYPES, generate_pseudo_directory
from .synthetic import generate_synthetic_directory


def _selection(values: list[str], allowed: tuple[str, ...], option: str) -> tuple[str, ...]:
    if values == ["all"]:
        return allowed
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{option} contains {unknown}; choose from {allowed} or all")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causaldrift", description="CausalDrift benchmark")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="generate synthetic or pseudo-realistic data")
    generators = generate.add_subparsers(dest="family", required=True)
    synthetic = generators.add_parser("synthetic", help="generate fixed-config synthetic samples")
    synthetic.add_argument("--out", type=Path, required=True)
    synthetic.add_argument("--instances", type=int, default=10)
    synthetic.add_argument("--timesteps", type=int, default=1_000)
    synthetic.add_argument("--nodes", type=int, default=5)
    synthetic.add_argument("--seed", type=int, default=202_600)
    synthetic.add_argument("--drift-types", nargs="+", default=["all"])
    synthetic.add_argument("--temporal-patterns", nargs="+", default=["all"])

    pseudo = generators.add_parser("pseudo", help="generate fixed pseudo-realistic scenarios")
    pseudo.add_argument("--out", type=Path, required=True)
    pseudo.add_argument("--instances", type=int, default=10)
    pseudo.add_argument("--timesteps", type=int, default=1_000)
    pseudo.add_argument("--seed", type=int, default=371_000)
    pseudo.add_argument("--types", nargs="+", default=["all"])
    pseudo.add_argument("--settings", nargs="+", choices=["static", "causal_drift"], default=["static", "causal_drift"])

    evaluate = subcommands.add_parser("evaluate", help="run models and evaluate ten random windows")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--models", nargs="+", default=["all"])
    evaluate.add_argument("--num-windows", type=int, default=10, help="Default benchmark protocol is 10")
    evaluate.add_argument("--window-size", type=int, default=None, help="Defaults to half the trajectory length")
    evaluate.add_argument("--seed", type=int, default=202_606)
    evaluate.add_argument("--third-party-dir", type=Path, default=None)
    evaluate.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    evaluate.add_argument("--continue-on-error", action="store_true", help="Record NaNs for unavailable model/files instead of failing")

    plot = subcommands.add_parser("plot", help="create trade-off and μ−σ ranking plots")
    plot.add_argument("--results", type=Path, required=True, help="evaluation directory or summary_metrics.csv")
    plot.add_argument("--out", type=Path, required=True)

    subcommands.add_parser("models", help="show the ten fixed model configurations")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "models":
        print(pd.DataFrame(model_table()).to_string(index=False))
        return
    if args.command == "generate" and args.family == "synthetic":
        manifest = generate_synthetic_directory(
            args.out,
            instances=args.instances,
            num_timesteps=args.timesteps,
            num_nodes=args.nodes,
            seed=args.seed,
            drift_types=_selection(args.drift_types, DRIFT_TYPES, "--drift-types"),
            temporal_patterns=_selection(args.temporal_patterns, TEMPORAL_PATTERNS, "--temporal-patterns"),
        )
        print(f"wrote {len(manifest)} synthetic samples to {args.out}")
        return
    if args.command == "generate" and args.family == "pseudo":
        manifest = generate_pseudo_directory(
            args.out,
            families=_selection(args.types, PSEUDO_TYPES, "--types"),
            settings=tuple(args.settings),
            instances=args.instances,
            num_timesteps=args.timesteps,
            seed=args.seed,
        )
        print(f"wrote {len(manifest)} pseudo-realistic samples to {args.out}")
        return
    if args.command == "evaluate":
        _raw, summary, overall = evaluate_directory(
            args.data,
            args.out,
            models=args.models,
            num_windows=args.num_windows,
            window_size=args.window_size,
            seed=args.seed,
            third_party_dir=args.third_party_dir,
            device=args.device,
            continue_on_error=args.continue_on_error,
        )
        print(f"\nPer-dataset {args.num_windows}-window summary")
        print(summary.to_string(index=False))
        print("\nOverall model summary")
        print(overall.to_string(index=False))
        return
    if args.command == "plot":
        aggregate, tradeoff, ranking = make_plots(args.results, args.out)
        print(aggregate.to_string(index=False))
        print(f"wrote {tradeoff} and {ranking}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")
