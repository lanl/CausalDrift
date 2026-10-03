"""Causal discovery under drift."""

from .evaluation import METRICS, evaluate_directory
from .pseudo import generate_pseudo_dataset
from .synthetic import DriftConfig, generate_synthetic_dataset

__all__ = [
    "DriftConfig",
    "METRICS",
    "evaluate_directory",
    "generate_pseudo_dataset",
    "generate_synthetic_dataset",
]

__version__ = "0.1.0"
