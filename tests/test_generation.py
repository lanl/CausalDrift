import numpy as np
import pytest

from causaldrift.graphs import DRIFT_TYPES
from causaldrift.pseudo import PSEUDO_TYPES, generate_pseudo_dataset
from causaldrift.synthetic import DriftConfig, generate_synthetic_dataset


@pytest.mark.parametrize("drift_type", DRIFT_TYPES)
def test_each_synthetic_drift_keeps_the_public_data_contract(drift_type):
    dataset = generate_synthetic_dataset(
        DriftConfig(
            drift_type=drift_type,
            temporal_pattern="gradual_increase",
            num_timesteps=40,
            num_nodes=5,
            seed=7,
        )
    )
    assert dataset.time_series.shape == (40, 5, 1)
    assert dataset.adjacency.shape == (40, 5, 5)
    assert np.isfinite(dataset.adjacency).all()
    assert dataset.attrs["drift_type"] == drift_type


def test_confounder_records_the_actual_latent_targets_without_inventing_directed_truth():
    dataset = generate_synthetic_dataset(
        DriftConfig(drift_type="confounder", num_timesteps=40, num_nodes=5, seed=11)
    )
    first = dataset.attrs["latent_confounder_target_1"]
    second = dataset.attrs["latent_confounder_target_2"]
    assert first != second
    assert 0 <= first < dataset.time_series.shape[1]
    assert 0 <= second < dataset.time_series.shape[1]
    assert dataset.attrs["truth_scope"] == "observed_directed_edges"
    assert dataset.attrs["drift_semantics"] == "latent_confounding_without_directed_edge_change"


@pytest.mark.parametrize("family", PSEUDO_TYPES)
def test_each_pseudo_realistic_family_uses_the_same_contract(family):
    dataset = generate_pseudo_dataset(family, num_timesteps=40, seed=8)
    assert dataset.time_series.sizes["time"] == 40
    assert dataset.adjacency.shape[0] == 40
    assert dataset.adjacency.shape[1] == dataset.time_series.shape[1]
    assert dataset.attrs["dataset_family"] == "pseudo_realistic"
