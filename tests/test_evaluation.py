import numpy as np

from causaldrift.data import make_dataset, save_dataset, series_matrix, standardize_window
from causaldrift.evaluation import evaluate_dataset, evaluate_directory, random_windows, score_graph, structural_hamming_distance
from causaldrift.models import ACD, MODEL_CLASSES, ModelConfig, UnCLe, DiscoveryModel, _grasp_directed_support, _source_to_target_coefficients
from causaldrift.plotting import _tradeoff_style_map, make_plots
from causaldrift.synthetic import DriftConfig, generate_synthetic_dataset


class _DeterministicModel(DiscoveryModel):
    name = "ngc"

    def predict_many(self, windows):
        nodes = windows.shape[-1]
        scores = np.zeros((windows.shape[0], nodes, nodes), dtype=np.float32)
        scores[:, 1:, :-1] = 0.75
        return scores


def test_shd_counts_a_reversal_as_one_edit():
    truth = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    reversed_edge = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    assert structural_hamming_distance(reversed_edge, truth) == 1.0
    metrics = score_graph(reversed_edge, truth)
    assert metrics["norm_shd"] == 0.5
    assert 0.0 <= metrics["norm_whd"] <= 1.0


def test_support_metrics_treat_a_positive_nonbinary_truth_strength_as_an_edge():
    truth = np.asarray([[0.0, 0.30], [0.0, 0.0]])
    prediction = np.asarray([[0.0, 0.70], [0.20, 0.0]])
    metrics = score_graph(prediction, truth)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["norm_shd"] == 0.0
    assert metrics["norm_whd"] == 0.30


def test_window_standardization_cannot_depend_on_future_values():
    first_window = np.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32)
    changed_future = np.vstack([first_window, [[1_000_000.0, -1_000_000.0]]])
    dataset = make_dataset(changed_future, np.zeros((3, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(series_matrix(dataset)[:2], first_window)
    np.testing.assert_array_equal(standardize_window(series_matrix(dataset)[:2]), standardize_window(first_window))

    missing_prefix = np.asarray([[np.nan, 1.0], [np.nan, 3.0], [999.0, 5.0]], dtype=np.float32)
    missing_dataset = make_dataset(missing_prefix, np.zeros((3, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        standardize_window(series_matrix(missing_dataset)[:2]),
        standardize_window(missing_prefix[:2]),
    )


def test_grasp_conversion_keeps_only_definite_directed_edges():
    directed = np.asarray([[0, -1], [1, 0]])  # X0 -> X1 in causal-learn endpoint encoding.
    undirected = np.asarray([[0, -1], [-1, 0]])
    np.testing.assert_array_equal(_grasp_directed_support(directed), [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(_grasp_directed_support(undirected), np.zeros((2, 2)))


def test_varlingam_coefficients_are_converted_to_source_target_order():
    # lingam stores B[target, source], hence B[1, 0] is X0 -> X1.
    coefficients = np.asarray([[0.0, 0.0], [0.8, 0.0]])
    np.testing.assert_array_equal(_source_to_target_coefficients(coefficients), [[0.0, 0.8], [0.0, 0.0]])


def test_amortized_models_fit_each_evaluation_window_independently(monkeypatch):
    import causaldrift.amortized as amortized

    calls = []

    class _Runner:
        def __init__(self, _config):
            return None

        def fit_predict(self, windows):
            calls.append(windows.shape)
            return np.zeros((windows.shape[0], windows.shape[2], windows.shape[2]), dtype=np.float32)

    windows = np.ones((3, 5, 2), dtype=np.float32)
    for model_class, runner_name in ((ACD, "ACDRunner"), (UnCLe, "UnCLeRunner")):
        calls.clear()
        monkeypatch.setattr(amortized, runner_name, _Runner)
        prediction = model_class(ModelConfig()).predict_many(windows)
        assert prediction.shape == (3, 2, 2)
        assert calls == [(1, 5, 2), (1, 5, 2), (1, 5, 2)]


def test_evaluation_uses_the_last_included_timestep_as_ground_truth(monkeypatch):
    monkeypatch.setitem(MODEL_CLASSES, "ngc", _DeterministicModel)
    adjacency = np.zeros((4, 2, 2), dtype=np.float32)
    adjacency[-1, 0, 1] = 1.0
    dataset = make_dataset(np.arange(8, dtype=np.float32).reshape(4, 2), adjacency)

    predictions, _metrics = evaluate_dataset(
        dataset,
        models=["ngc"],
        num_windows=1,
        window_size=4,
        seed=7,
    )

    np.testing.assert_array_equal(predictions["truth"].isel(window=0).to_numpy(), adjacency[-1])
    assert predictions["truth_time"].item() == 3
    assert "endpoint" in predictions.attrs["protocol"]


def test_tradeoff_styles_are_unique_for_the_ten_benchmark_models():
    models = ["acd", "cuts", "grasp", "ngc", "uncle", "varlingam", "cdans", "fpcmci", "kausal", "pcmci+"]
    styles = _tradeoff_style_map(np.asarray(models))
    assert set(styles) == set(models)
    assert len({color for color, _marker in styles.values()}) == len(models)
    assert len({marker for _color, marker in styles.values()}) == len(models)


def test_directory_evaluation_writes_ten_window_summary_and_two_plots(tmp_path, monkeypatch):
    monkeypatch.setitem(MODEL_CLASSES, "ngc", _DeterministicModel)
    data_dir = tmp_path / "data"
    save_dataset(
        generate_synthetic_dataset(
            DriftConfig(num_timesteps=60, num_nodes=4, seed=4)
        ),
        data_dir / "sample.nc",
    )
    raw, summary, overall = evaluate_directory(data_dir, tmp_path / "results", models=["ngc"])
    assert len(raw) == 10
    assert summary.iloc[0]["num_windows"] == 10
    assert summary.iloc[0]["successful_windows"] == 10
    assert len(overall) == 1
    starts = random_windows(60, count=10, seed=123)
    assert len(starts) == 10
    _aggregate, tradeoff, ranking = make_plots(tmp_path / "results", tmp_path / "figures")
    assert tradeoff.exists()
    assert ranking.exists()
