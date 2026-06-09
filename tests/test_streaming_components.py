import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import EnsembleClassifier
from numcompute.stream import StreamTrainer
from numcompute.metrics import StreamingAccuracy, StreamingClassificationMetrics
from numcompute.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder
from numcompute.pipeline import Pipeline
from numcompute.stats import StreamingStats, update_stats


def test_decision_tree_fit_predict():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    assert model.predict(X).shape == y.shape


def test_decision_tree_partial_fit():
    model = DecisionTreeClassifier(max_depth=2)
    model.partial_fit(np.array([[0], [1]]), np.array([0, 0]))
    model.partial_fit(np.array([[2], [3]]), np.array([1, 1]))
    assert model.predict(np.array([[0], [3]])).shape == (2,)


def test_decision_tree_entropy():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    model.fit(X, y)
    assert len(model.predict(X)) == 4


def test_ensemble_fit_predict():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=1)
    model.fit(X, y)
    assert model.predict(X).shape == y.shape


def test_ensemble_partial_fit():
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=1)
    model.partial_fit(np.array([[0], [1], [2]]), np.array([0, 0, 0]))
    model.partial_fit(np.array([[3], [4], [5]]), np.array([1, 1, 1]))
    assert model.predict(np.array([[0], [5]])).shape == (2,)


def test_stream_trainer_logs():
    model = DecisionTreeClassifier(max_depth=2)
    trainer = StreamTrainer(model)

    log = trainer.fit_chunk(
        np.array([[0], [1], [2], [3]], dtype=float),
        np.array([0, 0, 1, 1]),
    )

    assert "chunk_accuracy" in log
    assert "cumulative_accuracy" in log
    assert "memory_bytes" in log
    assert len(trainer.get_logs()) == 1


def test_stream_trainer_score_chunk():
    model = DecisionTreeClassifier(max_depth=2)
    trainer = StreamTrainer(model)

    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    trainer.fit_chunk(X, y)
    score = trainer.score_chunk(X, y)

    assert 0.0 <= score <= 1.0


def test_streaming_accuracy_update_result_reset():
    metric = StreamingAccuracy()
    metric.update(np.array([1, 0, 1]), np.array([1, 1, 1]))
    assert np.isclose(metric.result(), 2 / 3)
    metric.reset()
    assert metric.result() == 0.0


def test_streaming_classification_metrics():
    metric = StreamingClassificationMetrics(average="binary")
    metric.update(np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]))
    result = metric.result()

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert result["confusion_matrix"].shape[0] > 0


def test_streaming_classification_rolling_window():
    metric = StreamingClassificationMetrics(rolling_window=2)
    metric.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
    result = metric.result()
    assert 0.0 <= result["accuracy"] <= 1.0


def test_standard_scaler_partial_fit():
    sc = StandardScaler()
    sc.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    sc.partial_fit(np.array([[5.0, 6.0]]))
    out = sc.transform(np.array([[1.0, 2.0]]))
    assert out.shape == (1, 2)


def test_simple_imputer_partial_fit():
    imp = SimpleImputer(strategy="mean")
    imp.partial_fit(np.array([[1.0, np.nan], [3.0, 4.0]]))
    out = imp.transform(np.array([[np.nan, np.nan]]))
    assert out.shape == (1, 2)


def test_onehot_partial_fit_expands_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([[0], [1]], dtype=float))
    enc.partial_fit(np.array([[2]], dtype=float))
    out = enc.transform(np.array([[2]], dtype=float))
    assert out.shape == (1, 3)


def test_pipeline_partial_fit():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=2)),
    ])

    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    pipe.partial_fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape


def test_streaming_stats_update_stats():
    state = update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    state = update_stats(np.array([[5.0, 6.0]]), state)
    result = state.result()

    assert "mean" in result
    assert "variance" in result
    assert result["n_chunks"] == 2


def test_streaming_stats_class():
    ss = StreamingStats()
    ss.update_stats(np.array([[1.0], [2.0], [3.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 2.0)
def test_visualise_compare_models_runs():
    from numcompute.visualise import compare_models

    fig, ax = compare_models(
        [0.8, 0.9, 1.0],
        [0.7, 0.85, 0.95],
        labels=("Tree", "Ensemble"),
        show=False,
    )

    assert fig is not None
    assert ax is not None