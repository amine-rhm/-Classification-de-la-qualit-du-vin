from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from wine import load_data, make_pipe, prepare, split

ROOT = Path(__file__).resolve().parent.parent
RED, WHITE = ROOT / "winequality-red.csv", ROOT / "winequality-white.csv"
DATA_AVAILABLE = RED.exists() and WHITE.exists()

FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"]


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame(rng.normal(size=(n, 11)), columns=FEATURES)
    d["quality"] = rng.integers(3, 9, size=n)
    d["type"] = rng.choice(["red", "white"], size=n)
    return d


def test_threshold_boundary_is_7():
    d = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in FEATURES})
    d["quality"] = [6, 7, 8]
    d["type"] = "red"
    _, y = prepare(d)
    assert y.tolist() == [0, 1, 1]


def test_duplicates_are_removed(df):
    X, y = prepare(pd.concat([df, df.head(10)]))
    assert len(X) == len(df) and len(y) == len(df)
    assert X.index.equals(y.index)


def test_features_have_no_target_leakage(df):
    X, _ = prepare(df)
    assert not {"quality", "good", "type"} & set(X.columns)
    assert "type_white" in X.columns
    assert X.shape[1] == 12
    assert X.isna().sum().sum() == 0


def test_type_is_encoded_correctly(df):
    X, _ = prepare(df)
    d = df.drop_duplicates().reset_index(drop=True)
    assert (X["type_white"] == (d["type"] == "white").astype(int)).all()


def test_prepare_does_not_modify_input(df):
    before = df.copy()
    prepare(df)
    pd.testing.assert_frame_equal(df, before)


def test_split_is_stratified_and_disjoint(df):
    X, y = prepare(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_split_is_reproducible_and_seed_dependent(df):
    X, y = prepare(df)
    a = split(X, y, seed=1)[1].index
    b = split(X, y, seed=1)[1].index
    c = split(X, y, seed=2)[1].index
    assert a.equals(b) and not a.equals(c)


def test_scaler_is_fitted_on_training_data_only(df):
    X, y = prepare(df)
    X_tr, _, y_tr, _ = split(X, y)
    pipe = make_pipe(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    assert list(pipe.named_steps) == ["scaler", "select", "clf"]
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_tr.mean().values)


def test_make_pipe_selects_k_features(df):
    X, y = prepare(df)
    pipe = make_pipe(LogisticRegression(max_iter=1000), k=6).fit(X, y)
    assert pipe.named_steps["select"].get_support().sum() == 6


def test_pipeline_predicts_valid_labels(df):
    X, y = prepare(df)
    X_tr, X_te, y_tr, _ = split(X, y)
    pipe = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    assert pred.shape == (len(X_te),) and set(pred) <= {0, 1}


def test_pipeline_is_reproducible(df):
    X, y = prepare(df)
    X_tr, X_te, y_tr, _ = split(X, y)
    a = make_pipe(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr).predict(X_te)
    b = make_pipe(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr).predict(X_te)
    assert (a == b).all()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="données absentes")
def test_real_data_shape_after_cleaning():
    raw = load_data(RED, WHITE)
    assert raw.shape == (6497, 13)
    X, y = prepare(raw)
    assert len(X) == 5320 and y.sum() == 1009


@pytest.mark.skipif(not DATA_AVAILABLE, reason="données absentes")
def test_model_beats_baseline_on_real_data():
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC
    X, y = prepare(load_data(RED, WHITE))
    X_tr, X_te, y_tr, y_te = split(X, y)
    pipe = make_pipe(SVC(class_weight="balanced")).fit(X_tr, y_tr)
    assert roc_auc_score(y_te, pipe.decision_function(X_te)) > 0.75