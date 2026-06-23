"""End-to-end smoke test on synthetic imbalanced data.

Exercises the core path without MLflow/Streamlit: split → run a couple of models
with GridSearch+CV → blend → threshold tune → metrics → save → load → predict.
Uses a fast subset of models and disables MLflow logging.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from phishing.core import data as data_mod
from phishing.core.metrics import compute_metrics
from phishing.core.splits import stratified_split
from phishing.core.wrapper import ModelWrapper
from phishing.experiments.runner import ThresholdConfig, run_experiments

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_sample_data import make_dataset  # noqa: E402


def _split():
    df = make_dataset(n=2000, pos_rate=0.05, seed=0)  # 5% keeps folds non-degenerate
    X, y = data_mod.split_X_y(df)
    return stratified_split(X, y, val_size=0.25, test_size=0.25, random_state=0)


def test_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert 0.0 <= m.pr_auc <= 1.0
    assert m.n_positives == 3
    assert "accuracy" not in m.as_dict()  # accuracy is never reported


def test_end_to_end_train_save_load_predict(tmp_path):
    split = _split()
    results, ensembles = run_experiments(
        split,
        model_names=["lightgbm", "logreg_woe"],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=True,
        n_splits=3,
        log_mlflow=False,
    )
    assert len(results) == 2
    assert ensembles is not None and "blend" in ensembles
    for r in results:
        # Allow tiny floating-point overshoot above 1.0 from average_precision.
        assert 0.0 <= r.val_metrics["pr_auc"] <= 1.0 + 1e-9
        assert 0.0 < r.threshold < 1.0

    # Save best by test PR-AUC, reload, and predict.
    best = max(results, key=lambda r: r.test_metrics["pr_auc"])
    version_dir = best.wrapper.save(models_dir=tmp_path, test_metrics=best.test_metrics)
    assert (version_dir / "model.joblib").exists()
    assert (version_dir / "metadata.json").exists()

    reloaded = ModelWrapper.load(version_dir)
    preds = reloaded.predict(split.X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(split.X_test)


def test_binned_woe_mode_runs():
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=["logreg_woe"],
        feature_mode="binned_woe",
        threshold_cfg=ThresholdConfig(mode="recall_target", recall_target=0.8, precision_floor=0.1),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    assert results[0].wrapper.feature_mode == "binned_woe"


import pytest


@pytest.mark.parametrize("mode", ["quantile", "target", "autoencoder", "engineered", "engineered_quantile"])
def test_alternative_encodings_run(mode):
    """Each selectable feature/encoding strategy trains and predicts end to end."""
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=["randomforest"],
        feature_mode=mode,
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    r = results[0]
    assert r.wrapper.feature_mode == mode
    preds = r.wrapper.predict(split.X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_feature_engineer_adds_columns_and_is_deterministic():
    """FeatureEngineer adds flag/log/ratio columns without using the target."""
    from phishing.features.engineering import FeatureEngineer

    df = make_dataset(n=300, pos_rate=0.1, seed=1)
    X, _ = data_mod.split_X_y(df)
    fe = FeatureEngineer(keep_raw=True)
    out1 = fe.fit_transform(X)  # no y passed -> proves it is target-free
    out2 = fe.transform(X)
    assert out1.shape[1] > X.shape[1]
    assert "has_emails" in out1.columns
    assert "lexical_diversity" in out1.columns
    assert any(c.startswith("log_") for c in out1.columns)
    assert "logz_num_email_addresses" in out1.columns  # on by default
    # New families are present: densities, per-link ratios, interactions, flags.
    for col in ("word_repetition_ratio", "link_density", "domain_per_link_ratio",
                "links_x_urgency", "content_word_ratio", "short_email",
                "high_link_density"):
        assert col in out1.columns
    # Deterministic: same input -> same output.
    pd.testing.assert_frame_equal(out1, out2)

    # Toggle off removes the logz email feature (for with/without comparisons).
    fe_off = FeatureEngineer(keep_raw=True, add_logz_email=False)
    out_off = fe_off.fit_transform(X)
    assert "logz_num_email_addresses" not in out_off.columns


def test_feature_engineer_zero_division_safe():
    """Every engineered feature is finite even for all-zero / num_words=0 rows."""
    from phishing.features.engineering import FeatureEngineer
    from phishing.core.data import FEATURES

    # Adversarial rows: all zeros, and stopwords without words.
    rows = [{c: 0 for c in FEATURES}, {**{c: 0 for c in FEATURES}, "num_stopwords": 5}]
    X = pd.DataFrame(rows)
    for eps in (1.0, 0.0):
        out = FeatureEngineer(keep_raw=False, eps=eps).fit(X).transform(X)
        assert not out.isna().any().any()
        assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_interaction_features_adds_products():
    """InteractionFeatures appends pairwise products and is deterministic."""
    from phishing.features.interactions import InteractionFeatures

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0, 1, 0]})
    ix = InteractionFeatures(keep_raw=True, max_features=3)
    out = ix.fit_transform(X)
    assert "a__x__b" in out.columns
    assert list(out["a__x__b"]) == [4, 10, 18]


def test_cost_threshold_favours_recall():
    """Cost mode with FN >> FP yields a lower threshold (higher recall) than max-F1."""
    from phishing.core.thresholding import select_threshold

    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    score = np.clip(0.2 * y + rng.normal(0, 0.3, 2000), 0, 1)
    cost = select_threshold(y, score, mode="cost", fn_cost=20.0, fp_cost=1.0)
    f1 = select_threshold(y, score, mode="max_f1")
    assert cost.mode == "cost"
    assert cost.threshold <= f1.threshold + 1e-9  # cost(FN-heavy) is not stricter


def test_stacking_meta_model_runs():
    """Stacking builds a meta-model and produces valid probabilities."""
    from phishing.core.stacking import fit_stacker, stack_proba

    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    base = np.column_stack([
        np.clip(0.3 * y + rng.normal(0, 0.2, 500), 0, 1),
        np.clip(0.4 * y + rng.normal(0, 0.2, 500), 0, 1),
    ])
    st = fit_stacker(y, base, ["m1", "m2"])
    p = stack_proba(st, base)
    assert p.shape == (500,)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", [
    "lightgbm_focal", "xgboost_focal", "catboost_focal",
    "lightgbm_focal_native", "xgboost_focal_native",
])
def test_focal_models_train_and_predict(name):
    """Each focal-loss booster (incl. native variants) trains and yields [0,1] probs."""
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=[name],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    proba = results[0].wrapper.predict_proba(split.X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_blend_model_saves_and_loads(tmp_path):
    """The winning blend can be packaged, saved, reloaded and predicted."""
    split = _split()
    results, ensembles = run_experiments(
        split,
        model_names=["lightgbm", "randomforest"],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=True,
        n_splits=3,
        log_mlflow=False,
    )
    assert "blend" in ensembles
    blend_wrapper = ensembles["blend"]["wrapper"]
    # Predicts before saving.
    proba = blend_wrapper.predict_proba(split.X_test)
    assert ((proba >= 0) & (proba <= 1)).all()
    # Round-trip through versioned persistence.
    version_dir = blend_wrapper.save(models_dir=tmp_path, test_metrics=ensembles["blend"]["test_metrics"])
    reloaded = ModelWrapper.load(version_dir)
    preds = reloaded.predict(split.X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_isotonic_calibration_runs():
    """Isotonic calibration path works end to end."""
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=["lightgbm"],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        calibration="isotonic",
        n_splits=3,
        log_mlflow=False,
    )
    assert 0.0 <= results[0].test_metrics["pr_auc"] <= 1.0 + 1e-9


# --- new algorithms: logreg, cluster, adaboost, tensorflow_dnn ---------------

@pytest.mark.parametrize("name", ["logreg", "cluster", "adaboost"])
def test_new_models_train_and_predict(name):
    """Standalone logreg, clustering and AdaBoost train and predict 0/1."""
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=[name],
        feature_mode="engineered",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
    )
    proba = results[0].wrapper.predict_proba(split.X_test)
    assert ((proba >= 0) & (proba <= 1)).all()
    preds = results[0].wrapper.predict(split.X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_tensorflow_dnn_trains_saves_loads(tmp_path):
    """The TF dense net trains, predicts valid probabilities, and round-trips."""
    from phishing.models.tensorflow_dnn_model import _TFDenseNet

    df = make_dataset(n=800, pos_rate=0.1, seed=2)
    X, y = data_mod.split_X_y(df)
    net = _TFDenseNet(optimizer="adam", learning_rate=0.05, epochs=20)
    wrapper = ModelWrapper(net, name="tensorflow_dnn", feature_mode="raw")
    wrapper.fit(X, y)
    proba = wrapper.predict_proba(X)
    assert ((proba >= 0) & (proba <= 1)).all()
    version_dir = wrapper.save(models_dir=tmp_path)
    reloaded = ModelWrapper.load(version_dir)
    preds = reloaded.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_nn_embedding_produces_default_features():
    """NNEmbedding emits embedding_dim nn_* columns (default 16) and is leakage-safe."""
    from phishing.features.nn_embedding import NNEmbedding

    df = make_dataset(n=600, pos_rate=0.1, seed=3)
    X, y = data_mod.split_X_y(df)
    emb = NNEmbedding(epochs=15, keep_raw=True)
    fitted = emb.fit(X, y)
    out = fitted.transform(X)
    nn_cols = [c for c in out.columns if c.startswith("nn_")]
    assert len(nn_cols) == 16
    assert emb.embedding_dim == 16  # the new default width


@pytest.mark.parametrize("dim,periodic", [(32, False), (32, True), (64, False)])
def test_nn_embedding_dim_and_periodic(dim, periodic):
    """Embedding width is configurable and the periodic front-end runs/serialises."""
    import pickle

    from phishing.features.nn_embedding import NNEmbedding

    df = make_dataset(n=500, pos_rate=0.1, seed=6)
    X, y = data_mod.split_X_y(df)
    emb = NNEmbedding(
        epochs=12, keep_raw=False, embedding_dim=dim, periodic=periodic,
        cosine_schedule=True, optimizer="sgd", momentum_schedule=True,
    ).fit(X, y)
    out = emb.transform(X)
    assert out.shape[1] == dim
    # Round-trips through pickle (custom periodic layer is serialisable).
    reloaded = pickle.loads(pickle.dumps(emb))
    assert np.allclose(out.values, reloaded.transform(X).values, atol=1e-4)


def test_nn_embedding_dropout_and_hidden2_and_gap():
    """Dropout and second-layer width are configurable; train/val gap is recorded."""
    from phishing.features.nn_embedding import NNEmbedding

    df = make_dataset(n=600, pos_rate=0.1, seed=7)
    X, y = data_mod.split_X_y(df)
    emb = NNEmbedding(
        epochs=15, keep_raw=False, embedding_dim=16, hidden2_dim=32, dropout=0.6,
        optimizer="sgd", momentum_schedule=True,
    ).fit(X, y)
    assert emb.transform(X).shape[1] == 16
    # Overfit-signal attributes are populated.
    assert hasattr(emb, "train_pr_auc_") and hasattr(emb, "val_pr_auc_")
    assert np.isfinite(emb.overfit_gap_)


def test_random_search_runs():
    """RandomizedSearchCV path trains and predicts end to end."""
    split = _split()
    results, _ = run_experiments(
        split,
        model_names=["lightgbm", "xgboost"],
        feature_mode="engineered",
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=False,
        n_splits=3,
        log_mlflow=False,
        search_method="random",
        n_iter=5,
    )
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.test_metrics["pr_auc"] <= 1.0 + 1e-9


def test_feature_smoothing_winsorizes_and_preserves_flags():
    """FeatureSmoothing clips outliers, keeps shape, and passes flags through."""
    from phishing.features.engineering import FeatureEngineer
    from phishing.features.smoothing import FeatureSmoothing

    df = make_dataset(n=400, pos_rate=0.1, seed=4)
    X, _ = data_mod.split_X_y(df)
    eng = FeatureEngineer().fit_transform(X)
    sm = FeatureSmoothing(method="winsor_quantile").fit(eng)
    out = sm.transform(eng)
    assert out.shape[0] == eng.shape[0]
    # Flag columns are passed through unchanged.
    if "has_emails" in eng.columns:
        assert (out["has_emails"] == eng["has_emails"]).all()


def test_denoising_autoencoder_reconstruction():
    """Denoising AE outputs one smoothed _dn column per input feature."""
    from phishing.features.autoencoder import AutoencoderEncoder

    df = make_dataset(n=500, pos_rate=0.1, seed=5)
    X, _ = data_mod.split_X_y(df)
    dae = AutoencoderEncoder(denoising=True, output="reconstruction", max_iter=60)
    out = dae.fit_transform(X)
    assert out.shape == (len(X), X.shape[1])
    assert all(c.endswith("_dn") for c in out.columns)


# --- best-params cache --------------------------------------------------------

def test_param_cache_roundtrip(tmp_path):
    """Save/load round-trips, and the key changes when feature columns change."""
    from phishing.core.param_cache import (
        load_cached_params,
        make_cache_key,
        save_cached_params,
    )

    cols = ["a", "b", "c"]
    key = make_cache_key("xgboost", "engineered", "random", 40, 5, cols)
    save_cached_params(
        "xgboost", "engineered", "random", 40, 5, cols,
        best_params={"model__max_depth": 6}, cv_pr_auc=0.42, cache_dir=tmp_path,
    )

    loaded = load_cached_params(key, cache_dir=tmp_path)
    assert loaded is not None
    assert loaded.best_params == {"model__max_depth": 6}
    assert loaded.cv_pr_auc == 0.42

    # Different training columns -> different key -> cache miss.
    other_key = make_cache_key("xgboost", "engineered", "random", 40, 5, ["a", "b", "d"])
    assert other_key != key
    assert load_cached_params(other_key, cache_dir=tmp_path) is None

    # Unreadable/missing file -> None, not an exception.
    assert load_cached_params("does-not-exist", cache_dir=tmp_path) is None


def test_run_model_uses_cached_params_on_second_call(tmp_path, monkeypatch):
    """A second run_model call with the same config hits the cache and skips search."""
    import phishing.core.param_cache as param_cache_mod
    import phishing.experiments.runner as runner_mod
    from phishing.experiments.runner import run_model

    # Redirect the cache to an isolated tmp dir for this test.
    monkeypatch.setattr(param_cache_mod, "CACHE_DIR", tmp_path)

    split = _split()
    thr = ThresholdConfig(mode="max_f1")

    first = run_model(
        "lightgbm", split, feature_mode="raw", threshold_cfg=thr,
        n_splits=3, log_mlflow=False, search_method="random", n_iter=3,
    )
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1, "first call should search and cache the winner"

    # Spy on RandomizedSearchCV.fit to prove the second call never searches.
    calls = []
    original_init = runner_mod.RandomizedSearchCV.__init__

    def spy_init(self, *args, **kwargs):
        calls.append(1)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(runner_mod.RandomizedSearchCV, "__init__", spy_init)

    second = run_model(
        "lightgbm", split, feature_mode="raw", threshold_cfg=thr,
        n_splits=3, log_mlflow=False, search_method="random", n_iter=3,
    )
    assert calls == [], "second call must not construct a new search"
    assert second.best_params == first.best_params
    assert list(tmp_path.glob("*.json")) == cache_files, "no duplicate cache file written"


def test_run_model_force_search_ignores_cache(tmp_path, monkeypatch):
    """force_search=True re-runs the search even when a cached winner exists."""
    import phishing.core.param_cache as param_cache_mod
    from phishing.experiments.runner import run_model

    monkeypatch.setattr(param_cache_mod, "CACHE_DIR", tmp_path)

    split = _split()
    thr = ThresholdConfig(mode="max_f1")

    run_model(
        "lightgbm", split, feature_mode="raw", threshold_cfg=thr,
        n_splits=3, log_mlflow=False, search_method="random", n_iter=3,
    )
    assert len(list(tmp_path.glob("*.json"))) == 1

    result = run_model(
        "lightgbm", split, feature_mode="raw", threshold_cfg=thr,
        n_splits=3, log_mlflow=False, search_method="random", n_iter=3,
        force_search=True,
    )
    # force_search still re-caches the (possibly identical) winner; no crash,
    # and the result is a normal, fully-formed ModelResult.
    assert 0.0 <= result.test_metrics["pr_auc"] <= 1.0 + 1e-9


# --- trained-embedding cache --------------------------------------------------

def test_embedding_cache_roundtrip(tmp_path):
    """A fitted NNEmbedding round-trips through the cache and predicts identically."""
    from phishing.core.embedding_cache import (
        load_cached_embedding,
        make_embedding_cache_key,
        save_cached_embedding,
    )
    from phishing.features.nn_embedding import NNEmbedding

    df = make_dataset(n=400, pos_rate=0.1, seed=8)
    X, y = data_mod.split_X_y(df)
    cols = list(X.columns)

    emb = NNEmbedding(epochs=12, keep_raw=False, embedding_dim=8).fit(X, y)
    key = make_embedding_cache_key(
        embedding_dim=8, hidden1_dim=40, hidden2_dim=20, dropout1=0.4, dropout2=0.4,
        patience=50, optimizer="adam", learning_rate=0.05, periodic=False,
        cosine_schedule=False, feature_columns=cols, n_train_rows=len(X),
    )
    save_cached_embedding(
        emb, embedding_dim=8, hidden1_dim=40, hidden2_dim=20, dropout1=0.4,
        dropout2=0.4, patience=50, optimizer="adam", learning_rate=0.05,
        periodic=False, cosine_schedule=False, feature_columns=cols,
        n_train_rows=len(X), train_pr_auc=emb.train_pr_auc_,
        val_pr_auc=emb.val_pr_auc_, n_epochs_trained=emb.n_epochs_trained_,
        cache_dir=tmp_path,
    )

    loaded = load_cached_embedding(key, cache_dir=tmp_path)
    assert loaded is not None
    cached_emb, meta = loaded
    assert meta.embedding_dim == 8
    assert meta.n_train_rows == len(X)

    out_original = emb.transform(X)
    out_cached = cached_emb.transform(X)
    np.testing.assert_allclose(out_original.values, out_cached.values, atol=1e-5)

    # Different row count -> different key -> miss.
    other_key = make_embedding_cache_key(
        embedding_dim=8, hidden1_dim=40, hidden2_dim=20, dropout1=0.4, dropout2=0.4,
        patience=50, optimizer="adam", learning_rate=0.05, periodic=False,
        cosine_schedule=False, feature_columns=cols, n_train_rows=len(X) + 1,
    )
    assert other_key != key
    assert load_cached_embedding(other_key, cache_dir=tmp_path) is None

    # Missing cache -> None, not an exception.
    assert load_cached_embedding("does-not-exist", cache_dir=tmp_path) is None
