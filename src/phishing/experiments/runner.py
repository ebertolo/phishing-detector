"""Experiment runner: GridSearchCV over StratifiedKFold for each model.

For every selected model the runner:

1. builds its pipeline for the chosen ``feature_mode`` (raw or binned_woe),
2. runs ``GridSearchCV`` over ``StratifiedKFold`` scored by PR-AUC (refit metric),
3. calibrates the best estimator's probabilities on the validation set,
4. tunes the decision threshold on validation per the chosen mode,
5. computes validation and test metrics,
6. logs everything to MLflow (params, CV/val/test metrics, curves, WOE tables).

An optional blend of the calibrated base models is built on the validation set
and flows through the same threshold/metrics/logging path. The runner is pure
core logic — no Streamlit/FastAPI — and is independently runnable as cells.
"""

# %%
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..core.blend_model import BlendModel
from ..core.blending import blend_proba, optimize_weights
from ..core.metrics import PRIMARY_SCORING, compute_metrics
from ..core.param_cache import load_cached_params, make_cache_key, save_cached_params
from ..core.splits import DataSplit, make_cv
from ..core.stacking import fit_stacker, stack_proba
from ..core.thresholding import select_threshold
from ..core.wrapper import ModelWrapper
from ..models import ALL_MODELS
from ..models._common import gpu_available, lightgbm_cuda_available
from ..observability import plots, tracking

# Boosters that switch to GPU training inside their build() when GPU support is
# actually usable (see _common.gpu_available / lightgbm_cuda_available).
_GPU_CHECK = {
    "lightgbm": lightgbm_cuda_available,
    "xgboost": gpu_available,
    "catboost": gpu_available,
}


# %%
@dataclass
class ThresholdConfig:
    """How the operating point is chosen on the validation set."""

    mode: str = "max_f1"            # "recall_target" | "max_f1" | "manual" | "cost"
    recall_target: float = 0.90
    precision_floor: float = 0.30
    manual_value: float = 0.5
    fn_cost: float = 10.0          # cost-sensitive: false-negative weight
    fp_cost: float = 1.0           # cost-sensitive: false-positive weight


# %%
@dataclass
class ModelResult:
    """Outcome for one model: tuned wrapper, scores and metric dicts."""

    name: str
    wrapper: ModelWrapper
    best_params: dict[str, Any]
    cv_pr_auc: float
    threshold: float
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    val_scores: np.ndarray = field(repr=False, default=None)
    mlflow_run_id: str | None = None


# %%
def _calibrate(best_estimator, split: DataSplit, method: str = "sigmoid"):
    """Calibrate the fitted best estimator's probabilities on validation.

    The already-tuned model is frozen (``FrozenEstimator``) so calibration only
    fits the probability map on the held-out validation set — the modern
    replacement for the removed ``cv="prefit"`` option. This keeps the operating
    point and blend running on trustworthy probabilities.

    ``method`` selects the calibration map: ``"sigmoid"`` (Platt scaling, robust
    with little data) or ``"isotonic"`` (non-parametric, more flexible but needs
    more validation samples).
    """
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(best_estimator), method=method
    )
    calibrated.fit(split.X_val, split.y_val)
    return calibrated


# %%
def run_model(
    name: str,
    split: DataSplit,
    feature_mode: str,
    threshold_cfg: ThresholdConfig,
    n_splits: int = 5,
    log_mlflow: bool = True,
    calibration: str = "sigmoid",
    search_method: str = "grid",
    n_iter: int = 40,
    search_verbose: int = 0,
    use_param_cache: bool = True,
    force_search: bool = False,
    embedding_kwargs: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelResult:
    """Train, calibrate, threshold-tune and evaluate a single model.

    ``search_method`` selects the hyperparameter search: ``"grid"`` (exhaustive
    ``param_grid()``) or ``"random"`` (``RandomizedSearchCV`` sampling ``n_iter``
    points from ``param_distributions()`` if the model defines it, else from the
    grid). Randomized search covers a wider space at a fixed budget.
    ``search_verbose`` is passed to the search (>=1 prints fold/candidate progress).

    ``use_param_cache`` (default ``True``) looks up a previously-found search
    winner in ``best_params/`` (see ``core.param_cache``), keyed by model name,
    feature mode, search config and the exact training columns. On a cache hit the
    search is skipped entirely and the pipeline is fit once with the cached
    parameters; on a miss (or ``force_search=True``) the full search runs as
    before and its winner is cached for next time.

    ``embedding_kwargs`` overrides the NN embedding's architecture (e.g.
    ``embedding_dim``, ``hidden1_dim``, ``hidden2_dim``, ``dropout1``,
    ``dropout2``, ``patience``) when ``feature_mode`` ends in ``nnembed``;
    ignored otherwise. Note this also changes the param-cache key, since it
    changes ``feature_mode``'s effective pipeline (the embedding is trained
    fresh inside every CV fold for this mode — see ``docs/EXPERIMENT_JOURNEY.md``
    §5 on why a frozen, once-trained embedding is the leakage-safe alternative
    used by ``scripts/best_model_report.py``).
    """
    module = ALL_MODELS[name]
    pipeline = module.build(
        feature_mode=feature_mode, y=split.y_train, embedding_kwargs=embedding_kwargs
    )
    feature_columns = list(split.X_train.columns)

    cache_key = make_cache_key(
        name, feature_mode, search_method, n_iter, n_splits, feature_columns,
        embedding_kwargs,
    )
    cached = None if force_search else (
        load_cached_params(cache_key) if use_param_cache else None
    )

    if cached is not None:
        if progress:
            progress(f"{name}: using cached best params (cache_key={cache_key})")
        best_estimator = pipeline.set_params(**cached.best_params)
        best_estimator.fit(split.X_train, split.y_train)
        best_params = cached.best_params
        cv_pr_auc = cached.cv_pr_auc
    else:
        # A GPU is a single shared device: running several CV candidates as
        # parallel CPU processes (joblib's default for n_jobs=-1) makes them
        # fight over one GPU context instead of actually parallelising, which
        # is slower than CPU. Force sequential search (one candidate's GPU fit
        # at a time) for GPU-capable boosters when a GPU is present; CPU-only
        # models keep full CPU parallelism.
        gpu_check = _GPU_CHECK.get(name)
        search_n_jobs = 1 if (gpu_check is not None and gpu_check()) else -1
        # Allow forcing sequential search via env var. On Windows, joblib/loky's
        # parallel-worker teardown (n_jobs=-1) can trigger a native
        # StackOverflowException after the search completes — set
        # PHISHING_SEARCH_N_JOBS=1 (the test suite does this) to avoid it.
        n_jobs_override = os.environ.get("PHISHING_SEARCH_N_JOBS")
        if n_jobs_override:
            search_n_jobs = int(n_jobs_override)

        if search_method == "random":
            space = getattr(module, "param_distributions", module.param_grid)()
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=space,
                n_iter=n_iter,
                scoring=PRIMARY_SCORING,
                cv=make_cv(n_splits=n_splits),
                refit=True,
                n_jobs=search_n_jobs,
                random_state=42,
                verbose=search_verbose,
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=module.param_grid(),
                scoring=PRIMARY_SCORING,
                cv=make_cv(n_splits=n_splits),
                refit=True,
                n_jobs=search_n_jobs,
                verbose=search_verbose,
            )
        search.fit(split.X_train, split.y_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        cv_pr_auc = float(search.best_score_)
        if use_param_cache:
            save_cached_params(
                name, feature_mode, search_method, n_iter, n_splits,
                feature_columns, best_params, cv_pr_auc,
                embedding_kwargs=embedding_kwargs,
            )

    calibrated = _calibrate(best_estimator, split, method=calibration)
    wrapper = ModelWrapper(
        calibrated, name=name, feature_mode=feature_mode, threshold=0.5
    )
    wrapper.feature_names_ = list(split.X_train.columns)

    val_scores = wrapper.predict_proba(split.X_val)
    wrapper.set_threshold(
        np.asarray(split.y_val),
        val_scores,
        mode=threshold_cfg.mode,
        recall_target=threshold_cfg.recall_target,
        precision_floor=threshold_cfg.precision_floor,
        manual_value=threshold_cfg.manual_value,
        fn_cost=threshold_cfg.fn_cost,
        fp_cost=threshold_cfg.fp_cost,
    )

    val_metrics = wrapper.validate(split.X_val, split.y_val)
    test_metrics = wrapper.validate(split.X_test, split.y_test)

    run_id = None
    if log_mlflow:
        run_id = _log_fit_run(
            name, best_estimator, best_params, cv_pr_auc, wrapper, split, feature_mode
        )

    return ModelResult(
        name=name,
        wrapper=wrapper,
        best_params=best_params,
        cv_pr_auc=cv_pr_auc,
        threshold=wrapper.threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_scores=val_scores,
        mlflow_run_id=run_id,
    )


# %%
def _log_fit_run(
    name, best_estimator, best_params, cv_pr_auc, wrapper, split, feature_mode
) -> str:
    """Push one model's training run to MLflow with curves and WOE tables."""
    test_scores = wrapper.predict_proba(split.X_test)
    figures = {
        "pr_curve": plots.pr_curve_figure(split.y_test, test_scores),
        "roc_curve": plots.roc_curve_figure(split.y_test, test_scores),
        "confusion": plots.confusion_figure(
            compute_metrics(np.asarray(split.y_test), test_scores, wrapper.threshold).confusion
        ),
    }
    woe_tables = _extract_woe_tables(best_estimator)
    return tracking.log_fit(
        model_name=name,
        params={**best_params, "feature_mode": feature_mode},
        cv_metrics={"pr_auc": cv_pr_auc},
        val_metrics=wrapper.validate(split.X_val, split.y_val),
        test_metrics=wrapper.validate(split.X_test, split.y_test),
        model=best_estimator,
        figures=figures,
        woe_tables=woe_tables,
        tags={"feature_mode": feature_mode},
    )


# %%
def _extract_woe_tables(estimator) -> dict[str, pd.DataFrame]:
    """Pull binning/WOE tables from a pipeline that has a WOE front-end."""
    try:
        woe_step = estimator.named_steps.get("woe")
    except AttributeError:
        return {}
    if woe_step is None:
        return {}
    try:
        return woe_step.binning_tables()
    except Exception:  # pragma: no cover - defensive
        return {}


# %%
def run_experiments(
    split: DataSplit,
    model_names: list[str],
    feature_mode: str = "raw",
    threshold_cfg: ThresholdConfig | None = None,
    build_blend: bool = True,
    build_stacking: bool = False,
    calibration: str = "sigmoid",
    n_splits: int = 5,
    log_mlflow: bool = True,
    progress: Callable[[str], None] | None = None,
    search_method: str = "grid",
    n_iter: int = 40,
    search_verbose: int = 0,
    use_param_cache: bool = True,
    force_search: bool = False,
    embedding_kwargs: dict | None = None,
) -> tuple[list[ModelResult], dict[str, Any] | None]:
    """Run all selected models and (optionally) a blend and/or a stacking ensemble.

    Returns ``(results, ensembles)`` where ``ensembles`` is a dict that may hold
    ``"blend"`` and/or ``"stacking"`` entries (or ``None`` if neither is built).
    ``calibration`` is ``"sigmoid"`` or ``"isotonic"``; ``search_method`` is
    ``"grid"`` or ``"random"`` (with ``n_iter`` samples). ``progress`` is an
    optional callback invoked with status strings (per model, with index/total).
    ``use_param_cache``/``force_search`` control the best-params cache (see
    ``run_model`` / ``core.param_cache``). ``embedding_kwargs`` overrides the NN
    embedding's architecture when ``feature_mode`` ends in ``nnembed``.
    """
    threshold_cfg = threshold_cfg or ThresholdConfig()
    results: list[ModelResult] = []

    total = len(model_names)
    for i, name in enumerate(model_names, start=1):
        if progress:
            fits = (n_iter if search_method == "random" else None)
            detail = f"{search_method} search" + (f", {fits} iters" if fits else "")
            progress(f"[{i}/{total}] training {name} ({detail}, {n_splits}-fold CV) ...")
        results.append(
            run_model(
                name, split, feature_mode, threshold_cfg, n_splits, log_mlflow,
                calibration=calibration, search_method=search_method, n_iter=n_iter,
                search_verbose=search_verbose, use_param_cache=use_param_cache,
                force_search=force_search, embedding_kwargs=embedding_kwargs,
                progress=progress,
            )
        )
        if progress:
            progress(f"[{i}/{total}] {name} done — test PR-AUC {results[-1].test_metrics['pr_auc']:.4f}")

    ensembles: dict[str, Any] = {}
    if build_blend and len(results) >= 2:
        if progress:
            progress("Building blend ...")
        ensembles["blend"] = _build_blend(results, split, threshold_cfg, log_mlflow)
    if build_stacking and len(results) >= 2:
        if progress:
            progress("Building stacking ...")
        ensembles["stacking"] = _build_stacking(results, split, threshold_cfg, log_mlflow)

    return results, (ensembles or None)


# %%
def _build_blend(results, split, threshold_cfg, log_mlflow) -> dict[str, Any]:
    """Optimise blend weights on validation and evaluate on test."""
    names = [r.name for r in results]
    val_matrix = np.column_stack([r.val_scores for r in results])
    blend = optimize_weights(np.asarray(split.y_val), val_matrix, names)

    val_blend_scores = blend_proba(val_matrix, blend.weights)
    thr = select_threshold(
        np.asarray(split.y_val),
        val_blend_scores,
        mode=threshold_cfg.mode,
        recall_target=threshold_cfg.recall_target,
        precision_floor=threshold_cfg.precision_floor,
        manual_value=threshold_cfg.manual_value,
        fn_cost=threshold_cfg.fn_cost,
        fp_cost=threshold_cfg.fp_cost,
    )

    test_matrix = np.column_stack(
        [r.wrapper.predict_proba(split.X_test) for r in results]
    )
    test_blend_scores = blend_proba(test_matrix, blend.weights)
    val_metrics = compute_metrics(np.asarray(split.y_val), val_blend_scores, thr.threshold).as_dict()
    test_metrics = compute_metrics(np.asarray(split.y_test), test_blend_scores, thr.threshold).as_dict()

    # Package the blend as a single serializable estimator wrapped in a
    # ModelWrapper, so the winning blend can be saved/loaded like any model.
    blend_estimator = BlendModel(
        estimators=[r.wrapper.estimator for r in results],
        weights=blend.weights,
        names=names,
    )
    blend_wrapper = ModelWrapper(
        blend_estimator, name="blend",
        feature_mode=results[0].wrapper.feature_mode,
        threshold=thr.threshold,
    )
    blend_wrapper.threshold_mode = thr.mode
    blend_wrapper.feature_names_ = list(split.X_train.columns)

    info = {
        "names": names,
        "weights": blend.weights.tolist(),
        "threshold": thr.threshold,
        "threshold_mode": thr.mode,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "wrapper": blend_wrapper,
    }
    if log_mlflow:
        # Log the same diagnostic curves the individual models log, computed on
        # the blend's held-out test scores (parity with _log_fit_run).
        figures = {
            "pr_curve": plots.pr_curve_figure(split.y_test, test_blend_scores),
            "roc_curve": plots.roc_curve_figure(split.y_test, test_blend_scores),
            "confusion": plots.confusion_figure(
                compute_metrics(
                    np.asarray(split.y_test), test_blend_scores, thr.threshold
                ).confusion
            ),
        }
        info["mlflow_run_id"] = tracking.log_fit(
            model_name="blend",
            params={"weights": dict(zip(names, blend.weights.tolist()))},
            cv_metrics={"pr_auc": float(blend.val_pr_auc)},
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            figures=figures,
            tags={"kind": "blend"},
        )
    return info


# %%
def _threshold_from(threshold_cfg: ThresholdConfig, y_val, val_scores):
    """Select an operating threshold for an ensemble's validation scores."""
    return select_threshold(
        np.asarray(y_val),
        val_scores,
        mode=threshold_cfg.mode,
        recall_target=threshold_cfg.recall_target,
        precision_floor=threshold_cfg.precision_floor,
        manual_value=threshold_cfg.manual_value,
        fn_cost=threshold_cfg.fn_cost,
        fp_cost=threshold_cfg.fp_cost,
    )


# %%
def _build_stacking(results, split, threshold_cfg, log_mlflow) -> dict[str, Any]:
    """Fit a logistic meta-model over base validation probabilities (stacking)."""
    names = [r.name for r in results]
    val_matrix = np.column_stack([r.val_scores for r in results])
    stacker = fit_stacker(np.asarray(split.y_val), val_matrix, names)

    val_stack_scores = stack_proba(stacker, val_matrix)
    thr = _threshold_from(threshold_cfg, split.y_val, val_stack_scores)

    test_matrix = np.column_stack(
        [r.wrapper.predict_proba(split.X_test) for r in results]
    )
    test_stack_scores = stack_proba(stacker, test_matrix)
    val_metrics = compute_metrics(np.asarray(split.y_val), val_stack_scores, thr.threshold).as_dict()
    test_metrics = compute_metrics(np.asarray(split.y_test), test_stack_scores, thr.threshold).as_dict()

    coefs = dict(zip(names, stacker.meta_model.coef_.ravel().round(4).tolist()))
    info = {
        "names": names,
        "meta_coefficients": coefs,
        "threshold": thr.threshold,
        "threshold_mode": thr.mode,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if log_mlflow:
        info["mlflow_run_id"] = tracking.log_fit(
            model_name="stacking",
            params={"meta_coefficients": coefs},
            cv_metrics={"pr_auc": float(val_metrics["pr_auc"])},
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            tags={"kind": "stacking"},
        )
    return info
