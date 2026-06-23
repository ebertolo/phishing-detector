"""CLI: run the full experiment suite on a labelled CSV.

Performs the project's stratified 95/5 split (5% validation, the rest split into
train/test, target distribution preserved across all three), runs the selected
models through GridSearch + StratifiedKFold, builds an optional blend, tunes the
decision threshold, prints an imbalance-aware comparison, logs to MLflow, and
optionally saves the best model version.

Examples
--------
    uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv
    uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
        --models lightgbm catboost --feature-mode binned_woe \
        --threshold-mode recall_target --recall-target 0.9 --precision-floor 0.3 \
        --save-best
"""

# %%
from __future__ import annotations

import argparse
import sys

import pandas as pd

from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split, stratified_sample
from phishing.experiments.runner import ThresholdConfig, run_experiments
from phishing.models import ALL_MODELS, DEFAULT_MODELS
from phishing.models._common import FEATURE_MODES


# %%
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments",
        description="Run phishing-detection experiments on a labelled CSV.",
    )
    p.add_argument("--csv", required=True, help="Path to a labelled input CSV.")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(ALL_MODELS.keys()),
        help="Models to run (default: the recommended boosters "
        f"{DEFAULT_MODELS}). Pass names to override.",
    )
    p.add_argument(
        "--feature-mode",
        default="engineered",
        choices=FEATURE_MODES,
        help="Feature/encoding strategy. Default 'engineered' (flags + log + "
        "ratios). Combine with an encoding via 'engineered_<enc>', or use a bare "
        "encoding (raw | binned_woe | quantile | target | autoencoder).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Down-sample to ~N rows (stratified, preserves phishing rate) for "
        "faster runs. Omit to use the full dataset.",
    )
    p.add_argument("--cv-folds", type=int, default=5, help="StratifiedKFold splits.")
    p.add_argument(
        "--search", default="grid", choices=["grid", "random"],
        help="Hyperparameter search: grid (default) or random (wider space).",
    )
    p.add_argument(
        "--n-iter", type=int, default=40,
        help="RandomizedSearchCV iterations when --search random.",
    )
    p.add_argument(
        "--force-search",
        action="store_true",
        help="Ignore any cached best-params (best_params/*.json) and re-run the "
        "hyperparameter search even if a matching cached winner exists.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.05, help="Validation fraction (default 5%%)."
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Test fraction within the remaining 95%% (default 20%%).",
    )
    p.add_argument("--no-blend", action="store_true", help="Skip the blend.")
    p.add_argument(
        "--stacking", action="store_true",
        help="Also build a logistic stacking meta-model over the base models.",
    )
    p.add_argument(
        "--calibration", default="sigmoid", choices=["sigmoid", "isotonic"],
        help="Probability calibration method (default sigmoid).",
    )
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    p.add_argument(
        "--threshold-mode",
        default="max_f1",
        choices=["recall_target", "max_f1", "manual", "cost"],
    )
    p.add_argument("--recall-target", type=float, default=0.90)
    p.add_argument("--precision-floor", type=float, default=0.30)
    p.add_argument("--manual-threshold", type=float, default=0.5)
    p.add_argument(
        "--fn-cost", type=float, default=10.0,
        help="Cost-sensitive mode: false-negative weight (default 10x FP).",
    )
    p.add_argument("--fp-cost", type=float, default=1.0,
                   help="Cost-sensitive mode: false-positive weight.")
    p.add_argument(
        "--save-best",
        action="store_true",
        help="Persist the best model (by test PR-AUC) as a versioned artifact.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


# %%
def _ensemble_row(name, info) -> dict:
    """One comparison-table row for an ensemble (blend/stacking)."""
    return {
        "model": name,
        "cv_pr_auc": float("nan"),
        "val_pr_auc": round(info["val_metrics"]["pr_auc"], 4),
        "test_pr_auc": round(info["test_metrics"]["pr_auc"], 4),
        "test_recall": round(info["test_metrics"]["recall"], 4),
        "test_precision": round(info["test_metrics"]["precision"], 4),
        "test_f1": round(info["test_metrics"]["f1"], 4),
        "test_mcc": round(info["test_metrics"]["mcc"], 4),
        "threshold": round(info["threshold"], 4),
    }


# %%
def _results_table(results, ensembles) -> pd.DataFrame:
    """Build the imbalance-aware comparison table (sorted by test PR-AUC)."""
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "cv_pr_auc": round(r.cv_pr_auc, 4),
                "val_pr_auc": round(r.val_metrics["pr_auc"], 4),
                "test_pr_auc": round(r.test_metrics["pr_auc"], 4),
                "test_recall": round(r.test_metrics["recall"], 4),
                "test_precision": round(r.test_metrics["precision"], 4),
                "test_f1": round(r.test_metrics["f1"], 4),
                "test_mcc": round(r.test_metrics["mcc"], 4),
                "threshold": round(r.threshold, 4),
            }
        )
    for ens_name, info in (ensembles or {}).items():
        rows.append(_ensemble_row(ens_name, info))
    return pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)


# %%
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    df = data_mod.load_csv(args.csv)
    if not data_mod.has_labels(df):
        print(
            f"ERROR: {args.csv} has no '{data_mod.TARGET}' column — experiments "
            "require a labelled dataset.",
            file=sys.stderr,
        )
        return 2

    print(f"Loaded {len(df):,} rows from {args.csv}")
    if args.sample is not None:
        df = stratified_sample(df, args.sample, random_state=args.seed)
        print(f"Stratified sample -> {len(df):,} rows (rate {df[data_mod.TARGET].mean():.4%})")
    split = stratified_95_5_split(
        df,
        val_fraction=args.val_fraction,
        test_fraction_within_95=args.test_fraction,
        random_state=args.seed,
    )
    rates = split.positive_rates()
    print(
        "Stratified split — "
        f"train={len(split.y_train):,} ({rates['train']:.4%}), "
        f"test={len(split.y_test):,} ({rates['test']:.4%}), "
        f"val={len(split.y_val):,} ({rates['val']:.4%})"
    )

    thr_cfg = ThresholdConfig(
        mode=args.threshold_mode,
        recall_target=args.recall_target,
        precision_floor=args.precision_floor,
        manual_value=args.manual_threshold,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
    )

    results, ensembles = run_experiments(
        split,
        model_names=args.models,
        feature_mode=args.feature_mode,
        threshold_cfg=thr_cfg,
        build_blend=not args.no_blend,
        build_stacking=args.stacking,
        calibration=args.calibration,
        n_splits=args.cv_folds,
        log_mlflow=not args.no_mlflow,
        search_method=args.search,
        n_iter=args.n_iter,
        force_search=args.force_search,
        progress=lambda m: print(m, flush=True),
    )

    table = _results_table(results, ensembles)
    print("\n=== Comparison (sorted by test PR-AUC; accuracy intentionally omitted) ===")
    print(table.to_string(index=False))
    if ensembles and "blend" in ensembles:
        b = ensembles["blend"]
        weights = dict(zip(b["names"], [round(w, 3) for w in b["weights"]]))
        print(f"\nBlend weights: {weights}")
    if ensembles and "stacking" in ensembles:
        print(f"Stacking meta-coefficients: {ensembles['stacking']['meta_coefficients']}")

    if args.save_best:
        # Candidates: each single model, plus the blend (which carries a
        # serializable wrapper). Pick the highest test PR-AUC overall.
        candidates = [
            (r.name, r.test_metrics["pr_auc"], r.wrapper, r.test_metrics, r.val_metrics)
            for r in results
        ]
        if ensembles and "blend" in ensembles:
            b = ensembles["blend"]
            candidates.append(
                ("blend", b["test_metrics"]["pr_auc"], b["wrapper"],
                 b["test_metrics"], b["val_metrics"])
            )
        name, _, wrapper, test_metrics, val_metrics = max(candidates, key=lambda c: c[1])
        path = wrapper.save(val_metrics=val_metrics, test_metrics=test_metrics)
        print(f"\nSaved best model ({name}) -> {path}")

    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
