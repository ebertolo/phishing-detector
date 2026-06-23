"""General command-line interface for the phishing-detection framework.

The system is fully usable headless: identify the action (``train`` or
``infer``) and pass the CSV on the command line. Mirrors the Streamlit app's
two run modes — when an inference input carries the ``label_1`` column, metrics
are computed (evaluation mode); otherwise predictions only (inference mode).

Examples
--------
    # Train one model on the real data and save a versioned artifact
    uv run python scripts/cli.py train \
        --csv data/email_phishing_data.csv --model lightgbm \
        --feature-mode raw --threshold-mode recall_target \
        --recall-target 0.9 --precision-floor 0.3

    # Predict on a CSV using the latest saved version, write predictions
    uv run python scripts/cli.py infer \
        --csv data/email_phishing_data.csv --out predictions.csv

    # Predict using a specific saved version directory
    uv run python scripts/cli.py infer \
        --csv new_emails.csv --version models/lightgbm__20260619T230000Z
"""

# %%
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split
from phishing.core.metrics import compute_metrics
from phishing.core.persistence import list_versions
from phishing.core.wrapper import ModelWrapper
from phishing.experiments.runner import ThresholdConfig, run_model
from phishing.models import ALL_MODELS
from phishing.models._common import FEATURE_MODES
from phishing.observability import tracking


# %%
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phishing-cli",
        description="Train models or run inference on a CSV, headless via uv.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- train -------------------------------------------------------------
    t = sub.add_parser("train", help="Train a single model and save a version.")
    t.add_argument("--csv", required=True, help="Labelled training CSV.")
    t.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    t.add_argument("--feature-mode", default="engineered", choices=FEATURE_MODES)
    t.add_argument("--cv-folds", type=int, default=5)
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--test-fraction", type=float, default=0.20)
    t.add_argument(
        "--threshold-mode",
        default="max_f1",
        choices=["recall_target", "max_f1", "manual", "cost"],
    )
    t.add_argument("--recall-target", type=float, default=0.90)
    t.add_argument("--precision-floor", type=float, default=0.30)
    t.add_argument("--manual-threshold", type=float, default=0.5)
    t.add_argument("--fn-cost", type=float, default=10.0,
                   help="Cost mode: false-negative weight (default 10x FP).")
    t.add_argument("--fp-cost", type=float, default=1.0,
                   help="Cost mode: false-positive weight.")
    t.add_argument("--no-mlflow", action="store_true")
    t.add_argument("--no-save", action="store_true", help="Train but do not persist.")
    t.add_argument(
        "--force-search",
        action="store_true",
        help="Ignore any cached best-params (best_params/*.json) and re-run the "
        "hyperparameter search even if a matching cached winner exists.",
    )
    t.add_argument("--seed", type=int, default=42)

    # --- infer -------------------------------------------------------------
    i = sub.add_parser("infer", help="Predict over a CSV with a saved version.")
    i.add_argument("--csv", required=True, help="Input CSV (labelled or not).")
    i.add_argument(
        "--version",
        default=None,
        help="Saved version directory. Defaults to the most recent one.",
    )
    i.add_argument("--out", default="predictions.csv", help="Output predictions CSV.")
    i.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the version's saved decision threshold.",
    )
    i.add_argument("--no-mlflow", action="store_true")
    return p


# %%
def _require_labelled(df: pd.DataFrame, path: str) -> None:
    if not data_mod.has_labels(df):
        print(
            f"ERROR: {path} has no '{data_mod.TARGET}' column — training needs labels.",
            file=sys.stderr,
        )
        raise SystemExit(2)


# %%
def cmd_train(args) -> int:
    df = data_mod.load_csv(args.csv)
    _require_labelled(df, args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    split = stratified_95_5_split(
        df,
        val_fraction=args.val_fraction,
        test_fraction_within_95=args.test_fraction,
        random_state=args.seed,
    )
    rates = split.positive_rates()
    print(
        f"Stratified split — train={len(split.y_train):,} ({rates['train']:.4%}), "
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

    print(f"Training {args.model} (feature_mode={args.feature_mode}) ...", flush=True)
    result = run_model(
        args.model,
        split,
        feature_mode=args.feature_mode,
        threshold_cfg=thr_cfg,
        n_splits=args.cv_folds,
        log_mlflow=not args.no_mlflow,
        force_search=args.force_search,
    )

    print("\n=== Metrics (test; accuracy intentionally omitted) ===")
    for key in ("pr_auc", "roc_auc", "recall", "precision", "f1", "mcc"):
        print(f"  {key:>10}: {result.test_metrics[key]:.4f}")
    print(f"  threshold: {result.threshold:.4f} (mode={args.threshold_mode})")

    if not args.no_save:
        path = result.wrapper.save(
            cv_pr_auc=result.cv_pr_auc,
            val_metrics=result.val_metrics,
            test_metrics=result.test_metrics,
            best_params=result.best_params,
        )
        print(f"\nSaved model -> {path}")
    return 0


# %%
def cmd_infer(args) -> int:
    if args.version is not None:
        version_dir = args.version
    else:
        versions = list_versions()
        if not versions:
            print("ERROR: no saved versions found. Train one first.", file=sys.stderr)
            return 2
        version_dir = versions[0]["path"]
        print(f"Using latest version: {version_dir}")

    wrapper = ModelWrapper.load(version_dir)
    threshold = args.threshold if args.threshold is not None else wrapper.threshold

    df = data_mod.load_csv(args.csv)
    X = data_mod.feature_frame(df)
    proba = wrapper.predict_proba(X)
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["phishing_proba"] = proba
    out["phishing_pred"] = pred
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} predictions -> {args.out}")
    print(f"Predicted positive rate: {pred.mean():.4%} (threshold={threshold:.4f})")

    eval_metrics = None
    if data_mod.has_labels(df):
        _, y = data_mod.split_X_y(df)
        m = compute_metrics(np.asarray(y), proba, threshold)
        eval_metrics = m.as_dict()
        print("\n=== Evaluation metrics (labels present) ===")
        for key in ("pr_auc", "roc_auc", "recall", "precision", "f1", "mcc"):
            print(f"  {key:>10}: {eval_metrics[key]:.4f}")
        print(f"  confusion [[tn, fp],[fn, tp]]: {m.confusion.tolist()}")

    if not args.no_mlflow:
        version_name = str(version_dir).replace("\\", "/").rstrip("/").split("/")[-1]
        try:
            tracking.log_inference(
                model_version=version_name,
                n_samples=len(out),
                threshold=threshold,
                predicted_positive_rate=float(pred.mean()),
                eval_metrics=eval_metrics,
            )
            print("Logged batch summary to MLflow.")
        except Exception as exc:  # logging must not block predictions
            print(f"MLflow logging skipped: {exc}", file=sys.stderr)
    return 0


# %%
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "infer":
        return cmd_infer(args)
    return 1


# %%
if __name__ == "__main__":
    raise SystemExit(main())
