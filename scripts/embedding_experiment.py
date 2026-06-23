"""Pre-train the NN embedding once, then evaluate boosters that reuse it.

The earlier ``engineered_nnembed`` feature mode retrained the network inside every
CV fold — correct but very slow. This script instead:

1. builds the engineered features and does the stratified 95/5 split;
2. trains the NN embedding **once on the training split only** (leakage-safe:
   the embedding never sees val/test), using SGD with a small learning rate and
   a momentum schedule, reporting how many epochs early stopping used;
3. saves the embedding model as a versioned artifact;
4. pre-computes the 20 ``nn_*`` embedding features for every split and appends
   them to the engineered features;
5. runs the boosters + blend on the combined features (``feature_mode="raw"``,
   so nothing is retrained per fold), and prints the comparison.

A ``--no-embedding`` run gives the matching engineered-only baseline.

Examples
--------
    uv run python scripts/embedding_experiment.py --csv data/email_phishing_data.csv --sample 100000
    uv run python scripts/embedding_experiment.py --csv data/email_phishing_data.csv  # full dataset
"""

# %%
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split, stratified_sample
from phishing.core.splits import DataSplit
from phishing.experiments.runner import ThresholdConfig, run_experiments
from phishing.features.engineering import FeatureEngineer


# %%
def _engineer_all(split: DataSplit) -> tuple[FeatureEngineer, DataSplit]:
    """Fit FeatureEngineer on train, apply to all splits (leakage-safe)."""
    eng = FeatureEngineer(keep_raw=True).fit(split.X_train)
    return eng, DataSplit(
        X_train=eng.transform(split.X_train),
        y_train=split.y_train,
        X_val=eng.transform(split.X_val),
        y_val=split.y_val,
        X_test=eng.transform(split.X_test),
        y_test=split.y_test,
    )


# %%
def _append_embedding(split: DataSplit, embedder) -> DataSplit:
    """Append the 20 nn_* embedding columns to every split's features."""
    def add(X):
        emb = embedder.transform(X)
        nn = emb[[c for c in emb.columns if c.startswith("nn_")]].reset_index(drop=True)
        return pd.concat([X.reset_index(drop=True), nn], axis=1)

    return DataSplit(
        X_train=add(split.X_train), y_train=split.y_train,
        X_val=add(split.X_val), y_val=split.y_val,
        X_test=add(split.X_test), y_test=split.y_test,
    )


# %%
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-trained NN embedding experiment.")
    p.add_argument("--csv", required=True)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "catboost"])
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--no-embedding", action="store_true",
                   help="Engineered-only baseline (skip the embedding).")
    p.add_argument("--save-embedding", action="store_true",
                   help="Persist the trained embedding model as a version.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true",
                   help="Log each model run to MLflow for later analysis.")
    args = p.parse_args(argv)

    raw = data_mod.load_csv(args.csv)
    if args.sample is not None:
        raw = stratified_sample(raw, args.sample, random_state=args.seed)
    print(f"Rows: {len(raw):,} | phishing rate {raw[data_mod.TARGET].mean():.4%}")

    split = stratified_95_5_split(raw, random_state=args.seed)
    _, split = _engineer_all(split)
    print(f"Engineered features: {split.X_train.shape[1]}")

    if not args.no_embedding:
        from phishing.features.nn_embedding import NNEmbedding

        print(
            f"Training NN embedding once on the train split "
            f"({len(split.y_train):,} rows) with SGD lr={args.learning_rate}, "
            f"momentum schedule 0.5->0.95 ..."
        )
        embedder = NNEmbedding(
            optimizer="sgd",
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=512,
            momentum_schedule=True,
            keep_raw=False,
        ).fit(split.X_train, split.y_train)
        print(f"Embedding training stopped after {embedder.n_epochs_trained_} epochs.")

        if args.save_embedding:
            from phishing.core.wrapper import ModelWrapper

            w = ModelWrapper(embedder, name="nn_embedding", feature_mode="engineered")
            w.feature_names_ = list(split.X_train.columns)
            path = w.save(epochs_trained=embedder.n_epochs_trained_)
            print(f"Saved embedding model -> {path}")

        split = _append_embedding(split, embedder)
        print(f"Features after embedding: {split.X_train.shape[1]} (+20 nn_*)")

    results, ensembles = run_experiments(
        split,
        model_names=args.models,
        feature_mode="raw",  # features already prepared; no per-fold retrain
        threshold_cfg=ThresholdConfig(mode="max_f1"),
        build_blend=True,
        n_splits=args.cv_folds,
        log_mlflow=args.mlflow,
        progress=lambda m: print(m, flush=True),
    )

    rows = []
    for r in results:
        rows.append({
            "model": r.name,
            "cv_pr_auc": round(r.cv_pr_auc, 4),
            "test_pr_auc": round(r.test_metrics["pr_auc"], 4),
            "test_recall": round(r.test_metrics["recall"], 4),
            "test_precision": round(r.test_metrics["precision"], 4),
            "test_f1": round(r.test_metrics["f1"], 4),
            "test_mcc": round(r.test_metrics["mcc"], 4),
        })
    if ensembles and "blend" in ensembles:
        b = ensembles["blend"]
        rows.append({
            "model": "blend", "cv_pr_auc": float("nan"),
            "test_pr_auc": round(b["test_metrics"]["pr_auc"], 4),
            "test_recall": round(b["test_metrics"]["recall"], 4),
            "test_precision": round(b["test_metrics"]["precision"], 4),
            "test_f1": round(b["test_metrics"]["f1"], 4),
            "test_mcc": round(b["test_metrics"]["mcc"], 4),
        })
    table = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    tag = "WITHOUT embedding" if args.no_embedding else "WITH NN embedding"
    print(f"\n=== Comparison ({tag}; sorted by test PR-AUC) ===")
    print(table.to_string(index=False))
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
