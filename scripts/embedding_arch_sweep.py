"""Sweep the NN-embedding architecture (dropout x embedding width).

For each dropout (and a configurable embedding width / second-layer width), trains
the embedding **once** on the train split (leakage-safe, frozen), appends it to the
engineered features, and trains a single model (XGBoost by default, for speed) via
RandomizedSearch. Reports, per cell: **test PR-AUC** and the embedding's
**train/val PR-AUC gap** (the overfit signal that motivates higher dropout).

90/5/5 split, timestamped progress. One job at a time.

Examples
--------
    # 100k, embedding_dim=32, four dropouts, xgboost only
    uv run python scripts/embedding_arch_sweep.py --csv data/email_phishing_data.csv \
        --sample 100000 --embedding-dim 32 --hidden2-dim 32 \
        --dropouts 0.5 0.6 0.7 0.75

    # confirm winner on the full dataset
    uv run python scripts/embedding_arch_sweep.py --csv data/email_phishing_data.csv \
        --embedding-dim 32 --hidden2-dim 32 --dropouts 0.6
"""

# %%
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split, stratified_sample
from phishing.core.metrics import compute_metrics
from phishing.core.splits import DataSplit
from phishing.experiments.runner import ThresholdConfig, run_model
from phishing.features.engineering import FeatureEngineer
from phishing.features.nn_embedding import NNEmbedding


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.75])
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--hidden2-dim", type=int, default=32)
    p.add_argument("--model", default="xgboost")
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--embedding-epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args(argv)

    raw = data_mod.load_csv(args.csv)
    if args.sample is not None:
        raw = stratified_sample(raw, args.sample, random_state=args.seed)
    split = stratified_95_5_split(
        raw, val_fraction=0.05, test_fraction_within_95=0.0526, random_state=args.seed,
    )
    log(f"Split — train={len(split.y_train):,}, val={len(split.y_val):,}, test={len(split.y_test):,}")

    eng = FeatureEngineer(keep_raw=True).fit(split.X_train)
    Xtr, Xval, Xte = (eng.transform(split.X_train), eng.transform(split.X_val),
                      eng.transform(split.X_test))
    log(f"Engineered features: {Xtr.shape[1]} columns "
        f"(embedding_dim={args.embedding_dim}, hidden2_dim={args.hidden2_dim}).")

    thr = ThresholdConfig(mode="max_f1")
    rows = []
    for dr in args.dropouts:
        log(f"--- dropout={dr}: training embedding once ---")
        emb = NNEmbedding(
            optimizer="sgd", learning_rate=0.005, epochs=args.embedding_epochs,
            batch_size=512, momentum_schedule=True, keep_raw=False,
            embedding_dim=args.embedding_dim, hidden2_dim=args.hidden2_dim, dropout=dr,
        ).fit(Xtr, split.y_train)
        log(f"dropout={dr}: embedding {emb.n_epochs_trained_} epochs, "
            f"train_pr_auc={emb.train_pr_auc_:.4f}, val_pr_auc={emb.val_pr_auc_:.4f}, "
            f"gap={emb.overfit_gap_:+.4f}")

        def add_emb(X):
            e = emb.transform(X)
            nn = e[[c for c in e.columns if c.startswith("nn_")]].reset_index(drop=True)
            return pd.concat([X.reset_index(drop=True), nn], axis=1)

        aug = DataSplit(add_emb(Xtr), split.y_train, add_emb(Xval), split.y_val,
                        add_emb(Xte), split.y_test)
        log(f"dropout={dr}: training {args.model} (random {args.n_iter}) ...")
        r = run_model(args.model, aug, feature_mode="raw", threshold_cfg=thr,
                      n_splits=args.cv_folds, log_mlflow=args.mlflow,
                      search_method="random", n_iter=args.n_iter, search_verbose=2)
        m = compute_metrics(np.asarray(aug.y_test),
                            r.wrapper.predict_proba(aug.X_test), r.threshold)
        log(f"dropout={dr}: {args.model} test PR-AUC = {m.pr_auc:.4f}")
        rows.append({
            "dropout": dr, "emb_dim": args.embedding_dim,
            "test_pr_auc": round(m.pr_auc, 4),
            "emb_train_pr_auc": round(emb.train_pr_auc_, 4),
            "emb_val_pr_auc": round(emb.val_pr_auc_, 4),
            "emb_gap": round(emb.overfit_gap_, 4),
            "recall": round(m.recall, 4), "precision": round(m.precision, 4),
            "f1": round(m.f1, 4), "mcc": round(m.mcc, 4),
            "confusion": f"[[{m.tn} {m.fp}][{m.fn} {m.tp}]]",
        })

    table = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    print(f"\n=== Embedding arch sweep ({args.model}, dim={args.embedding_dim}, "
          f"hidden2={args.hidden2_dim}, 90/5/5) ===")
    print(table.to_string(index=False))
    best = table.iloc[0]
    print(f"\nBest: dropout={best['dropout']} -> test PR-AUC {best['test_pr_auc']} "
          f"(embedding gap {best['emb_gap']:+})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
