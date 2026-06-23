"""Per-model comparison: each booster, with vs without the NN embedding.

Answers two questions on the current (rich) engineered features:
1. Is XGBoost still the best single model?
2. Does the frozen NN embedding still help per model?

Trains each booster individually (RandomizedSearch) on the engineered features,
once **without** the embedding and once **with** the frozen 20-dim embedding
appended, on the 90/5/5 split. Prints test PR-AUC for every cell plus a summary.
Progress is timestamped.

Run: uv run python scripts/per_model_embedding.py --csv data/email_phishing_data.csv
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
from phishing.core.splits import DataSplit
from phishing.experiments.runner import ThresholdConfig, run_model
from phishing.features.engineering import FeatureEngineer
from phishing.features.nn_embedding import NNEmbedding

MODELS = ["lightgbm", "xgboost", "catboost"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true",
                   help="Log each model run to MLflow for later analysis.")
    args = p.parse_args(argv)

    raw = data_mod.load_csv(args.csv)
    if args.sample is not None:
        raw = stratified_sample(raw, args.sample, random_state=args.seed)
    # 90/5/5 split (val 5% calibration, test 5% report) — matches best_model_report.
    split = stratified_95_5_split(
        raw, val_fraction=0.05, test_fraction_within_95=0.0526, random_state=args.seed,
    )
    log(f"Split — train={len(split.y_train):,}, val={len(split.y_val):,}, test={len(split.y_test):,}")

    eng = FeatureEngineer(keep_raw=True).fit(split.X_train)
    Xtr, Xval, Xte = (eng.transform(split.X_train), eng.transform(split.X_val),
                      eng.transform(split.X_test))
    log(f"Engineered feature set: {Xtr.shape[1]} columns.")

    base = DataSplit(Xtr, split.y_train, Xval, split.y_val, Xte, split.y_test)

    log("Training NN embedding once (dim=20) ...")
    embedder = NNEmbedding(optimizer="sgd", learning_rate=0.005, epochs=1000,
                           batch_size=512, momentum_schedule=True, keep_raw=False)
    embedder.fit(Xtr, split.y_train)
    log(f"Embedding stopped after {embedder.n_epochs_trained_} epochs.")

    def add_emb(X):
        e = embedder.transform(X)
        nn = e[[c for c in e.columns if c.startswith("nn_")]].reset_index(drop=True)
        return pd.concat([X.reset_index(drop=True), nn], axis=1)

    aug = DataSplit(add_emb(Xtr), split.y_train, add_emb(Xval), split.y_val,
                    add_emb(Xte), split.y_test)

    thr = ThresholdConfig(mode="max_f1")
    results = {}
    for label, ds in (("without_embedding", base), ("with_embedding", aug)):
        for name in MODELS:
            log(f"[{label}] training {name} (random {args.n_iter}) ...")
            r = run_model(name, ds, feature_mode="raw", threshold_cfg=thr,
                          n_splits=args.cv_folds, log_mlflow=args.mlflow,
                          search_method="random", n_iter=args.n_iter,
                          search_verbose=2)
            results[(label, name)] = r.test_metrics["pr_auc"]
            log(f"[{label}] {name} test PR-AUC = {r.test_metrics['pr_auc']:.4f}")

    print("\n=== Per-model test PR-AUC (rich engineered features, 90/5/5) ===")
    print(f"{'model':<12} {'without_emb':>12} {'with_emb':>12} {'delta':>8}")
    for name in MODELS:
        wo = results[("without_embedding", name)]
        wi = results[("with_embedding", name)]
        print(f"{name:<12} {wo:>12.4f} {wi:>12.4f} {wi - wo:>+8.4f}")

    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\nBest single model/condition: {best[0][1]} ({best[0][0]}) = {best[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
