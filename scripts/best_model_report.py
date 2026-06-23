"""Report the confusion matrix and metrics of the best model.

Best configuration: XGBoost on `engineered` features augmented with the frozen
NN embedding (pre-trained once on the train split), evaluated on the held-out
test set of the full dataset. Reuses the same pipeline as the experiment runner
so the numbers match the journey/results docs.

Run: uv run python scripts/best_model_report.py --csv data/email_phishing_data.csv
"""

# %%
from __future__ import annotations

import argparse
import os

import time
from datetime import datetime

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line so long runs show where they are."""
    elapsed = time.time() - _T0
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} +{elapsed:6.0f}s] {msg}", flush=True)


from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split, stratified_sample
from phishing.core.embedding_cache import (
    load_cached_embedding,
    make_embedding_cache_key,
    save_cached_embedding,
)
from phishing.core.metrics import compute_metrics
from phishing.core.splits import DataSplit
from phishing.experiments.runner import ThresholdConfig, run_experiments
from phishing.features.engineering import FeatureEngineer
from phishing.features.nn_embedding import NNEmbedding

_EMB_OPTIMIZER = "sgd"
_EMB_LEARNING_RATE = 0.005


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--sample", type=int, default=None)
    # Default = 90/5/5: validation 5% (calibration + threshold + blend weights),
    # test 5% (clean holdout where the confusion matrix is reported), train ~90%.
    # test_fraction is taken *within the remaining 95%*, so 0.0526 ≈ 5% of total.
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.0526)
    p.add_argument("--seed", type=int, default=42)
    # Embedding architecture knobs.
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--embedding-hidden1-dim", type=int, default=40,
                    help="Width of the embedding net's 1st intermediate layer.")
    p.add_argument("--embedding-hidden2-dim", type=int, default=20,
                    help="Width of the embedding net's 2nd intermediate layer.")
    p.add_argument("--embedding-dropout1", type=float, default=0.4,
                    help="Dropout after the 1st intermediate layer.")
    p.add_argument("--embedding-dropout2", type=float, default=0.4,
                    help="Dropout after the 2nd intermediate layer.")
    p.add_argument("--periodic", action="store_true",
                   help="Use periodic (Fourier) numerical embeddings in the net.")
    p.add_argument("--cosine-schedule", action="store_true",
                   help="Cosine LR decay over the embedding training.")
    p.add_argument("--embedding-epochs", type=int, default=1000)
    p.add_argument(
        "--embedding-patience", type=int, default=50,
        help="Early-stopping patience for the NN embedding (epochs without a "
        "val_pr_auc improvement before stopping; default 50).",
    )
    p.add_argument(
        "--force-embedding-search",
        action="store_true",
        help="Ignore any cached trained embedding (embeddings/*) and retrain it "
        "even if a matching cached one exists.",
    )
    # Booster hyperparameter search + ensembling.
    # Default 'random': RandomizedSearchCV over wider param distributions is what
    # produced the headline result (~0.44 blend); plain grid search undertunes
    # the boosters (~0.37). See docs/RESULTS.md.
    p.add_argument("--search", default="random", choices=["grid", "random"])
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--stacking", action="store_true",
                   help="Also build a logistic stacking meta-model.")
    p.add_argument("--mlflow", action="store_true",
                   help="Log every model run to MLflow (params, metrics, curves) "
                        "for later analysis in the MLflow UI.")
    p.add_argument(
        "--threshold-mode",
        default="max_f1",
        choices=["recall_target", "max_f1", "manual", "cost"],
        help="Decision-threshold strategy on validation (default max_f1).",
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
        "--force-search",
        action="store_true",
        help="Ignore any cached best-params (best_params/*.json) and re-run the "
        "hyperparameter search even if a matching cached winner exists.",
    )
    args = p.parse_args(argv)

    raw = data_mod.load_csv(args.csv)
    if args.sample is not None:
        raw = stratified_sample(raw, args.sample, random_state=args.seed)
    split = stratified_95_5_split(
        raw, val_fraction=args.val_fraction,
        test_fraction_within_95=args.test_fraction, random_state=args.seed,
    )
    log(
        "Split (stratified) — "
        f"train={len(split.y_train):,} ({split.y_train.mean():.4%}), "
        f"val={len(split.y_val):,} ({split.y_val.mean():.4%}, calibration/threshold), "
        f"test={len(split.y_test):,} ({split.y_test.mean():.4%}, matrix report)"
    )

    # Engineer features (fit on train, apply to all splits).
    log("Engineering features (fit on train) ...")
    eng = FeatureEngineer(keep_raw=True).fit(split.X_train)
    Xtr, Xval, Xte = (eng.transform(split.X_train), eng.transform(split.X_val),
                      eng.transform(split.X_test))
    log(f"Feature set: {Xtr.shape[1]} columns (raw + engineered) feed the embedding.")

    # Train the NN embedding once on the train split (leakage-safe) and freeze it.
    # The embedding input width auto-adjusts to the engineered feature count;
    # its output stays at --embedding-dim (default 16). A cached embedding is
    # reused when every hyperparameter and the input columns/row count match a
    # previous run (see core.embedding_cache); --force-embedding-search ignores it.
    feature_columns = list(Xtr.columns)
    emb_cache_key = make_embedding_cache_key(
        args.embedding_dim, args.embedding_hidden1_dim, args.embedding_hidden2_dim,
        args.embedding_dropout1, args.embedding_dropout2, args.embedding_patience,
        _EMB_OPTIMIZER, _EMB_LEARNING_RATE, args.periodic, args.cosine_schedule,
        feature_columns, len(split.y_train),
    )
    cached = None if args.force_embedding_search else load_cached_embedding(emb_cache_key)
    if cached is not None:
        embedder, emb_meta = cached
        log(
            f"Using cached NN embedding (cache_key={emb_cache_key}, "
            f"trained {emb_meta.n_epochs_trained} epochs, "
            f"val_pr_auc={emb_meta.val_pr_auc:.4f}) — skipping embedding training."
        )
    else:
        log(
            f"Training NN embedding once on {len(split.y_train):,} train rows "
            f"(input={Xtr.shape[1]} feats, output dim={args.embedding_dim}, "
            f"hidden1={args.embedding_hidden1_dim}, hidden2={args.embedding_hidden2_dim}, "
            f"dropout1={args.embedding_dropout1}, dropout2={args.embedding_dropout2}, "
            f"periodic={args.periodic}, cosine={args.cosine_schedule}, "
            f"patience={args.embedding_patience}, max_epochs={args.embedding_epochs}) ..."
        )
        embedder = NNEmbedding(
            optimizer=_EMB_OPTIMIZER, learning_rate=_EMB_LEARNING_RATE,
            epochs=args.embedding_epochs, batch_size=512, momentum_schedule=True,
            keep_raw=False, embedding_dim=args.embedding_dim,
            hidden1_dim=args.embedding_hidden1_dim, hidden2_dim=args.embedding_hidden2_dim,
            dropout1=args.embedding_dropout1, dropout2=args.embedding_dropout2,
            periodic=args.periodic, cosine_schedule=args.cosine_schedule,
            patience=args.embedding_patience,
        )
        embedder.fit(Xtr, split.y_train)
        log(f"Embedding training stopped after {embedder.n_epochs_trained_} epochs.")
        save_cached_embedding(
            embedder, args.embedding_dim, args.embedding_hidden1_dim,
            args.embedding_hidden2_dim, args.embedding_dropout1, args.embedding_dropout2,
            args.embedding_patience, _EMB_OPTIMIZER, _EMB_LEARNING_RATE,
            args.periodic, args.cosine_schedule, feature_columns, len(split.y_train),
            embedder.train_pr_auc_, embedder.val_pr_auc_, embedder.n_epochs_trained_,
        )
        log(f"Cached embedding for reuse (cache_key={emb_cache_key}).")

    def add_emb(X):
        e = embedder.transform(X)
        nn = e[[c for c in e.columns if c.startswith("nn_")]].reset_index(drop=True)
        import pandas as pd
        return pd.concat([X.reset_index(drop=True), nn], axis=1)

    aug = DataSplit(
        X_train=add_emb(Xtr), y_train=split.y_train,
        X_val=add_emb(Xval), y_val=split.y_val,
        X_test=add_emb(Xte), y_test=split.y_test,
    )

    # Train the full booster set + blend (+ stacking) on the augmented features
    # (feature_mode="raw": features already prepared, no per-fold retransform).
    log(f"Training 3 boosters ({args.search} search) on engineered + frozen embedding ...")
    results, ensembles = run_experiments(
        aug,
        model_names=["lightgbm", "xgboost", "catboost"],
        feature_mode="raw",
        threshold_cfg=ThresholdConfig(
            mode=args.threshold_mode,
            recall_target=args.recall_target,
            precision_floor=args.precision_floor,
            manual_value=args.manual_threshold,
            fn_cost=args.fn_cost,
            fp_cost=args.fp_cost,
        ),
        build_blend=True,
        build_stacking=args.stacking,
        n_splits=args.cv_folds,
        search_method=args.search,
        n_iter=args.n_iter,
        search_verbose=2,  # sklearn prints one line per CV fit (40 cand x folds)
        progress=log,
        log_mlflow=args.mlflow,
        force_search=args.force_search,
    )
    if args.mlflow:
        log("Runs logged to MLflow (experiment 'phishing-fit'). View: uv run mlflow ui")
    log("Boosters + ensembles done; computing held-out test metrics ...")

    def report(name, m):
        print(f"\n=== {name} | held-out test ===")
        print(f"threshold ({args.threshold_mode}): {m.threshold:.4f}")
        print(f"PR-AUC   : {m.pr_auc:.4f}")
        print(f"ROC-AUC  : {m.roc_auc:.4f}")
        print(f"precision: {m.precision:.4f}")
        print(f"recall   : {m.recall:.4f}")
        print(f"F1       : {m.f1:.4f}")
        print(f"MCC      : {m.mcc:.4f}")
        print(f"confusion [[tn fp][fn tp]] = [[{m.tn} {m.fp}][{m.fn} {m.tp}]]")
        print(f"n_samples={m.n_samples:,}  n_positives={m.n_positives:,}")

    xgb = next(r for r in results if r.name == "xgboost")
    report("XGBoost + frozen NN embedding",
           compute_metrics(np.asarray(aug.y_test),
                           xgb.wrapper.predict_proba(aug.X_test), xgb.threshold))

    if ensembles and "blend" in ensembles:
        bw = ensembles["blend"]["wrapper"]
        report("Blend (LGBM+XGB+CatBoost) + frozen NN embedding",
               compute_metrics(np.asarray(aug.y_test),
                               bw.predict_proba(aug.X_test), bw.threshold))

    if ensembles and "stacking" in ensembles:
        # Stacking's test_metrics are already computed on this test set by the runner.
        tm = ensembles["stacking"]["test_metrics"]
        print("\n=== Stacking (logistic meta) + frozen NN embedding | held-out test ===")
        print(f"threshold ({args.threshold_mode}): {tm['threshold']:.4f}")
        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "mcc"):
            print(f"{k:9s}: {tm[k]:.4f}")
        print(f"confusion [[tn fp][fn tp]] = [[{tm['tn']} {tm['fp']}][{tm['fn']} {tm['tp']}]]")
        print(f"meta-coefficients: {ensembles['stacking']['meta_coefficients']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
