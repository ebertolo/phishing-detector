"""One-off experiment: train using only the 3 best (engineered) features.

The 3 highest-signal features identified by mutual-information analysis are:

- ``has_emails``      : 1 if the email embeds any address (MI ~0.035, top overall)
- ``is_short_email`` : 1 if num_words < 50 (MI ~0.023, "phishing = short")
- ``has_urgent``      : 1 if any urgent keyword present (MI ~0.013)

This builds those 3 features from the raw counts, runs the same stratified 95/5
split + GridSearch/CV + calibration + threshold + blend pipeline as the main
suite (feature_mode="raw", since the 3 columns are already the final features),
and prints the imbalance-aware comparison. Run:

    uv run python scripts/experiment_top3.py --csv data/email_phishing_data.csv [--sample N]
"""

# %%
from __future__ import annotations

import argparse

import pandas as pd

from phishing.core import data as data_mod
from phishing.core.dataset import stratified_95_5_split, stratified_sample
from phishing.experiments.runner import ThresholdConfig, run_experiments
from phishing.models import DEFAULT_MODELS

TARGET = data_mod.TARGET


# %%
def build_top3(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with only the 3 best engineered features + target."""
    out = pd.DataFrame(index=df.index)
    out["has_emails"] = (df["num_email_addresses"] > 0).astype(int)
    out["is_short_email"] = (df["num_words"] < 50).astype(int)
    out["has_urgent"] = (df["num_urgent_keywords"] > 0).astype(int)
    if TARGET in df.columns:
        out[TARGET] = df[TARGET].astype(int)
    return out


# %%
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Top-3 engineered features experiment.")
    p.add_argument("--csv", required=True)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--recall-target", type=float, default=0.6)
    p.add_argument("--precision-floor", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    raw = data_mod.load_csv(args.csv)
    if args.sample is not None:
        raw = stratified_sample(raw, args.sample, random_state=args.seed)
    df3 = build_top3(raw)
    print(f"Top-3 features: {[c for c in df3.columns if c != TARGET]}")
    print(f"Rows: {len(df3):,} | phishing rate: {df3[TARGET].mean():.4%}")

    split = stratified_95_5_split(df3, random_state=args.seed)
    print(f"Split positive rates: {split.positive_rates()}")

    results, ensembles = run_experiments(
        split,
        model_names=DEFAULT_MODELS,
        feature_mode="raw",  # the 3 columns ARE the final features
        threshold_cfg=ThresholdConfig(
            mode="recall_target",
            recall_target=args.recall_target,
            precision_floor=args.precision_floor,
        ),
        build_blend=True,
        n_splits=args.cv_folds,
        log_mlflow=False,
        progress=lambda m: print(m, flush=True),
    )
    blend = (ensembles or {}).get("blend")

    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "test_pr_auc": round(r.test_metrics["pr_auc"], 4),
                "test_recall": round(r.test_metrics["recall"], 4),
                "test_precision": round(r.test_metrics["precision"], 4),
                "test_f1": round(r.test_metrics["f1"], 4),
                "test_mcc": round(r.test_metrics["mcc"], 4),
                "threshold": round(r.threshold, 4),
            }
        )
    if blend:
        rows.append(
            {
                "model": "blend",
                "test_pr_auc": round(blend["test_metrics"]["pr_auc"], 4),
                "test_recall": round(blend["test_metrics"]["recall"], 4),
                "test_precision": round(blend["test_metrics"]["precision"], 4),
                "test_f1": round(blend["test_metrics"]["f1"], 4),
                "test_mcc": round(blend["test_metrics"]["mcc"], 4),
                "threshold": round(blend["threshold"], 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    print("\n=== Top-3 features comparison (test; accuracy omitted) ===")
    print(table.to_string(index=False))
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
