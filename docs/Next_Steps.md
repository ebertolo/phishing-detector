# Next Steps — Prioritised Improvements

Strategic assessment of what could move the **best model** (blend of tuned
LightGBM + XGBoost + CatBoost on `engineered` features + frozen NN embedding,
90/5/5 split, **test PR-AUC ≈ 0.43**). Each lever is rated by **impact × effort ×
risk**. Items marked ✅ **DONE** have been tested; the rest are open.

Honest framing: phishing here is genuinely hard (8 weak integer count features,
~1.3% positives), so realistic gains are **increments of ~0.01–0.05 PR-AUC**, not
jumps. The single biggest jump available — richer raw data (URLs/headers/text) —
is a data problem, not a modelling one.

---

## 1. Hyperparameter tuning of the boosters — ✅ DONE (the biggest lever)

**Impact: HIGH · Effort: LOW · Risk: LOW.** The original grids were tiny (16/16/8
combos). Wider grids + `RandomizedSearchCV` (40 samples over `n_estimators`,
`learning_rate`, depth/`num_leaves`, `subsample`, `colsample_bytree`,
regularisation) lifted the blend from **0.367 → 0.432 (+0.065)** — the largest
gain since feature engineering. Implemented via `param_distributions()` on each
booster and `--search random --n-iter N` in `run_experiments.py` /
`best_model_report.py`. **Conclusion: the boosters were undertuned; this was the
top cheap win, as predicted.**

*Open follow-up:* try Optuna (TPE) instead of random sampling, and raise `n_iter`
to 80–100 — likely a small additional gain.

## 2. Embedding width / architecture — ✅ DONE (no improvement)

**Impact: turned out LOW · Effort: MEDIUM.** `EMBEDDING_DIM` was parameterised
(20/32/64), a linear+ReLU head added on top (Yandex), and **periodic (Fourier)
numerical embeddings** + a cosine LR schedule implemented in `_tf_net.py` /
`nn_embedding.py`. Tested dim 32 + periodic + cosine on the full dataset: it made
the blend **slightly worse (0.432 → 0.413)**. Consistent with the fraud
literature ([Booking.com, arXiv 2405.13692](https://arxiv.org/pdf/2405.13692)):
sophisticated deep components rarely beat tuned GBDTs under extreme imbalance.
**Conclusion: the simple 20-dim embedding is enough; keep it.** The periodic /
wider options remain available (`--embedding-dim`, `--periodic`,
`--cosine-schedule`) for future data.

A second round also swept the **second intermediate layer (20→32)**,
**embedding_dim ∈ {16, 32}** and **dropout ∈ {0.5, 0.6, 0.7, 0.75}**
(`embedding_arch_sweep.py`). On 100k all dropouts tied (~0.22); on the full
dataset dropout 0.5 / dim 32 gave 0.4381 vs the **0.4385 baseline** (dim 20,
dropout 0.4) — i.e. **no improvement**, and the embedding train/val gaps were tiny
& negative (no overfit to fix). **Conclusion: the dim-20 / dropout-0.4 embedding
is already at its sweet spot.** This closes the embedding-architecture line.

## 3. Ensemble strategy: blend vs stacking — ✅ DONE (blend wins, narrowly)

**Impact: LOW.** With tuned boosters all three strategies land within ~0.003
PR-AUC: **blend 0.432** (best PR-AUC, fewest false positives), **stacking 0.429**
(best MCC), single XGBoost 0.432 (best recall). The blend is preferred for
simplicity and serialisability (`BlendModel`). Enable stacking with `--stacking`.

## 4. K-fold size & repetition — OPEN (robustness, not PR-AUC)

**Impact: MEDIUM (reliability) · Effort: LOW.** The best-model runs use CV=3; the
default is 5. Moving to **CV=5** and/or `RepeatedStratifiedKFold` reduces the
variance of the hyperparameter selection and the PR-AUC estimate. It mainly buys
**confidence**, not a higher headline number. Low risk; cost is linear in folds.

## 5. Calibration: isotonic vs sigmoid on the full dataset — OPEN

**Impact: LOW–MEDIUM · Effort: LOW.** Already selectable (`calibration="isotonic"`)
but only compared on a sample. The 5% validation set (~26k rows) has enough data
for isotonic, which may sharpen probabilities and improve the F1/MCC at the chosen
threshold (it does not change PR-AUC). Worth a quick A/B on the full data.

## 6. Operating point — ✅ DONE (deployment knob, not PR-AUC)

**Impact: deployment-critical · Effort: trivial.** Tested `--threshold-mode cost`
with **FN = 20×** (vs the 10× default) on the tuned XGBoost (full dataset): recall
rose to **0.527** (from ~0.36 at max-F1) at precision ~0.18. PR-AUC is unchanged —
the threshold only moves along the curve. **Conclusion: use `--fn-cost` to set the
production operating point by business cost** (higher FN weight → more phishing
caught, more false alarms). Still open: a full recall-target sweep
(0.6/0.7/0.8) + isotonic vs sigmoid (§5) to finalise the deployment point.

## 7. Bigger bets (higher effort / uncertain)

- **Native focal loss (`lgb.train`/`xgb.train`)** — ✅ TRIED, did not work.
  Reimplemented as `lightgbm_focal_native` / `xgboost_focal_native` using the
  native APIs (not the sklearn wrapper). They still **collapse** on the full
  imbalanced dataset (PR-AUC ~0.01, NaN CV). Confirms the instability is the ~1.3%
  imbalance, not the wrapper. Kept available; CatBoost's native focal is the only
  working focal variant.
- **FT-Transformer / self-supervised (SCARF, VIME)** — state-of-the-art tabular DL
  but needs PyTorch (new heavy dep) and, per the fraud literature, an uncertain
  win over tuned GBDTs here. **Not prioritised** given §2's negative result.
- **Richer raw features** (actual URLs, headers, body text) — the literature's
  strongest phishing signal, and the single biggest *potential* jump. Blocked on
  data we don't have. If it arrives, this is the priority.
- **Investigate the ~225 false negatives** of the best blend — cluster them to see
  whether a phishing sub-type is systematically missed, motivating a targeted
  feature.
- **Serve the saved model via FastAPI — ✅ DONE.** `fastapi_app/` loads the most
  recently saved version (or a pinned one via `MODEL_VERSION_DIR`) and exposes
  `POST /predict`, returning the phishing likelihood as the primary output. See
  [`fastapi_app/README.md`](../fastapi_app/README.md) and
  [DESIGN.md](DESIGN.md#rest-api-fastapi_app). Still open: **drift monitoring**
  on the MLflow inference summaries (no alerting/dashboard built yet, only the
  logged summaries themselves).

---

## Recommended order of attack (updated)

| # | lever | status | impact | result / next |
|---|---|---|---|---|
| 1 | Tune boosters (RandomizedSearch) | ✅ done | HIGH | **0.367 → 0.432**; try Optuda/n_iter↑ |
| 2 | Embedding width / periodic | ✅ done | LOW | 0.413 (worse) — keep dim 20 |
| 3 | Blend vs stacking | ✅ done | LOW | blend wins narrowly (0.432) |
| 4 | CV=5 / repeated | open | MED (robustness) | tighten variance |
| 5 | Isotonic calibration (full) | open | LOW–MED | sharpen F1/MCC at threshold |
| 6 | Threshold sweep (recall/cost) | open | deploy | pick production point |
| 7 | Richer raw data / FT-Transformer | open | HIGH/uncertain | blocked on data / heavy dep |

**Bottom line:** the cheap, high-impact win (booster tuning) is captured. The
remaining open items are about **robustness, calibration and deployment**, not
big PR-AUC jumps — those require richer input data, not more modelling.
