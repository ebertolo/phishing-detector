# Experiment Results — Encoding & Model Comparison

Full suite run on the real dataset (`data/email_phishing_data.csv`, 524 846 rows,
~1.32% phishing) using a **stratified 100 000-row sample** (rate preserved at
1.324%) for fast iteration. Split: 76 000 train / 19 000 test / 5 000 validation,
stratified. CV folds = 3, threshold mode = max-F1.

Reproduce any row with:

```bash
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
    --sample 100000 --feature-mode <MODE> --cv-folds 3
```

Primary metric is **test PR-AUC** (average precision). Accuracy is intentionally
omitted — at ~1.3% positives it is uninformative. For the *narrative* of how
these results drove the design — what was kept and discarded — see
[EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md).

## Headline — best result so far

**Blend (LightGBM + XGBoost + CatBoost) on the full `engineered` catalogue (52
features) + frozen NN embedding, 90/5/5 split, with hyperparameter-tuned boosters
→ test PR-AUC ≈ 0.44.** The progression that got there:

| stage | test PR-AUC |
|---|---|
| raw counts, blend (100k) | ~0.21 |
| engineered features, blend (100k) | ~0.25 |
| engineered, blend, full dataset | 0.31 |
| + frozen NN embedding (full) | 0.33 |
| + 90/5/5 split (more training data) | 0.37 |
| + tuned boosters (RandomizedSearch, wider grids) | 0.43 |
| + expanded feature catalogue (52 features) | **0.44** |

Modest in absolute terms (phishing is genuinely hard with 8 weak count features),
but each step is a real, leakage-safe gain. **Hyperparameter tuning of the
boosters gave the biggest single jump (+0.065)**; the expanded feature catalogue
added a final +0.01. Reproduce the best with
`uv run python scripts/best_model_report.py --csv data/email_phishing_data.csv \
    --search random --n-iter 40 --stacking`.

## Hyperparameter and embedding caching

Both the hyperparameter search and the NN embedding training are the
expensive parts of `best_model_report.py` on the full dataset — a cold run
(no cache) takes roughly **55–60 minutes** end to end. Two on-disk caches,
keyed by a deterministic hash of everything that affects the result, remove
that cost on a repeated run with the same configuration:

- **`best_params/`** (`src/phishing/core/param_cache.py`) — caches each
  booster's winning hyperparameters, keyed by model name, feature mode,
  search method/budget, CV folds, and the exact training columns. A cache hit
  skips `RandomizedSearchCV`/`GridSearchCV` entirely and fits once with the
  stored parameters.
- **`embeddings/`** (`src/phishing/core/embedding_cache.py`) — caches the
  fitted NN embedding itself (the Keras model, via joblib), keyed by every
  architecture hyperparameter (layer widths, dropout, patience, optimiser,
  learning rate) plus the input feature columns and training-row count.

Both invalidate automatically when the configuration changes (e.g. a
different `--embedding-dim` or feature set produces a different key) — there
is no manual cache-clearing step in normal use. Changing only the **decision
threshold** (`--threshold-mode`, `--fn-cost`/`--fp-cost`) does not affect
either key, since thresholding happens after training; a re-run that only
changes the threshold hits both caches.

Measured on the full dataset (`engineered` features + frozen 16-dim NN
embedding, `lightgbm`/`xgboost`/`catboost`, `--search random --n-iter 40`):

| run | embedding | hyperparameter search | wall time |
|---|---|---|---|
| cold (first run, this configuration) | trained (101 epochs) | full search, all 3 models | ~56 min |
| repeat, same config, different `--fn-cost` | cache hit | cache hit (all 3) | ~96 s |
| repeat, same config, different `--fn-cost` again | cache hit | cache hit (all 3) | ~5 min* |

*the middle row reused an embedding cached by an earlier, separate run; the
last row retrained the embedding once (cache miss on a new embedding-only
parameter) but still hit the hyperparameter cache for all three boosters.
Final metrics at this configuration: XGBoost test PR-AUC 0.434, blend 0.437,
stacking 0.428 — within normal run-to-run variation of the headline 0.44
result above, not a different model.

## New algorithms: NN, logistic regression, clustering, AdaBoost

Four additional independent algorithms were added, all reusing the same
`engineered` features (100k sample, except the NN on 40k for speed):

| model | test PR-AUC | note |
|---|---|---|
| lightgbm (reference) | 0.247 | gradient booster |
| logreg | 0.069 | standalone logistic regression |
| adaboost | 0.021 | AdaBoost over depth-2/3 trees |
| cluster | 0.013 | unsupervised KMeans/GMM, 2 groups |
| tensorflow_dnn | 0.007 (CV) / ~0.08 isolated | dense net, CPU |

**Verdict (honest):** all four under-perform the boosters by a wide margin — the
expected outcome for this problem, where signal is weak and the engineered
features already give tree ensembles the edge.

- **Clustering** sits at the base rate (0.013) — unsupervised splits do not align
  with the label because the two classes overlap heavily in feature space.
- **AdaBoost** is fragile here: depth-1 stumps become "worse than random" under the
  imbalance and abort, so the grid uses depth ≥ 2; CV scores still go NaN in some
  folds.
- **TensorFlow DNN** trains and serializes (round-trips through joblib via a
  sidecar `.keras`), and the optimizer grid (SGD / RMSprop / Adam / Nadam) runs —
  but PR-AUC is weak (~0.08 trained in isolation with enough epochs; RMSprop ≳
  Adam ≳ SGD). It is excluded from `DEFAULT_MODELS` (slow on CPU, low ceiling).

### NN embedding as features — a leakage cautionary tale

The 20-dim embedding layer was first tested via the `engineered_nnembed` feature
mode, where the network **retrains inside every CV fold**. That showed an apparent
+50% lift (lightgbm 0.150 → 0.226 on a 40k sample) — but the gain was an artifact
of **leakage**: training the net inside each fold lets the embedding see that
fold's validation rows (and their labels through the supervised loss).

Re-evaluated correctly — the embedding trained **once on the training split only**
(`scripts/embedding_experiment.py`, SGD lr=0.005 with a 0.5→0.95 momentum
schedule), then frozen and reused as fixed features — the result depends on how
much training data the network gets:

**100k sample** (embedding train ≈ 76k rows, ~1000 positives, early-stopped 151
epochs): the gain disappears — the booster is slightly *worse*. Too little data
for the net to learn a useful representation, so it adds noise.

| model | no embedding | + frozen embedding |
|---|---|---|
| blend | **0.2555** | 0.2376 |
| lightgbm | 0.2470 | 0.2300 |

**Full dataset** (embedding train ≈ 400k rows, ~5300 positives, early-stopped 280
epochs): the gain is **real and leakage-safe** — the network now has enough data
to learn non-linear representations that complement the engineered features.

| model | no embedding | + frozen embedding | Δ |
|---|---|---|---|
| **blend** | 0.3112 | **0.3286** | **+0.017** |
| **xgboost** | 0.3068 | **0.3273** | **+0.021** |
| lightgbm | 0.2829 | 0.2936 | +0.011 |
| catboost | 0.2416 | 0.2348 | −0.007 |

**Verdict:** the NN embedding is a small but genuine improvement **on the full
dataset** (blend 0.311 → 0.329, XGBoost 0.307 → 0.327), but only once the network
has enough positives to train on — it hurts on the 100k sample. Two lessons: (1)
never retrain a supervised feature generator per CV fold (it leaked a fake +50%);
(2) the embedding's value scales with data, so it must be judged on the full set,
not a sample. Pre-train it once with `scripts/embedding_experiment.py`
(`--save-embedding`) and reuse the frozen features.

### Split ratio: 76/19/5 vs 90/5/5 (more training data)

The two best embedding configurations re-run with a **90/5/5** split (train 90%,
val 5% calibration, test 5% report) instead of the original **76/19/5**. Same
convention (val tunes the threshold, test reports), more training data. Test
PR-AUC on the held-out test set:

| option | 76/19/5 | **90/5/5** | Δ |
|---|---|---|---|
| Blend + embedding | 0.336 | **0.367** | +0.031 |
| XGBoost + embedding | 0.321 | **0.356** | +0.035 |

**More training data helps:** +0.03 PR-AUC for both, confirming the model was
training-data-limited rather than at its ceiling. The 5% test holdout still has
~347 phishing — enough for a stable confusion matrix (see
`EXPERIMENT_JOURNEY.md` for the side-by-side matrices). Splits of 98/2 and 99/1
were considered but discarded: a 1–2% holdout has only ~69–138 phishing, too few
for reliable metrics. Reproduce with `uv run python scripts/best_model_report.py
--csv data/email_phishing_data.csv` (90/5/5 is the default).

### Hyperparameter tuning & embedding architecture (full dataset, 90/5/5)

Two experiments from `Next_Steps.md`, both with the frozen NN embedding and the
three boosters, evaluated on the 5% test holdout (26 227 rows, 347 phishing):

| experiment | blend PR-AUC | XGBoost | stacking |
|---|---|---|---|
| baseline (grid search, embedding dim 20) | 0.367 | 0.356 | – |
| tuned boosters (RandomizedSearch n_iter=40, dim 20) | 0.432 | 0.432 | 0.429 |
| tuned boosters + periodic embedding (dim 32, cosine LR) | 0.413 | 0.413 | 0.401 |
| **+ expanded feature catalogue (52 features, dim 20)** | **0.442** | 0.438 | 0.430 |

**Two clear findings:**

1. **Hyperparameter tuning is the biggest lever after feature engineering.**
   Widening the booster grids and using `RandomizedSearchCV` (40 samples over
   `n_estimators`/`learning_rate`/depth/`subsample`/`colsample`/regularisation)
   lifted the blend from **0.367 → 0.432 (+0.065)** — the largest single jump
   since feature engineering. The boosters were genuinely undertuned.
2. **A fancier embedding did not help.** Widening the embedding to 32 dims and
   adding periodic (Fourier) numerical embeddings + a cosine LR schedule made it
   slightly *worse* (0.432 → 0.413). Consistent with the fraud-detection
   literature ([Booking.com, arXiv 2405.13692](https://arxiv.org/pdf/2405.13692)):
   sophisticated deep components rarely beat tuned GBDTs under extreme imbalance.
   The simple 20-dim embedding is enough; the value was in the boosters.

3. **The expanded feature catalogue adds a small further gain.** Growing
   `FeatureEngineer` from the original set to the full **52-feature catalogue**
   (extra densities, structural ratios, content-word intensities, graded flags,
   pairwise interactions; see EXPERIMENT_JOURNEY.md) lifted the blend **0.432 →
   0.442** on the full dataset with tuned boosters. On the 100k sample with grid
   search it was flat (0.252) — the richer features pay off only with enough data
   and a wide search, the same data-hungry pattern as the embedding.

**Blend vs stacking vs single model** (tuned, 52 features, dim 20): all land
within ~0.01 PR-AUC. The **blend wins** (0.442); stacking has slightly higher
recall. The blend is preferred for simplicity and serialisability. Reproduce:

```bash
uv run python scripts/best_model_report.py --csv data/email_phishing_data.csv \
    --search random --n-iter 40 --stacking            # best (0.442)
```

### Per-model: with vs without the NN embedding (52 features, full, 90/5/5)

Each booster tuned individually (RandomizedSearch n_iter=40), with and without the
frozen 20-dim embedding appended:

| model | without embedding | with embedding | Δ |
|---|---|---|---|
| **xgboost** | 0.4235 | **0.4385** | +0.0150 |
| lightgbm | 0.4249 | 0.4308 | +0.0059 |
| catboost | 0.3481 | 0.3648 | +0.0167 |

**Findings:** (1) **XGBoost is still the best single model** (0.4385 with
embedding); (2) **the embedding helps every model** — a consistent, leakage-safe
per-model gain, strongest for XGBoost (+0.015) and CatBoost (+0.017). Without the
embedding LightGBM edges XGBoost narrowly; with it, XGBoost leads. Reproduce with
`uv run python scripts/per_model_embedding.py --csv data/email_phishing_data.csv`.

### Final optimization round — all three levers, all DISCARDED/neutral

A last round tested three open levers from `Next_Steps.md`. **None beat the
current best**; all are documented as closed investigations.

**(a) Embedding dropout / width / 2nd-layer (`embedding_arch_sweep.py`).** Second
intermediate layer 20→32, embedding_dim ∈ {16, 32}, dropout ∈ {0.5, 0.6, 0.7,
0.75}. On 100k all dropouts tied (~0.21–0.23). On the full dataset (dim 32,
hidden2 32, xgboost):

| config | xgboost test PR-AUC | embedding train/val gap |
|---|---|---|
| **baseline (dim 20, hidden2 20, dropout 0.4)** | **0.4385** | — |
| dropout 0.5, dim 32, hidden2 32 | 0.4381 | −0.012 |
| dropout 0.7, dim 32, hidden2 32 | 0.4325 | −0.003 |

No improvement — the variants match or trail the baseline. The train/val gaps are
tiny and *negative*, i.e. **there was no overfit to fix**; the existing dim-20 /
dropout-0.4 embedding was already well-regularised. **Verdict: keep dim 20.**

**(b) Native focal loss (`lgb.train` / `xgb.train`).** New `lightgbm_focal_native`
and `xgboost_focal_native` models were built with the native APIs (the earlier
focal used the sklearn wrapper). They train fine on clean/balanced data, but on
the full imbalanced dataset they **still collapse** (PR-AUC ~0.01, NaN CV folds —
predict all-one-class). **Verdict: the problem is the ~1.3% imbalance, not the
wrapper; native focal does not fix it.** CatBoost's native focal (0.233) works but
trails the tuned XGBoost (0.375). Kept available, excluded from the default roster.

**(c) Cost-sensitive threshold FN = 20× (vs 10×).** Raising `fn_cost` to 20 on the
tuned XGBoost (full dataset) moves the operating point as intended:

| threshold mode | recall | precision | (PR-AUC unchanged) |
|---|---|---|---|
| max-F1 | ~0.36 | ~0.63 | 0.44 |
| cost FN=20× | **0.527** | 0.178 | 0.44 |

FN=20× catches ~53% of phishing (up from ~36%) at much lower precision — the right
trade when a missed phishing email is far costlier than a false alarm. PR-AUC is
unchanged (the threshold only moves along the curve). **Verdict: a useful
deployment knob, not a model gain.**

### Feature smoothing (winsor+quantile, denoising AE)

| feature mode (lightgbm, 100k) | test PR-AUC |
|---|---|
| engineered | 0.247 |
| engineered_smooth (winsor+quantile) | 0.236 |
| engineered_denoise (denoising AE recon) | 0.145 |

Smoothing **does not help** here: boosters are already robust to outliers, and the
denoising-autoencoder reconstruction discards the fine resolution the trees use.
Both modes stay available for validation but are not recommended defaults —
consistent with the broader finding that the plain `engineered` features are hard
to beat.

## Full dataset — all models, with vs without `logz_num_email_addresses`

Run on the **entire 524 846-row dataset** (5 models + blend, CV=3, max-F1
threshold), comparing the engineered set **with** the `logz_num_email_addresses`
feature (`engineered`) against **without** it (`engineered_noemail`):

| model | test PR-AUC (without) | test PR-AUC (with) | Δ |
|---|---|---|---|
| **blend** | 0.3892 | **0.3893** | +0.0001 |
| randomforest | 0.3654 | 0.3632 | −0.0022 |
| xgboost | 0.3068 | 0.3068 | 0.0000 |
| lightgbm | 0.2829 | 0.2829 | 0.0000 |
| catboost | 0.2369 | 0.2416 | +0.0047 |
| logreg_woe | 0.0270 | 0.0270 | 0.0000 |

**Conclusion:** the new feature has **negligible PR-AUC impact**. It is a
*monotonic* transform of `log_num_email_addresses` (already engineered), and
gradient-boosted trees are invariant to monotonic transforms — hence the exact
0.0000 deltas for LightGBM/XGBoost. The only visible effect was a small shift of
the blend's operating point toward higher precision (recall 0.376→0.346,
precision 0.479→0.541; F1/MCC ≈ unchanged). The feature is kept (on by default in
`engineered`; toggle off with `engineered_noemail`) but is not a real
performance gain — consistent with earlier findings that the engineered set
already captures this signal.

Also note: on the **full dataset**, RandomForest (0.365) and the blend (0.389)
are far stronger than on the 100k sample (~0.25) — more data matters a lot here.

### Saved operational model (the blend)

The blend is the best model overall (test PR-AUC **0.389**, F1 0.422, MCC 0.427 at
max-F1). It is now persisted as a single loadable artifact via the serializable
`BlendModel` (`core/blend_model.py`), which packages the calibrated base
estimators plus their weights — so `--save-best` saves the blend when it wins, and
the Inference page/CLI load it like any other version:

```bash
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
    --feature-mode engineered \
    --models lightgbm xgboost catboost randomforest \
    --threshold-mode max_f1 --cv-folds 3 --save-best
# -> Saved best model (blend) -> models/blend__<timestamp>
```

Blend weights: `{lightgbm: 0.3, xgboost: 0.0, catboost: 0.3, randomforest: 0.4}`.

## Ensembles, focal loss & cost threshold (100k sample, engineered)

Comparison with the extended capabilities: blend, logistic **stacking**, CatBoost
**focal** loss, and a **cost-sensitive** threshold (FN = 10× FP):

| model | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc |
|---|---|---|---|---|---|
| **blend** | **0.261** | 0.298 | 0.328 | 0.312 | 0.303 |
| stacking | 0.256 | **0.349** | 0.212 | 0.264 | 0.260 |
| lightgbm | 0.247 | 0.258 | 0.359 | 0.300 | 0.296 |
| catboost | 0.239 | 0.274 | 0.254 | 0.263 | 0.253 |
| xgboost | 0.223 | 0.321 | 0.208 | 0.252 | 0.246 |
| catboost_focal | 0.218 | 0.270 | 0.274 | 0.272 | 0.262 |

Reproduce:

```bash
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
    --sample 100000 --feature-mode engineered \
    --models lightgbm xgboost catboost catboost_focal \
    --stacking --threshold-mode cost --fn-cost 10 --fp-cost 1 --cv-folds 3
```

**Takeaways:** the **blend wins** on PR-AUC; **stacking** trades precision for the
**highest recall** (0.349), useful when catching phishing matters most;
**CatBoost focal** is competitive and even earns weight in the blend. The
LightGBM/XGBoost custom-objective focal variants are unstable at this imbalance
and are excluded from the default roster (see `ML_STACK.md`).

## Full-dataset result (best configuration)

Running the recommended setup on the **entire 524 846-row dataset** (no sampling,
`engineered` features, default boosters + blend, CV=3) gives the strongest
numbers — more data compounds with feature engineering:

| model | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc | threshold |
|---|---|---|---|---|---|---|
| **blend** | **0.311** | 0.324 | 0.388 | 0.353 | 0.347 | 0.174 |
| xgboost | 0.307 | 0.347 | 0.340 | 0.343 | 0.335 | 0.176 |
| lightgbm | 0.283 | 0.331 | 0.302 | 0.316 | 0.307 | 0.155 |
| catboost | 0.237 | 0.230 | 0.337 | 0.273 | 0.270 | 0.192 |

Blend weights: `{lightgbm: 0.5, catboost: 0.0, xgboost: 0.5}`. Reproduce with:

```bash
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
    --feature-mode engineered --cv-folds 3 --save-best
```

### Final operational model (saved version)

Re-running the full-dataset suite with a **recall-target threshold** (recall ≥
0.60, precision floor 0.05) produces the operational model we persist. The best
single model (XGBoost, calibrated, `engineered`) is saved as a version with this
operating point. Held-out **test** performance at the chosen threshold (≈0.060):

| metric | value |
|---|---|
| test PR-AUC | 0.307 |
| test ROC-AUC | 0.884 |
| test recall | 0.556 |
| test precision | 0.123 |
| blend test recall / precision | 0.563 / 0.130 |

Operating point trade-off: lowering the threshold from max-F1 (~0.17) to the
recall-target point (~0.06) raises recall from ~0.35 to ~0.56 at the cost of
precision — the right trade for phishing, where a missed phishing email (false
negative) is costlier than a false alarm. Reproduce + save with:

```bash
uv run python scripts/run_experiments.py --csv data/email_phishing_data.csv \
    --feature-mode engineered --cv-folds 3 \
    --threshold-mode recall_target --recall-target 0.6 --precision-floor 0.05 \
    --save-best
```

The sections below use a 100k stratified sample for fast side-by-side comparison.

## Headline #1: feature engineering is the biggest win

Adding the deterministic `FeatureEngineer` layer (presence flags + log
transforms + ratios) before the model is the single most impactful change.
Same 3 boosters, same 100k sample/split:

| Feature mode (`--feature-mode`) | Best model | Best test PR-AUC | Blend PR-AUC |
|---|---|---|---|
| `raw` | catboost | 0.190 | 0.183 |
| **`engineered`** | lightgbm | **0.247** | **0.252** |

**+33% relative PR-AUC** (0.190 → 0.252) just from feature engineering, before any
extra model tuning. Why it works: the raw counts are heavy-tailed (max
`num_words` ~2.3M) and **zero-inflated** (links/domains 82% zero, emails 69%
zero), so the strongest signal is *presence*, not magnitude — `has_emails` alone
has ~7.7× the mutual information of the best raw feature. This is why
`engineered` is now the default mode.

## Headline #2: encoding strategy comparison (without engineering)

Best **test PR-AUC** by encoding, on the raw counts only:

| Encoding (`--feature-mode`) | Best model | Test PR-AUC | Best blend PR-AUC |
|---|---|---|---|
| **raw** | blend | **0.209** | 0.209 |
| binned_woe | xgboost | 0.111 | 0.080 |
| quantile | xgboost | 0.101 | 0.082 |
| target | catboost | 0.043 | 0.036 |

**Takeaway:** for these integer **count** features, `raw` (letting the gradient
boosters split internally) beats every pre-discretising encoding — WOE/quantile
bins throw away resolution the boosters exploit, and target encoding collapses
too much signal. So the recommended pipeline is **engineering on raw counts**
(`engineered`), not engineering + an extra encoding. The encodings remain
selectable (incl. `engineered_<enc>` combos) for validation.

## Detailed tables

### `engineered` (recommended default) — blend wins

| model | cv_pr_auc | val_pr_auc | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc | threshold |
|---|---|---|---|---|---|---|---|---|
| **blend** | – | 0.219 | **0.252** | 0.274 | 0.319 | 0.295 | 0.287 | 0.153 |
| lightgbm | 0.198 | 0.205 | 0.247 | 0.258 | 0.359 | 0.300 | 0.297 | 0.196 |
| catboost | 0.187 | 0.168 | 0.239 | 0.274 | 0.254 | 0.263 | 0.253 | 0.145 |
| xgboost | 0.196 | 0.195 | 0.223 | 0.298 | 0.246 | 0.269 | 0.260 | 0.133 |

Blend weights: `{lightgbm: 0.5, catboost: 0.0, xgboost: 0.5}`

### `raw` — blend wins (no engineering)

| model | cv_pr_auc | val_pr_auc | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc | threshold |
|---|---|---|---|---|---|---|---|---|
| **blend** | – | 0.156 | **0.209** | 0.310 | 0.218 | 0.256 | 0.248 | 0.087 |
| randomforest | 0.158 | 0.134 | 0.202 | 0.187 | 0.416 | 0.258 | 0.272 | 0.208 |
| catboost | 0.154 | 0.105 | 0.190 | 0.254 | 0.205 | 0.227 | 0.216 | 0.104 |
| lightgbm | 0.153 | 0.139 | 0.175 | 0.194 | 0.343 | 0.248 | 0.251 | 0.143 |
| xgboost | 0.148 | 0.129 | 0.174 | 0.242 | 0.242 | 0.242 | 0.232 | 0.120 |
| logreg_woe | 0.028 | 0.028 | 0.024 | 0.218 | 0.025 | 0.044 | 0.036 | 0.027 |

Blend weights: `{lightgbm: 0.2, xgboost: 0.5, catboost: 0.0, logreg_woe: 0.1, randomforest: 0.2}`

### `binned_woe`

| model | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc |
|---|---|---|---|---|---|
| xgboost | 0.111 | 0.135 | 0.170 | 0.150 | 0.141 |
| catboost | 0.109 | 0.151 | 0.168 | 0.159 | 0.149 |
| lightgbm | 0.095 | 0.187 | 0.160 | 0.173 | 0.161 |
| blend | 0.080 | 0.071 | 0.194 | 0.104 | 0.111 |
| randomforest | 0.071 | 0.111 | 0.151 | 0.128 | 0.120 |
| logreg_woe | 0.024 | 0.218 | 0.025 | 0.044 | 0.036 |

### `quantile`

| model | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc |
|---|---|---|---|---|---|
| xgboost | 0.101 | 0.131 | 0.213 | 0.162 | 0.158 |
| catboost | 0.101 | 0.222 | 0.087 | 0.125 | 0.121 |
| lightgbm | 0.093 | 0.187 | 0.109 | 0.138 | 0.128 |
| blend | 0.082 | 0.087 | 0.195 | 0.121 | 0.123 |
| randomforest | 0.065 | 0.099 | 0.120 | 0.109 | 0.098 |
| logreg_woe | 0.024 | 0.218 | 0.025 | 0.044 | 0.036 |

### `target`

| model | test_pr_auc | test_recall | test_precision | test_f1 | test_mcc |
|---|---|---|---|---|---|
| catboost | 0.043 | 0.187 | 0.049 | 0.077 | 0.072 |
| xgboost | 0.038 | 0.190 | 0.039 | 0.065 | 0.059 |
| lightgbm | 0.037 | 0.083 | 0.063 | 0.072 | 0.058 |
| randomforest | 0.037 | 0.175 | 0.037 | 0.060 | 0.053 |
| blend | 0.036 | 0.183 | 0.041 | 0.068 | 0.061 |
| logreg_woe | 0.024 | 0.218 | 0.025 | 0.044 | 0.036 |

> `autoencoder` mode is available (`--feature-mode autoencoder`) but is not part
> of this headline comparison: it is unsupervised, non-deterministic and far less
> interpretable, and is provided only so it can be validated on demand.

## Notes & caveats

- This is the **best operating point per (encoding, model)** at max-F1 on a 100k
  sample. Absolute PR-AUC values are modest because phishing here is genuinely
  hard at ~1.3% with only 8 count features; the **relative** ranking across
  encodings/models is the actionable signal.
- For a production candidate, re-run on the **full dataset** without `--sample`
  and with a higher `--cv-folds`, choosing the threshold mode that matches the
  business cost (e.g. `--threshold-mode recall_target --recall-target 0.9`).
- All `raw` runs were logged to MLflow (`./mlruns`); open with `uv run mlflow ui`.
