# Experiment Journey — How the Model Evolved

This document explains how the phishing detector was built: the nature of the
problem, the families of techniques considered, what was **kept**, what was
**discarded**, and *why*. It is organised **thematically** (not in the
chronological order the experiments happened) so each topic reads as one story.
Discarded approaches are documented on purpose — each taught something about this
problem (extreme imbalance, few integer count features, weak per-feature signal).

Primary metric throughout: **PR-AUC (average precision)**. Numbers are **test
PR-AUC** on a stratified holdout; "100k" = a stratified 100 000-row sample for
fast iteration, "full" = the entire 524 846-row dataset.

---

## 1. Data nature, candidate algorithms, and evaluation metrics

### The dataset and why it is hard

524 846 emails, **1.32% phishing**, 8 integer count features (`num_words`,
`num_unique_words`, `num_stopwords`, `num_links`, `num_unique_domains`,
`num_email_addresses`, `num_spelling_errors`, `num_urgent_keywords`) + label
`label_1`. The features are **heavy-tailed** (max `num_words` ≈ 2.3M) and
**zero-inflated** (links/domains 82% zero, emails 69% zero). Per-feature mutual
information is tiny (max ≈ 0.0045) — the core difficulty of the whole project: no
single feature separates the classes, so signal must come from combinations.

### Metrics chosen for this nature (imbalanced, rare-positive)

Because only ~1.3% are positive, **accuracy is meaningless** (predicting "legit"
always scores ~98.7%). The framework reports the imbalance-aware set:

- **PR-AUC (average precision)** — the primary, threshold-independent ranking
  metric; the most informative under extreme imbalance.
- **Precision, Recall, F1** on the positive (phishing) class.
- **MCC** — balanced correlation, robust when classes are skewed.
- **ROC-AUC** — secondary (optimistic under heavy imbalance, so not headline).
- **Confusion matrix** at a deliberately chosen operating threshold.

### Candidate algorithms / strategies considered

- **Gradient boosters** (LightGBM, XGBoost, CatBoost) — the expected winners on
  tabular data; imbalance handled with class weights / `scale_pos_weight`.
- **Interpretable baselines** — RandomForest, logistic regression (plain and over
  WOE bins).
- **Imbalance handling** — class weights vs resampling. The study
  ["Do we need rebalancing strategies?" (arXiv 2402.03819)](https://arxiv.org/abs/2402.03819)
  shows *not* rebalancing is competitive with tree models and SMOTE tends to hurt
  under extreme imbalance → **decision: class weights, not SMOTE.**
- **Ensembling** — weighted blend and logistic stacking of calibrated models.
- **Deep tabular** — a CPU dense net and its reusable embedding; clustering and
  AdaBoost as comparison points.

**Baseline (KEPT):** stratified split, class-weighted boosters + RF + logreg,
GridSearch/StratifiedKFold scored by PR-AUC, calibration, blend, threshold tuning
→ raw counts blend ≈ **0.21** (100k). A solid, honest starting point.

---

## 2. Feature engineering — versions and the transformation sequence

Feature engineering proved the **single biggest lever** (more than model choice),
because the raw signal is so weak. It evolved in stages.

### v0 — Encodings on the raw counts (mostly DISCARDED)

Five selectable encodings tested first: `raw`, optimal-binning + WOE, quantile
bins, cross-fit target encoding, autoencoder latent. Best PR-AUC (100k):

| encoding | best PR-AUC |
|---|---|
| **raw** | **0.209** |
| binned_woe | 0.111 |
| quantile | 0.101 |
| target | 0.043 |

**Lesson:** for integer **count** features, pre-discretising throws away
resolution the boosters split on — `raw` beats every pre-encoding. WOE is kept
only for the explainable `logreg_woe` baseline; encodings stay selectable but are
not the path forward.

### v1 — `FeatureEngineer` (KEPT — biggest single win)

A deterministic, leakage-safe transformer adding presence flags, log transforms
and ratios. Data analysis showed *presence* carries far more signal than
magnitude — `has_emails` alone has ~7.7× the MI of the best raw feature. Result:
**+33% relative** over raw (0.190 → 0.252, 100k; blend 0.311 full). `engineered`
became the default feature mode.

### v2 — The full 52-feature catalogue (KEPT)

`FeatureEngineer` ([engineering.py](../src/phishing/features/engineering.py)) now
turns the **8 raw counts** into **52 features**. The transformation **sequence**
applied row-wise (only the logz stats and the p95 density thresholds are learned
on train; every feature is divide-by-zero safe — `num_words` denominators are
`+eps` and floored at 1, per-X ratios use `max(x, 1)`):

| group | features | why |
|---|---|---|
| **Presence flags** | `has_links`, `has_domains`, `has_emails`, `has_urgent` | Counts are zero-inflated; *whether* something is present is the strongest single signal. |
| **Graded flags** | `has_multiple_links`, `has_multiple_domains`, `has_spelling_errors`, `many_spelling_errors` (≥3), `many_urgent_keywords` (≥2), `short_email` (<50), `very_short_email` (<20), `long_email` (>300) | Non-linear thresholds (phishing tends to be short / urgent). |
| **Log transforms** | `log_num_*` for all 8 counts | Tame the heavy tails so magnitude is usable. |
| **Length densities** | `lexical_diversity`, `word_repetition_ratio`, `stopword_ratio`, `content_word_ratio`, `link_density`, `email_density`, `spelling_error_ratio`, `urgency_ratio` | Normalise each count by message length. |
| **Structural ratios** | `domain_per_link_ratio`, `links_per_domain_ratio`, `emails_per_link_ratio`, `emails_per_domain_ratio`, `spelling_errors_per_link`, `urgent_keywords_per_link`, `spelling_errors_per_unique_word`, `urgency_per_unique_word`, `text_complexity` | Relate features to *each other* (phishing-structure cues). |
| **Content-word intensities** | `link_to_content_ratio`, `email_to_content_ratio`, `error_to_content_ratio`, `urgency_to_content_ratio` | Densities over *non-stopword* (content) words. |
| **High-density flags** | `high_link_density`, `high_urgency_density`, `high_error_density` | Mark the top 5% (train-learned p95 threshold) of each density. |
| **Pairwise interactions** | `links_x_urgency`, `links_x_errors`, `domains_x_urgency`, `domains_x_errors`, `errors_x_urgency`, `emails_x_links`, `emails_x_domains` | Products of risk-bearing counts that combine weak signals. |
| **Leakage-safe z-score** | `logz_num_email_addresses` | log + train-fit z-score of the top feature. |

Expanding to the full catalogue lifted the tuned blend **0.432 → 0.442** (full
dataset). On 100k with grid search it was flat (0.252) — richer features pay off
only with enough data + a wide search (the same data-hungry pattern as the
embedding).

### Feature ideas that were DISCARDED

- **Top-3 features only** → PR-AUC collapsed to **0.020** (3 binary flags = 8
  distinct inputs, too coarse). *High individual MI ≠ predictive power; signal is
  in the combination.*
- **Extra weighted sums / z-scores** → **zero** PR-AUC gain. *Boosters are
  invariant to monotonic transforms; z-scoring a log feature is redundant.*
- **Smoothing / denoising** (`engineered_smooth` 0.236, `engineered_denoise`
  0.145) → both *below* plain `engineered` (0.247). *Boosters are already robust
  to outliers; smoothing removes resolution.* Kept selectable for validation.

---

## 3. Models tested — discarded, kept, and why; ensembling & calibration

### Kept

- **Gradient boosters (LightGBM, XGBoost, CatBoost)** — dominate here. With weak
  per-feature signal, tree ensembles combine many faint cues better than any
  linear or shallow model. XGBoost is the strongest single model.
- **CatBoost focal loss** — its *native* focal implementation works and even earns
  weight in the blend.

### Discarded (kept in the codebase as baselines)

| model | PR-AUC | why discarded |
|---|---|---|
| logreg | 0.069 | linear; can't capture the feature interactions |
| adaboost | 0.021 | depth-1 stumps abort under imbalance; NaN folds |
| cluster (KMeans/GMM) | 0.013 | unsupervised split ≈ base rate; classes overlap |
| tensorflow_dnn | ~0.08 | weak ceiling; slow on CPU (but its *embedding* helps — §4) |
| LightGBM/XGBoost focal | unstable | hand-rolled custom objective collapses at ~1.3% imbalance (all-positive, NaN CV) — **and the native-API rewrite (`lgb.train`/`xgb.train`) collapses too**, so the cause is the imbalance, not the sklearn wrapper. *Custom gradient objectives are fragile; prefer tested implementations.* |

### Ensembling & calibration (KEPT)

- **Blend** — weighted average of *calibrated* base probabilities; the consistent
  winner. Packaged as a serializable `BlendModel` so the winning blend saves/loads
  like any single model.
- **Stacking** — a logistic meta-model over base probabilities; ~equal PR-AUC to
  the blend, usually a touch more recall. Kept as an option.
- **Calibration** — `CalibratedClassifierCV` over a `FrozenEstimator`, sigmoid
  (default) or isotonic; blending/thresholds then act on trustworthy probabilities.

Why the blend wins: the boosters make partly *independent* errors, so averaging
their calibrated scores reduces variance; calibration first ensures the averages
are meaningful.

---

## 4. Training optimisation — splits, search, and rare-positive focus

This group covers everything about *how the models are trained and selected*.

### Splits

- Stratified **train / validation / test**, preserving the ~1.3% rate in each.
  **Validation** calibrates and tunes the threshold + blend weights; **test** is
  a clean holdout used only for the final confusion matrix.
- Two ratios compared: original **76/19/5** and **90/5/5** (more training data).
  Moving 76% → 90% training added ~0.03 PR-AUC — the model was training-data
  limited. Splits of 98/2 and 99/1 were **discarded**: a 1–2% holdout has only
  ~69–138 phishing, too few for stable metrics.

### Hyperparameter search — GridSearch → RandomizedSearch (big win)

The original grids were tiny (16/16/8 combos). Widening them and switching to
**`RandomizedSearchCV`** (40 samples over `n_estimators`, `learning_rate`, depth /
`num_leaves`, `subsample`, `colsample`, regularisation) lifted the blend
**0.367 → 0.432 (+0.065)** — the **largest single jump after feature
engineering**. The boosters were genuinely undertuned. CV uses StratifiedKFold,
scored by PR-AUC (`average_precision`).

### Focus on the rare positive class

- **Class weights / `scale_pos_weight`** on every booster so the rare class is not
  ignored — the chosen alternative to resampling (SMOTE hurts here).
- **PR-AUC as the refit metric** — selection optimises the positive class directly.
- **Threshold tuning** on validation: recall-target with precision floor, max-F1,
  manual, and **cost-sensitive** (`fn_cost`/`fp_cost`, default FN = 10× FP) — the
  operating point is a business choice, not a fixed 0.5. Raising the cost to
  **FN = 20×** on the tuned XGBoost pushed **recall to 0.527** (from ~0.36 at
  max-F1) at precision ~0.18; PR-AUC is unchanged — the threshold only slides
  along the curve. Use `--fn-cost` to pick the production point by business cost.
- **Focal loss** (down-weighting easy negatives) was explored in this same spirit;
  **only CatBoost's native focal works** — the hand-rolled LightGBM/XGBoost
  objective collapses at ~1.3% imbalance, and so does a native-API
  (`lgb.train`/`xgb.train`) rewrite, so the instability is the imbalance itself,
  not the sklearn wrapper.

### Reproducibility

All seeds fixed — sklearn `random_state=42`, TensorFlow via
`keras.utils.set_random_seed(42)` — so every result above is reproducible.

---

## 5. NN embedding — the net is weak, but its embedding helps

The standalone dense net is a weak classifier here (~0.08 PR-AUC). The valuable
part is its **20-neuron embedding layer**: a learned non-linear projection of the
features. Even when the net itself can't win, the embedding can **abstract
non-linear feature combinations that the gradient boosters do not capture on their
own**, and feeding those 20 values back to the boosters as extra features helps.

**A leakage lesson first.** The initial `engineered_nnembed` mode retrained the
net **inside every CV fold**, showing an apparent **+50%** lift — but that was
leakage (the net saw each fold's validation labels through its loss). The fix:
`scripts/embedding_experiment.py` trains the embedding **once on the train split
only** (SGD lr=0.005, momentum schedule 0.5→0.95, early stopping), freezes it, and
reuses the 20 features — no per-fold retraining.

**The gain is real once trained correctly, and scales with data:**

| dataset | embedding train size | no embedding | + frozen embedding |
|---|---|---|---|
| 100k sample | ~76k rows, ~1000 pos | blend 0.2555 | 0.2376 (worse) |
| **full 524k** | ~400k rows, ~5300 pos | blend 0.3112 | **0.3286** (better) |

**Per-model check (52 features, full, with vs without the embedding):**

| model | without embedding | with embedding | Δ |
|---|---|---|---|
| **xgboost** | 0.4235 | **0.4385** | +0.0150 |
| lightgbm | 0.4249 | 0.4308 | +0.0059 |
| catboost | 0.3481 | 0.3648 | +0.0167 |

The embedding helps **every** booster — confirming per-model (not just in the
blend) that it adds signal the trees miss.

**Architecture is already at its sweet spot.** Two rounds of architecture search
found no improvement over the simple **dim 20 / dropout 0.4** embedding:

- A *periodic* (Fourier) embedding at dim 32 + cosine LR made the blend slightly
  *worse* (0.442 → 0.413).
- A dropout × width sweep (second layer 20→32, embedding_dim ∈ {16, 32}, dropout
  ∈ {0.5–0.75}, `embedding_arch_sweep.py`) tied on 100k (~0.22) and, on the full
  dataset, gave xgboost **0.4381** (dropout 0.5, dim 32) vs the **0.4385**
  baseline — i.e. no gain. The embedding train/val gaps were tiny and *negative*
  (no overfit to regularise away), explaining why more dropout did not help.

**Lessons: never retrain a supervised feature generator per fold; judge
data-hungry generators on the full dataset, not a sample; and a well-tuned simple
embedding can already be at its ceiling — deeper/wider nets do not beat tuned
GBDTs here.**

---

## 6. Best models — final results and confusion matrices

The best configuration combines everything above: **tuned boosters
(RandomizedSearch) on the 52-feature catalogue + the frozen 20-dim embedding**,
90/5/5 split, threshold tuned for max-F1, reproducible seeds. On the 5% test
holdout (26 227 emails, 347 phishing):

| strategy | PR-AUC | precision | recall | F1 | MCC | threshold |
|---|---|---|---|---|---|---|
| **blend** | **0.442** | 0.630 | 0.349 | 0.449 | 0.464 | 0.291 |
| xgboost | 0.438 | 0.627 | 0.363 | 0.460 | 0.472 | 0.299 |
| stacking | 0.430 | 0.484 | 0.441 | 0.462 | 0.455 | 0.977 |

Confusion matrices (rows = actual, columns = predicted):

| blend | legit | phishing |   | xgboost | legit | phishing |   | stacking | legit | phishing |
|---|---|---|---|---|---|---|---|---|---|---|
| **legit** | TN 25 809 | FP 71 |   | **legit** | TN 25 805 | FP 75 |   | **legit** | TN 25 717 | FP 163 |
| **phishing** | FN 226 | TP 121 |   | **phishing** | FN 221 | TP 126 |   | **phishing** | FN 194 | TP 153 |

**Reading the best result:**

- **Blend** maximises PR-AUC (0.442) and flags the fewest legit emails (71 FP) —
  best when false alarms are costly. **XGBoost** is the best single model and has
  the best F1/MCC (0.460 / 0.472). **Stacking** catches the most phishing (153 TP,
  recall 0.44) at more false alarms — best when missing phishing is costliest.
- **False negatives** (~60% of phishing missed at max-F1) are the costlier error
  class. In production, lower the threshold via `--threshold-mode recall_target`
  (e.g. recall ≥ 0.6) or `--threshold-mode cost` (FN = 10× FP): this moves mass
  from FN to FP. PR-AUC is fixed by the model; only the FP/FN split moves.

**The full progression:** 0.21 raw → 0.25 engineered → 0.31 blend(full) → 0.33
+embedding → 0.37 (90/5/5) → 0.43 (tuned boosters) → **0.44 (52-feature
catalogue)**. Modest in absolute terms — phishing is genuinely hard with 8 weak
count features — but every step is a real, leakage-safe gain. Reproduce with
`uv run python scripts/best_model_report.py --csv data/email_phishing_data.csv
--search random --n-iter 40 --stacking`. The hyperparameter search and the NN
embedding training are now cached (see `param_cache`/`embedding_cache` in
[DESIGN.md](DESIGN.md#caching)), so a repeated run with the same configuration
finishes in roughly a minute instead of close to an hour — see
[RESULTS.md](RESULTS.md#caching) for measured timings.

---

## 7. Interpreting the model

This section answers two practical questions for anyone consuming the model's
output: which metrics to look at, and which features are actually driving the
predictions.

### Reading the output

The model's primary output is a continuous probability (the predicted
likelihood that a sample is phishing), not a forced binary label — both the
CLI (`phishing_proba` column) and the REST API (`phishing_likelihood` field,
see [`fastapi_app/README.md`](../fastapi_app/README.md)) expose it directly. A
binary label is only meaningful once an operating threshold is chosen (§4,
"Focus on the rare positive class"); the same probability can be cut at
different thresholds for different cost trade-offs.

To judge the quality of that probability and of any downstream decision, three
kinds of metrics matter, in order:

1. **PR-AUC (average precision)** — threshold-independent; measures how well
   the model ranks phishing above legitimate email across all possible cutoffs.
   This is the metric used to select and tune models throughout this document,
   and the only one safe to compare across different threshold choices.
2. **Precision, recall, F1, MCC at the chosen threshold** — once an operating
   point is fixed, these describe what actually happens in production: how
   many flagged emails are real phishing (precision), how much phishing is
   caught (recall), and a single balanced figure (F1/MCC). See
   [RESULTS.md](RESULTS.md#headline--best-result-so-far) for the current
   numbers at each threshold mode.
3. **Confusion matrix** — the raw counts behind those rates (true/false
   positives and negatives), useful for sizing the operational cost of false
   alarms versus missed phishing in absolute terms, not just rates.

Accuracy is deliberately excluded throughout (§1) because at ~1.3% positives
it stays near 98.7% regardless of whether the model does anything useful.

### Which features drive the predictions

The framework ships two feature-attribution methods
(`src/phishing/features/selection.py`, exposed on the Streamlit "Importance"
page): permutation-free **RandomForest impurity importance** and **mutual
information**, both computed against the 52-feature engineered set. Run on a
stratified 100k sample of the full dataset, the two methods agree on the
broad picture but disagree on specifics, which is itself informative:

**RandomForest importance** (top contributors):

| feature | importance |
|---|---|
| `word_repetition_ratio` | 0.065 |
| `lexical_diversity` | 0.065 |
| `text_complexity` | 0.060 |
| `content_word_ratio` | 0.057 |
| `stopword_ratio` | 0.054 |
| `num_unique_words` | 0.050 |
| `spelling_errors_per_unique_word` | 0.046 |
| `num_words` | 0.046 |

**Mutual information** (top contributors):

| feature | mutual information |
|---|---|
| `many_spelling_errors` | 0.0210 |
| `has_spelling_errors` | 0.0162 |
| `logz_num_email_addresses` | 0.0094 |
| `has_emails` | 0.0075 |
| `short_email` | 0.0045 |
| `long_email` | 0.0035 |

RandomForest importance is dominated by **density and ratio features**
(repetition, lexical diversity, content-word ratio) — these are continuous
signals the trees can split on repeatedly across many nodes, so impurity-based
importance naturally favours them. Mutual information instead ranks
**presence/threshold flags** highest (`many_spelling_errors`,
`has_spelling_errors`, `has_emails`) — these summarise the same underlying
signal as a single bit, which the univariate MI calculation rewards. Both
views are consistent with the central finding of §2: no individual raw count
separates the classes (max raw MI ≈ 0.0045), but the *engineered* features —
both the density ratios and the presence flags built from the same raw counts
— carry meaningfully more signal once message length and word composition are
normalised. Spelling-error density and email-address presence are the most
consistent signal across both methods.

These rankings describe **average, dataset-wide** feature contribution, not a
per-prediction explanation (no SHAP/LIME is used in this project). For
debugging an individual prediction, the confusion-matrix breakdown combined
with the raw input counts is the available tool today; per-sample attribution
is listed as an open item in [Next_Steps.md](Next_Steps.md).

---

## The lessons, distilled

1. **Feature engineering > model tuning** when per-feature signal is weak.
2. **Don't rebalance** (no SMOTE); use class weights + boosters + PR-AUC.
3. **High MI ≠ predictive value**; weak features matter in combination, not alone.
4. **Boosters are invariant to monotonic transforms** — z-scores/scaling add nothing.
5. **Never retrain a supervised feature generator per CV fold** (leakage).
6. **Judge data-hungry generators on the full dataset**, not a sample.
7. **Custom gradient objectives are fragile under extreme imbalance.**
8. **Hyperparameter tuning matters** — wide RandomizedSearch was the second-biggest lever.
9. **The operating threshold is a deliberate business choice.**

## Lines of investigation that are closed

These were run to completion, with results recorded above, and are not
expected to be revisited without new evidence:

- **Embedding architecture.** Two sweeps (periodic/Fourier embedding at dim
  32 with cosine LR; a dropout × width grid over the second layer and the
  embedding dimension) both matched or trailed the simple dim-20/dropout-0.4
  baseline (§5). The embedding is at its practical ceiling for this feature
  set; widening or regularising it further is not expected to help.
- **Native-API focal loss.** Rewriting the custom LightGBM/XGBoost focal
  objective against the native `lgb.train`/`xgb.train` APIs still collapses
  at this dataset's ~1.3% imbalance (§3, §4) — the instability is the
  imbalance itself, not the sklearn-wrapper objective.
- **Operating-point sweep at FN=20×.** Confirmed the cost-sensitive threshold
  trades precision for recall as expected (recall ~0.53 vs ~0.36 at max-F1,
  full dataset, §4) without changing PR-AUC.

Everything in this section was implemented and measured; for proposals that
have **not** been attempted yet, see the prioritised, impact/effort-rated
backlog in [Next_Steps.md](Next_Steps.md) (richer raw data, isotonic
calibration on the full dataset, repeated/larger CV, and other open items).

---

For the concrete numbers behind every claim, see [RESULTS.md](RESULTS.md); for
the libraries and techniques, see [ML_STACK.md](ML_STACK.md); for the
architecture — including the caching layer and the REST API — see
[DESIGN.md](DESIGN.md); for what remains open, prioritised by impact, effort
and risk, see [Next_Steps.md](Next_Steps.md).
