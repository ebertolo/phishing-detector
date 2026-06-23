"""Model layer: one file per algorithm.

Each module exposes a common interface so the experiment runner and generic
wrapper stay algorithm-agnostic:

- ``NAME``                      : str identifier.
- ``build(feature_mode, y)``    : returns an unfitted sklearn/imblearn Pipeline.
- ``param_grid()``              : dict of GridSearchCV parameters (pipeline-prefixed).

``feature_mode`` is ``"raw"`` (integer counts straight into the model) or
``"binned_woe"`` (optimal-binning + WOE front-end). Imbalance is handled inside
each estimator (scale_pos_weight / class weights), so no resampling is required.
"""

from . import (
    adaboost_model,
    catboost_focal_model,
    catboost_model,
    cluster_model,
    lightgbm_focal_model,
    lightgbm_focal_native_model,
    lightgbm_model,
    logreg_model,
    logreg_woe_model,
    randomforest_model,
    tensorflow_dnn_model,
    xgboost_focal_model,
    xgboost_focal_native_model,
    xgboost_model,
)

# Registry consumed by the experiment runner and the Streamlit UI.
#
# CatBoost focal uses CatBoost's native, well-tested Focal loss and performs
# competitively. The LightGBM/XGBoost focal variants use a hand-rolled custom
# objective; it is mathematically correct on balanced/clean data but proved
# unstable at this dataset's ~1.3% imbalance (it collapses to predicting the
# positive class). They are kept available for experimentation but are NOT in
# the default roster. Prefer class-weighted boosters + ``engineered`` features.
ALL_MODELS = {
    lightgbm_model.NAME: lightgbm_model,
    xgboost_model.NAME: xgboost_model,
    catboost_model.NAME: catboost_model,
    logreg_woe_model.NAME: logreg_woe_model,
    randomforest_model.NAME: randomforest_model,
    catboost_focal_model.NAME: catboost_focal_model,
    lightgbm_focal_model.NAME: lightgbm_focal_model,
    xgboost_focal_model.NAME: xgboost_focal_model,
    lightgbm_focal_native_model.NAME: lightgbm_focal_native_model,
    xgboost_focal_native_model.NAME: xgboost_focal_native_model,
    logreg_model.NAME: logreg_model,
    cluster_model.NAME: cluster_model,
    adaboost_model.NAME: adaboost_model,
    tensorflow_dnn_model.NAME: tensorflow_dnn_model,
}

# Recommended default roster: the strongest gradient boosters with class
# weighting. Empirically these dominate on this data; the other models stay
# available for comparison but are not run by default to keep runs lean.
DEFAULT_MODELS = [lightgbm_model.NAME, catboost_model.NAME, xgboost_model.NAME]

__all__ = ["ALL_MODELS", "DEFAULT_MODELS"]
