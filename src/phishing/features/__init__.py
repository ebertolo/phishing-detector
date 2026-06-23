"""Feature-engineering layer.

Leakage-safe transformers (fit on training folds only) that turn integer count
features into semantic ranges: optimal binning + WOE, quantile discretisation,
and feature selection. All are sklearn-compatible so they compose inside the
sklearn / imbalanced-learn pipeline used by the generic wrapper.
"""
