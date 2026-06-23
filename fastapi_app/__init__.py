"""REST API serving the trained phishing-detection model.

Thin serving layer over the reusable ``phishing`` core package
(``src/phishing``) — it loads a saved ``ModelWrapper`` version and exposes its
``predict_proba`` through HTTP. No modeling logic lives here.
"""
