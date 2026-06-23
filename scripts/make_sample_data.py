"""Generate a synthetic, highly imbalanced phishing dataset for demos/tests.

Produces integer count features matching the project schema with a ~1% phishing
rate. Phishing rows are shifted toward more links/domains/urgent keywords/spelling
errors so the signal is learnable. Writes ``data/sample_phishing.csv``.

Run: ``uv run python scripts/make_sample_data.py``
"""

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phishing.core.data import FEATURES, TARGET


# %%
def make_dataset(n: int = 8000, pos_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Synthetic integer-feature dataset with a rare phishing class."""
    rng = np.random.default_rng(seed)
    n_pos = max(int(n * pos_rate), 2)
    n_neg = n - n_pos

    def legit(size):
        return {
            "num_words": rng.poisson(180, size),
            "num_unique_words": rng.poisson(120, size),
            "num_stopwords": rng.poisson(60, size),
            "num_links": rng.poisson(1, size),
            "num_unique_domains": rng.poisson(1, size),
            "num_email_addresses": rng.poisson(1, size),
            "num_spelling_errors": rng.poisson(1, size),
            "num_urgent_keywords": rng.poisson(0, size),
        }

    def phish(size):
        return {
            "num_words": rng.poisson(90, size),
            "num_unique_words": rng.poisson(70, size),
            "num_stopwords": rng.poisson(25, size),
            "num_links": rng.poisson(6, size),
            "num_unique_domains": rng.poisson(4, size),
            "num_email_addresses": rng.poisson(3, size),
            "num_spelling_errors": rng.poisson(6, size),
            "num_urgent_keywords": rng.poisson(4, size),
        }

    neg = pd.DataFrame(legit(n_neg))
    pos = pd.DataFrame(phish(n_pos))
    neg[TARGET] = 0
    pos[TARGET] = 1
    df = pd.concat([neg, pos], ignore_index=True).sample(frac=1.0, random_state=seed)
    return df[FEATURES + [TARGET]].reset_index(drop=True)


# %%
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    df = make_dataset()
    path = out_dir / "sample_phishing.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} — {len(df):,} rows, phishing rate {df[TARGET].mean():.4%}")
