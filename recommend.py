import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from evaluation import load_data
from fuzzy_system import fuzzy_score, BASELINE_WEIGHTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"


def load_weights(family="Triangular", source="ga"):
    if source == "baseline":
        return BASELINE_WEIGHTS
    result = json.loads((OUT / f"ga_{family}.json").read_text())
    best = max(result["runs"], key=lambda r: r["train_p"])
    return np.array(best["weights"])


def recommend(user_id, items, ratings, n=10):
    """Top-n highest-scoring movies the user has not rated in their train split.
    `items` must already have a "score" column."""
    seen = ratings.loc[(ratings["userId"] == user_id) & (ratings["split"] == "train"), "item_idx"]
    unseen = items[~items["item_idx"].isin(seen)]
    return unseen.sort_values("score", ascending=False).head(n)


if __name__ == "__main__":
    items, ratings, _ = load_data()
    user_id = 1
    cols = ["primaryTitle", "startYear", "rating", "numVotes", "score"]
    for source in ("ga", "baseline"):
        items["score"] = fuzzy_score(items, load_weights(source=source), "Triangular")
        recs = recommend(user_id, items, ratings)
        print(f"\nTop 10 for user {user_id} ({source} weights):")
        print(recs[cols].round(3).to_string(index=False))
