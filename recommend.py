"""Step 11: top-N recommendations per user.

python recommend.py                      # global top 10 (GA weights, Triangular)
python recommend.py --user 1             # top 10 for user 1
python recommend.py --user 1 --n 20 --source baseline --family Gaussian

For a user we score every title, drop the movies they already rated in the TRAIN
split (they have "seen" those), and return the N highest-scored titles. Movies the
user rated in the held-out TEST split are kept as candidates, so we can show which
recommendations the user really watched later, and what they rated them.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from evaluation import load_data, LIKE_THRESHOLD
from fuzzy_system import fuzzy_score, BASELINE_WEIGHTS, FAMILIES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
COLS = ["primaryTitle", "startYear", "genres", "rating", "numVotes", "score"]


def load_weights(family="Triangular", source="ga"):
    if source == "baseline":
        return BASELINE_WEIGHTS
    result = json.loads((OUT / f"ga_{family}.json").read_text())
    best = max(result["runs"], key=lambda r: r["train_p"])
    return np.array(best["weights"])


def recommend(user_id, items, ratings, n=10):
    """Top-n titles for one user. `items` must already have a 'score' column."""
    user = ratings[ratings["userId"] == user_id]
    if user.empty:
        raise ValueError(f"user {user_id} not found in the ratings")

    seen = user.loc[user["split"] == "train", "movieId"]
    candidates = items[~items["movieId"].isin(seen)]
    top = candidates.sort_values(["score", "numVotes"], ascending=False).head(n).copy()

    # held-out check: did the user rate any of these later (test split)?
    later = user.loc[user["split"] == "test"].set_index("movieId")["rating"]
    top["test_rating"] = top["movieId"].map(later)
    return top


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Top-N fuzzy-GA movie recommendations")
    ap.add_argument("--user", type=int, help="MovieLens userId (omit for the global top N)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--family", default="Triangular", choices=list(FAMILIES))
    ap.add_argument("--source", default="ga", choices=["ga", "baseline"])
    args = ap.parse_args()

    items, ratings, _ = load_data()
    w = load_weights(args.family, args.source)
    items["score"] = fuzzy_score(items, w, args.family)
    print(f"Weights: {args.source} | family: {args.family}")

    if args.user is None:
        top = items.sort_values(["score", "numVotes"], ascending=False).head(args.n)
        print(f"\nGlobal top {args.n}:")
        print(top[COLS].round(3).to_string(index=False))
    else:
        top = recommend(args.user, items, ratings, args.n)
        n_train = ((ratings["userId"] == args.user) & (ratings["split"] == "train")).sum()
        print(f"\nTop {args.n} for user {args.user} ({n_train} train ratings excluded as already seen):")
        print(top[COLS + ["test_rating"]].round(3).to_string(index=False, na_rep="-"))

        hits = top["test_rating"].notna()
        liked = (top["test_rating"] >= LIKE_THRESHOLD).sum()
        print(f"\nWatched later (test split): {hits.sum()} of {args.n} | rated >= {LIKE_THRESHOLD}: {liked}")
