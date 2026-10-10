"""Step 9a: evaluation / fitness for the fuzzy recommender.

Idea: each title gets ONE fuzzy score (built only from IMDb features, so no MovieLens
rating ever leaks into it). We then ask: for each MovieLens user, does ranking the
movies THEY rated by that score put the movies they actually liked near the top?

  * Split: for every user, ratings are sorted by time. The earliest 70% are "train"
    (the GA will tune weights on these); the latest 30% are "test" (final report only).
  * Spearman: per-user rank correlation between score and the user's own ratings, averaged.
  * Precision@K: per-user share of the top-K scored movies that the user rated >= 4.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fuzzy_system import fuzzy_score, BASELINE_WEIGHTS, FAMILIES, RULES

ROOT = Path(__file__).resolve().parent
K = 10
LIKE_THRESHOLD = 4.0
TRAIN_FRAC = 0.7
MIN_ITEMS = K          # users with fewer rated movies in a split are skipped


def load_data():
    items = pd.read_csv(ROOT / "data/processed/ml_items.csv.gz")
    items = items.drop_duplicates("movieId").reset_index(drop=True)
    items["item_idx"] = np.arange(len(items))
    ratings = pd.read_csv(ROOT / "data/movielens/ml-latest-small/ratings.csv")
    n_all = len(ratings)
    ratings = ratings.merge(items[["movieId", "item_idx"]], on="movieId", how="inner")
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    pos = ratings.groupby("userId").cumcount()
    size = ratings.groupby("userId")["rating"].transform("size")
    ratings["split"] = np.where(pos < np.floor(TRAIN_FRAC * size), "train", "test")
    return items, ratings, n_all


class Evaluator:
    """Scores a vector of per-item scores against one split of the ratings."""

    def __init__(self, ratings, split):
        part = ratings[ratings["split"] == split]
        part = part[part.groupby("userId")["rating"].transform("size") >= MIN_ITEMS]
        part = part.sort_values("userId", kind="stable")
        self.users = part["userId"].to_numpy()
        self.item_idx = part["item_idx"].to_numpy()
        self.liked = part["rating"].to_numpy() >= LIKE_THRESHOLD
        self.actual_rank = part.groupby("userId")["rating"].rank(method="average").to_numpy()
        _, self.starts, self.sizes = np.unique(self.users, return_index=True, return_counts=True)
        self.group = np.repeat(np.arange(len(self.starts)), self.sizes)
        self.n_users, self.n_ratings = len(self.starts), len(part)
        self.like_rate = self.liked.mean()

    def evaluate(self, item_scores):
        s = np.asarray(item_scores, float)[self.item_idx]
        order = np.lexsort((-s, self.users))               # users ascending, best score first
        pos = np.empty(len(s), dtype=np.int64)             # 0 = this user's top-scored movie
        pos[order] = np.arange(len(s)) - self.starts[self.group]
        x = (self.sizes[self.group] - pos).astype(float)   # score rank, higher = better
        y = self.actual_rank
        n = self.sizes.astype(float)
        red = lambda a: np.add.reduceat(a, self.starts)
        sx, sy, sxy, sxx, syy = red(x), red(y), red(x * y), red(x * x), red(y * y)
        den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(den > 0, (n * sxy - sx * sy) / den, np.nan)
        p_at_k = red(((pos < K) & self.liked).astype(float)) / K
        return {"spearman": float(np.nanmean(rho)), "p_at_k": float(p_at_k.mean())}


def make_fitness(split="train", family="Triangular", metric="spearman"):
    """Returns f(weights) -> float. This is what the GA will maximise."""
    items, ratings, _ = load_data()
    ev = Evaluator(ratings, split)
    return lambda w: ev.evaluate(fuzzy_score(items, w, family))[metric]


if __name__ == "__main__":
    items, ratings, n_all = load_data()
    train, test = Evaluator(ratings, "train"), Evaluator(ratings, "test")
    print(f"Movies with IMDb features: {len(items)}")
    print(f"Ratings kept: {len(ratings)} of {n_all} ({len(ratings) / n_all:.1%}) after matching to IMDb")
    print(f"Train: {train.n_users} users, {train.n_ratings} ratings, share rated >= {LIKE_THRESHOLD}: {train.like_rate:.1%}")
    print(f"Test : {test.n_users} users, {test.n_ratings} ratings, share rated >= {LIKE_THRESHOLD}: {test.like_rate:.1%}")

    candidates = {
        "popularity only (log votes)": items["popularity"].to_numpy(),
        "IMDb rating, raw": items["rating_raw"].to_numpy(),
        "IMDb rating, Bayesian": items["rating"].to_numpy(),
        "fuzzy, baseline weights": fuzzy_score(items, BASELINE_WEIGHTS, "Triangular"),
        "fuzzy, all weights = 1": fuzzy_score(items, np.ones(len(RULES)), "Triangular"),
        "fuzzy, baseline, Gaussian": fuzzy_score(items, BASELINE_WEIGHTS, "Gaussian"),
        "fuzzy, baseline, Trapezoidal": fuzzy_score(items, BASELINE_WEIGHTS, "Trapezoidal"),
    }
    rows = {}
    rng = np.random.default_rng(0)
    rand = [[ev.evaluate(rng.random(len(items))) for _ in range(20)] for ev in (train, test)]
    rows["random (mean of 20)"] = [np.mean([r["spearman"] for r in rand[0]]), np.mean([r["p_at_k"] for r in rand[0]]),
                                   np.mean([r["spearman"] for r in rand[1]]), np.mean([r["p_at_k"] for r in rand[1]])]
    for name, sc in candidates.items():
        a, b = train.evaluate(sc), test.evaluate(sc)
        rows[name] = [a["spearman"], a["p_at_k"], b["spearman"], b["p_at_k"]]
    table = pd.DataFrame(rows, index=["train Spearman", f"train P@{K}", "test Spearman", f"test P@{K}"]).T
    print("\n" + table.round(4).to_string())
