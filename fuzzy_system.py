"""Step 8: the fuzzy system (Mamdani-style) with a weighted rule base.

score = fuzzy_score(table, weights, family)   -> one number in [0, 1] per title

The 10 rule weights are exactly what the GA will evolve in the next module.
We use skfuzzy for the membership functions but do the rule evaluation and
centroid defuzzification ourselves with numpy, because the GA will call this
thousands of times and skfuzzy.control scores one item at a time (far too slow).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import skfuzzy as fuzz

AGE_CAP = 60

# ---- input sets: triangles (a, b, c); other families are derived from them ----
INPUT_SETS = {
    "rating":     {"Low": (1, 1, 6.4), "Medium": (5.75, 6.2, 6.7), "High": (6.0, 10, 10)},
    "popularity": {"Low": (2, 2, 2.6), "Medium": (2.3, 2.9, 3.6),  "High": (2.8, 6.5, 6.5)},
    "recency":    {"New": (0, 0, 12),  "Mid": (6, 18, 35),         "Old": (25, 60, 60)},
}
# ---- output set: how much we recommend (score universe 0..1) ----
OUTPUT_SETS = {"Low": (0, 0, 0.5), "Medium": (0, 0.5, 1), "High": (0.5, 1, 1)}
Y = np.linspace(0, 1, 101)


def triangular(x, abc):
    return fuzz.trimf(x, list(abc))


def gaussian(x, abc):
    a, b, c = abc
    return fuzz.gaussmf(x, b, max(b - a, c - b) / 2)


def trapezoidal(x, abc):
    a, b, c = abc
    return fuzz.trapmf(x, [a, b - 0.25 * (b - a), b + 0.25 * (c - b), c])


FAMILIES = {"Triangular": triangular, "Gaussian": gaussian, "Trapezoidal": trapezoidal}

# ---- rule base: (list of (feature, set) joined by AND, output set) ----
RULES = [
    ([("rating", "High"), ("popularity", "High")], "High"),     # r1 acclaimed and widely seen
    ([("rating", "High"), ("recency", "New")], "High"),         # r2 acclaimed and recent
    ([("rating", "High"), ("popularity", "Low")], "Medium"),    # r3 hidden gem
    ([("rating", "Medium"), ("popularity", "High")], "Medium"), # r4 crowd favourite, so-so rating
    ([("rating", "Medium"), ("recency", "New")], "Medium"),     # r5 average but fresh
    ([("rating", "Low")], "Low"),                               # r6 poorly rated
    ([("popularity", "Low"), ("recency", "Old")], "Low"),       # r7 obscure and old
    ([("rating", "Medium"), ("popularity", "Low")], "Low"),     # r8 average and unseen
    ([("rating", "High"), ("popularity", "Medium")], "High"),   # r9 acclaimed, moderately seen
    ([("rating", "Medium"), ("popularity", "Medium")], "Medium"),  # r10 average, moderately seen
]
# Hand-tuned starting point: how much each rule is trusted (0 = ignore, 1 = full).
# This is the "fixed baseline" the GA must beat.
BASELINE_WEIGHTS = np.array([1.0, 0.8, 0.5, 0.6, 0.5, 1.0, 0.7, 0.6, 0.8, 0.6])


def fuzzify(table, family="Triangular"):
    """Membership degree of every title in every input set."""
    fam = FAMILIES[family]
    x = {
        "rating": table["rating"].to_numpy(float),
        "popularity": table["popularity"].to_numpy(float),
        "recency": table["recency"].clip(upper=AGE_CAP).to_numpy(float),
    }
    return {f: {s: fam(x[f], abc) for s, abc in sets.items()} for f, sets in INPUT_SETS.items()}


def fuzzy_score(table, weights=BASELINE_WEIGHTS, family="Triangular", return_fired=False):
    weights = np.asarray(weights, float)
    assert len(weights) == len(RULES)
    fam = FAMILIES[family]
    mem = fuzzify(table, family)
    n = len(table)

    # 1. firing strength of each rule (AND = min), scaled by its weight;
    #    rules with the same output set are combined with max (OR)
    agg = {s: np.zeros(n) for s in OUTPUT_SETS}
    for w, (conds, out) in zip(weights, RULES):
        strength = np.minimum.reduce([mem[f][s] for f, s in conds])
        agg[out] = np.maximum(agg[out], w * strength)

    # 2. clip each output set at its strength, merge with max, take the centroid
    out_mf = {s: fam(Y, abc) for s, abc in OUTPUT_SETS.items()}
    score = np.empty(n)
    fired = np.empty(n, bool)
    for lo in range(0, n, 20000):                     # chunked to keep memory small
        hi = min(lo + 20000, n)
        mu = np.maximum.reduce([np.minimum(agg[s][lo:hi, None], out_mf[s][None, :]) for s in OUTPUT_SETS])
        den = mu.sum(axis=1)
        fired[lo:hi] = den > 1e-9
        score[lo:hi] = np.where(fired[lo:hi], (mu * Y).sum(axis=1) / np.where(fired[lo:hi], den, 1), 0.5)
    return (score, fired) if return_fired else score


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    items = pd.read_csv(ROOT / "data/processed/items.csv.gz")

    # 1. Sanity check on hand-made titles: do the scores point the right way?
    demo = pd.DataFrame({
        "name": ["acclaimed, popular, new", "hidden gem", "average, mid-age", "bad, old, obscure"],
        "rating": [9.0, 8.5, 6.3, 3.5],
        "popularity": [5.5, 2.2, 3.0, 2.1],
        "recency": [1, 10, 15, 50],
    })
    for fam in FAMILIES:
        demo[fam] = fuzzy_score(demo, family=fam).round(3)
    print("Sanity check (higher = more recommended):")
    print(demo.to_string(index=False))

    # 2. Whole catalogue
    print("\nWhole catalogue:")
    for fam in FAMILIES:
        s, fired = fuzzy_score(items, family=fam, return_fired=True)
        print(f"  {fam:11s} no rule fired: {(~fired).sum():6d} | score min {s.min():.3f} "
              f"median {np.median(s):.3f} max {s.max():.3f}")

    # 3. Do the weights matter? (if not, the GA would have nothing to optimise)
    base = fuzzy_score(items)
    flat = fuzzy_score(items, np.ones(len(RULES)))
    print("\nSpearman correlation, baseline weights vs all-ones weights:",
          round(pd.Series(base).corr(pd.Series(flat), method="spearman"), 4))

    # 4. Top picks among recent titles
    items["score"] = base
    DEMO_MIN_VOTES = 1000     # tiny-sample ratings are unreliable for a top-N list
    recent = items[(items.startYear >= 2024) & (items.numVotes >= DEMO_MIN_VOTES)]
    print(f"\nRecent titles with >= {DEMO_MIN_VOTES} votes: {len(recent)}")
    recent = recent.sort_values("score", ascending=False)
    cols = ["primaryTitle", "titleType", "startYear", "averageRating", "rating", "numVotes", "score"]
    print("Top 10 titles from 2024 onwards (triangular, baseline weights):")
    print(recent[cols].head(10).round(3).to_string(index=False))
