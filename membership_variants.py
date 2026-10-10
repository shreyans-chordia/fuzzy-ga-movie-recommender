"""Step 7c: compare three membership-function families on the same fuzzy sets.

Every set starts as a triangle (a, b, c): feet at a and c, peak at b.
The other two families are derived from the same numbers, so the comparison is fair:
  - Gaussian : mean = b, sigma = max(b - a, c - b) / 2   (value ~0.135 at the old feet)
  - Trapezoid: same feet a and c, but a flat top of full membership
               from b - 0.25*(b - a)  to  b + 0.25*(c - b)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz

ROOT = Path(__file__).resolve().parent
items = pd.read_csv(ROOT / "data/processed/items.csv.gz")

AGE_CAP = 60
items["recency"] = items["recency"].clip(upper=AGE_CAP)

# feature -> (universe, {set name: triangle (a, b, c)})  -- same numbers as plot_membership.py
FEATURES = {
    "rating":     (np.linspace(1, 10, 361),      {"Low": (1, 1, 6.4), "Medium": (5.75, 6.2, 6.7), "High": (6.0, 10, 10)}),
    "popularity": (np.linspace(2, 6.5, 361),     {"Low": (2, 2, 2.6), "Medium": (2.3, 2.9, 3.6), "High": (2.8, 6.5, 6.5)}),
    "recency":    (np.linspace(0, AGE_CAP, 361), {"New": (0, 0, 12), "Mid": (6, 18, 35),     "Old": (25, 60, 60)}),
}
XLABEL = {"rating": "IMDb rating", "popularity": "log10(votes)", "recency": "age (years)"}


def triangular(x, abc):
    return fuzz.trimf(x, list(abc))


def gaussian(x, abc):
    a, b, c = abc
    sigma = max(b - a, c - b) / 2
    return fuzz.gaussmf(x, b, sigma)


def trapezoidal(x, abc):
    a, b, c = abc
    top_left = b - 0.25 * (b - a)
    top_right = b + 0.25 * (c - b)
    return fuzz.trapmf(x, [a, top_left, top_right, c])


FAMILIES = {"Triangular": triangular, "Gaussian": gaussian, "Trapezoidal": trapezoidal}

# 3 rows (families) x 3 columns (features)
fig, axes = plt.subplots(len(FAMILIES), len(FEATURES), figsize=(15, 10), sharey=True)
for row, (fam_name, fam) in enumerate(FAMILIES.items()):
    for col, (feat, (x, sets)) in enumerate(FEATURES.items()):
        ax = axes[row, col]
        for label, abc in sets.items():
            ax.plot(x, fam(x, abc), label=label, linewidth=2)
        ax.set_title(f"{fam_name} - {feat}")
        ax.set_xlabel(XLABEL[feat])
        if col == 0:
            ax.set_ylabel("membership")
        ax.legend(loc="upper center", ncol=3, fontsize=8)
plt.tight_layout()
out = ROOT / "data/processed/membership_variants.png"
plt.savefig(out, dpi=120)
print("saved", out)

# Balance check: share of titles whose strongest set is each one, per family
for fam_name, fam in FAMILIES.items():
    print(f"\n{fam_name}: share of titles whose strongest set is")
    for feat, (x, sets) in FEATURES.items():
        vals = items[feat].to_numpy(dtype=float)
        mem = np.column_stack([fam(vals, abc) for abc in sets.values()])
        best = mem.argmax(axis=1)
        shares = {label: f"{(best == i).mean():.1%}" for i, label in enumerate(sets)}
        print(f"  {feat:11s}", shares)
