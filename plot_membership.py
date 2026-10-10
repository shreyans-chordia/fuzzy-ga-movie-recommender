"""Step 7: define triangular membership functions and check them against the real data."""
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
items["recency"] = items["recency"].clip(upper=AGE_CAP)   # ancient titles all count as "Old"

# feature -> (universe, {set name: triangle (a, b, c)})
FEATURES = {
    "rating":     (np.linspace(1, 10, 361),  {"Low": (1, 1, 6.4), "Medium": (5.75, 6.2, 6.7), "High": (6.0, 10, 10)}),
    "popularity": (np.linspace(2, 6.5, 361), {"Low": (2, 2, 2.6), "Medium": (2.3, 2.9, 3.6), "High": (2.8, 6.5, 6.5)}),
    "recency":    (np.linspace(0, AGE_CAP, 361), {"New": (0, 0, 12), "Mid": (6, 18, 35),     "Old": (25, 60, 60)}),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (x, sets)) in zip(axes, FEATURES.items()):
    for label, abc in sets.items():
        ax.plot(x, fuzz.trimf(x, list(abc)), label=label, linewidth=2)
    ax.set_title(name)
    ax.set_xlabel({"rating": "IMDb rating", "popularity": "log10(votes)", "recency": "age (years)"}[name])
    ax.set_ylabel("membership")
    ax.legend()
plt.tight_layout()
out = ROOT / "data/processed/membership.png"
plt.savefig(out, dpi=120)
print("saved", out)

# Balance check: which set is each title most strongly in?
print("\nShare of titles whose strongest set is:")
for name, (x, sets) in FEATURES.items():
    mem = np.column_stack([fuzz.trimf(items[name].to_numpy(dtype=float), list(abc)) for abc in sets.values()])
    best = mem.argmax(axis=1)
    shares = {label: f"{(best == i).mean():.1%}" for i, label in enumerate(sets)}
    print(f"  {name:11s}", shares)
