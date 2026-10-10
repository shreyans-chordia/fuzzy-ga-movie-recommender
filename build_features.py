"""Module 1: build the per-title feature table (rating, popularity, recency) from IMDb,
and link it to MovieLens through imdbId."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CURRENT_YEAR = 2026
MIN_VOTES = 100          # titles with fewer votes have unreliable averages
KEEP_TYPES = {"movie", "tvSeries", "tvMiniSeries"}

# 1. Load only the columns we need (saves memory on the 228 MB file)
# Read in chunks and keep only the title types we need, so the ~12M rows (mostly
# TV episodes) never sit in memory at once
chunks = []
for chunk in pd.read_csv(ROOT / "data/imdb/title.basics.tsv.gz", sep="\t", na_values="\\N",
                         quoting=3, chunksize=1_000_000,
                         usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres", "isAdult"]):
    chunks.append(chunk[chunk.titleType.isin(KEEP_TYPES)])
basics = pd.concat(chunks, ignore_index=True)
del chunks
ratings = pd.read_csv(ROOT / "data/imdb/title.ratings.tsv.gz", sep="\t", na_values="\\N")

# 2. Keep movies and series, non-adult, with a usable year
basics = basics[basics.titleType.isin(KEEP_TYPES)].copy()
basics["isAdult"] = pd.to_numeric(basics["isAdult"], errors="coerce").fillna(0)
basics = basics[basics.isAdult == 0]
basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
basics = basics.dropna(subset=["startYear"])
n_future = (basics.startYear > CURRENT_YEAR).sum()
basics = basics[basics.startYear <= CURRENT_YEAR]
print(f"Dropped {n_future} titles with a future startYear")

# 3. Attach ratings (inner join: titles without any rating are useless to us)
items = basics.merge(ratings, on="tconst", how="inner")
items = items[items.numVotes >= MIN_VOTES].copy()

# 4. The three fuzzy inputs
BAYES_M = 1000   # how strongly small-sample ratings are pulled toward the average
C = items["averageRating"].mean()
v = items["numVotes"]
items["rating_raw"] = items["averageRating"]
items["rating"] = v / (v + BAYES_M) * items["averageRating"] + BAYES_M / (v + BAYES_M) * C   # 1-10
print(f"Bayesian rating: C = {C:.3f}, m = {BAYES_M}")
items["popularity"] = np.log10(items["numVotes"])         # log scale: votes span 100 .. 3,000,000
items["recency"] = CURRENT_YEAR - items["startYear"]      # age in years (0 = this year)

print("\nTitles kept:", len(items))
print(items.titleType.value_counts())
print(items[["rating", "popularity", "recency"]].describe())

# 5. How many recent titles survive? (this is the "relevant today" check)
for y in (2024, 2025, 2026):
    sub = items[items.startYear >= y]
    print(f"startYear >= {y}: {len(sub)} titles ({(sub.titleType != 'movie').sum()} series)")

# 6. Link to MovieLens: imdbId 114709 -> tt0114709
links = pd.read_csv(ROOT / "data/movielens/ml-latest-small/links.csv")
links["tconst"] = "tt" + links["imdbId"].astype(str).str.zfill(7)
linked = links.merge(items, on="tconst", how="left", indicator=True)
matched = (linked["_merge"] == "both").sum()
print(f"\nMovieLens movies matched to IMDb items: {matched} / {len(links)} ({matched / len(links):.1%})")

# 7. Save for the next modules
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)
items.to_csv(out / "items.csv.gz", index=False)
linked[linked["_merge"] == "both"].drop(columns="_merge").to_csv(out / "ml_items.csv.gz", index=False)
print("saved data/processed/items.csv.gz and ml_items.csv.gz")
