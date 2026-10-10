from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# IMDb marks missing values as \N, and titles can contain quote characters,
# so we tell pandas both (na_values, and quoting=3 which means "don't treat quotes specially")
basics = pd.read_csv(ROOT / "data/imdb/title.basics.tsv.gz", sep="\t",
                     na_values="\\N", quoting=3, low_memory=False)
ratings = pd.read_csv(ROOT / "data/imdb/title.ratings.tsv.gz", sep="\t", na_values="\\N")
links = pd.read_csv(ROOT / "data/movielens/ml-latest-small/links.csv")

print(basics.titleType.value_counts().head(10))
print("Latest startYear in IMDb:", basics.startYear.max())
print(ratings.describe())
print(links.head())
