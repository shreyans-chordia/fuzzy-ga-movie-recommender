"""Download IMDb + MovieLens data into ./data (skips files that already exist)."""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    ROOT / "data" / "imdb" / "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    ROOT / "data" / "imdb" / "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    ROOT / "data" / "movielens" / "ml-latest-small.zip": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        print(f"\r  {min(done, total) / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)


for dest, url in FILES.items():
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip (exists): {dest.name}")
        continue
    print(f"downloading {dest.name}")
    urllib.request.urlretrieve(url, dest, progress)
    print()

zip_path = ROOT / "data" / "movielens" / "ml-latest-small.zip"
if not (ROOT / "data" / "movielens" / "ml-latest-small").exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(zip_path.parent)
    print("extracted ml-latest-small")
print("done")
