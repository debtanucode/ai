"""
data_loader.py
--------------
Loads the Indian Movies corpus from CSV.
The 'Description' column is used as the searchable document text.
"""
import csv
from pathlib import Path

DATA_FILE = Path(__file__).parent / "indian_movies.csv"


def load_movies():
    """
    Load movies from CSV.

    Returns
    -------
    movies : list[dict]
        Full rows — Movie Name, Description, Release Date, IMDB Rating
    texts : list[str]
        Just the descriptions (used as documents by all retrievers / rerankers)
    """
    movies = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append(dict(row))
    texts = [m["Description"] for m in movies]
    return movies, texts
