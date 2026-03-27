import pandas as pd


def load_movies(csv_path: str) -> tuple[list[str], list[dict]]:
    """
    Load movies from CSV and return:
      - texts : list of strings to embed (name + description)
      - metadata : list of dicts with full movie details
    """
    df = pd.read_csv(csv_path)

    # Support both "Name" and "Movie Name" column headers
    name_col = "Movie Name" if "Movie Name" in df.columns else "Name"

    # Combine name + description so the embedding captures both
    movie_chunks = (df[name_col] + ". " + df["Description"]).tolist()

    movie_metadata = df.to_dict(orient="records")

    return movie_chunks, movie_metadata
