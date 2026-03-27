from src.data_loader import load_movies
from src.embedder import Embedder
from src.search_index import SearchIndex


class MovieSearchEngine:
    """
    Ties together data loading, embedding, and index search.

    Book flow:
      1. Load & chunk text  → data_loader
      2. Embed chunks       → embedder
      3. Build index        → search_index
      4. Query              → embed query → nearest neighbour → return results
    """

    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.movie_metadata = []
        self.embedder = Embedder(model_name)
        self.index = SearchIndex()
        self._build(csv_path)

    def _build(self, csv_path: str) -> None:
        print("\nLoading movie data ...")
        movie_chunks, self.movie_metadata = load_movies(csv_path)

        print(f"Embedding {len(movie_chunks)} movie descriptions ...")
        chunk_embeddings = self.embedder.embed(movie_chunks)

        print("Building FAISS search index ...")
        self.index.build(chunk_embeddings)
        print("Search engine ready.\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and return the top_k most similar movies.
        Each result includes the movie metadata plus the similarity distance.
        """
        query_embedding = self.embedder.embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)

        search_results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            movie = self.movie_metadata[idx].copy()
            movie["rank"] = rank
            movie["distance"] = round(float(dist), 4)
            search_results.append(movie)

        return search_results
