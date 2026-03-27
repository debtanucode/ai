import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a sentence-transformers model.
    Concept from book: embedding model converts text -> dense vector.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns float32 numpy array of shape (N, dim)."""
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns float32 array of shape (1, dim)."""
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding.astype("float32")
