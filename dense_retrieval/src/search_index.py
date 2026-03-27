import numpy as np
import faiss


class SearchIndex:
    """
    FAISS-based nearest neighbor index.
    Concept from book: store embeddings in an index optimised for fast similarity search.
    """

    def __init__(self):
        self.index = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build a flat L2 index from a (N, dim) float32 array."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors of dimension {dim}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the nearest neighbours of the query.
        Returns (distances, indices) each of shape (1, top_k).
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices
