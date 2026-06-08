import numpy as np
import faiss

from src.core.index import VectorIndex


class FlatIndex(VectorIndex):
    """FAISS Flat (brute-force) index as baseline.

    Performs exhaustive search over all vectors. Guarantees 100%
    recall and serves as ground truth. Not saved to disk since it
    is fast to rebuild.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatL2 | None = None
        self.n_vectors: int = 0

    def build(self, embeddings: np.ndarray) -> None:
        self._index = faiss.IndexFlatL2(self.dim)
        self._index.add(embeddings.astype(np.float32))
        self.n_vectors = len(embeddings)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return self._index.search(query.astype(np.float32), k)

    def get_memory_mb(self) -> float:
        return (self.n_vectors * self.dim * 4) / (1024**2)
