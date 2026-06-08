import os

import numpy as np
import faiss

from src.core.index import VectorIndex


class IVFIndex(VectorIndex):
    """FAISS IVF (Inverted File Index) with configurable nprobe.

    Divides the vector space into nlist clusters via k-means.
    Search probes nprobe nearest clusters for a tradeoff between
    speed and accuracy.
    """

    def __init__(
        self,
        dim: int = 768,
        nlist: int = 256,
        nprobe: int = 32,
        artifact_path: str = "artifacts/index_ivf.faiss",
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.artifact_path = artifact_path
        self._index: faiss.IndexIVFFlat | None = None
        self.n_vectors: int = 0

    def build(self, embeddings: np.ndarray) -> None:
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
        faiss.write_index(index, self.artifact_path)
        self._index = index
        self.n_vectors = len(embeddings)

    def load(self, path: str | None = None) -> None:
        self._index = faiss.read_index(path or self.artifact_path)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        if query.ndim == 1:
            query = query.reshape(1, -1)
        self._index.nprobe = self.nprobe
        return self._index.search(query.astype(np.float32), k)

    def get_memory_mb(self) -> float:
        if os.path.exists(self.artifact_path):
            return os.path.getsize(self.artifact_path) / (1024**2)
        return (self.n_vectors * self.dim * 4) / (1024**2)
