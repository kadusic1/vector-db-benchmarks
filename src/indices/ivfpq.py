import os

import numpy as np
import faiss

from src.core.index import VectorIndex


class IVFPQIndex(VectorIndex):
    """FAISS IVF-PQ index with Product Quantization compression.

    Combines IVF clustering with PQ compression to dramatically
    reduce memory footprint at the cost of approximate distance
    computation.
    """

    def __init__(
        self,
        dim: int = 768,
        nlist: int = 256,
        M: int = 64,
        nbits: int = 8,
        nprobe: int = 32,
        artifact_path: str = "artifacts/index_ivfpq.faiss",
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.artifact_path = artifact_path
        self._index: faiss.IndexIVFPQ | None = None
        self.n_vectors: int = 0

    def build(self, embeddings: np.ndarray) -> None:
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.M, self.nbits)
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
