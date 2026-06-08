from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    """Abstract base class for all vector indices.

    Provides a uniform interface for building, searching, and
    querying the memory footprint of different index types.
    Concrete subclasses implement FAISS Flat, IVF, IVF-PQ, and
    Qdrant HNSW indices.
    """

    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        """Build the index from embeddings.

        Args:
            embeddings: Float32 array of shape (n_vectors, dim).
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors.

        Args:
            query: Query vector of shape (768,) or (1, 768).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices), each of shape (1, k).
        """

    @abstractmethod
    def get_memory_mb(self) -> float:
        """Return the memory footprint of the index in megabytes."""
