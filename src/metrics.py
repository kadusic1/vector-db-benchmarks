import os

import numpy as np


class MetricsCalculator:
    """Compute recall@10, per-query medians, and memory footprint."""

    def __init__(self, n_queries: int = 1000, n_reps: int = 5) -> None:
        self.n_queries = n_queries
        self.n_reps = n_reps

    def compute_recall_at_k(
        self,
        retrieved: list[list[int]],
        ground_truth: list[list[int]],
        k: int = 10,
    ) -> tuple[float, list[float]]:
        """Compute recall@k for each query.

        Args:
            retrieved: List of retrieved ID lists per query.
            ground_truth: List of ground truth ID lists per query.
            k: Number of top results to consider.

        Returns:
            Tuple of (mean_recall, per-query recall list).
        """
        recalls: list[float] = []
        for ret, gt in zip(retrieved, ground_truth):
            n_correct = len(set(ret[:k]) & set(gt[:k]))
            recalls.append(n_correct / k)
        return float(np.mean(recalls)), recalls

    def compute_per_query_medians(self, latencies_ms: list[float]) -> np.ndarray:
        """Compute per-query median latency across repetitions.

        Input order: [q0_r0, q1_r0, ..., q999_r0, q0_r1, ...]

        Args:
            latencies_ms: Flat list of n_reps * n_queries values.

        Returns:
            Array of n_queries median latencies.
        """
        arr = np.array(latencies_ms).reshape(self.n_reps, self.n_queries)
        return np.median(arr, axis=0)

    @staticmethod
    def get_index_size_mb(
        path: str,
        n_vectors: int | None = None,
        dim: int = 768,
    ) -> float:
        """Get index file size in MB (disk or theoretical).

        Args:
            path: Path to the index file.
            n_vectors: Number of vectors (for theoretical size).
            dim: Vector dimension (default 768).

        Returns:
            Size in megabytes.
        """
        if os.path.exists(path):
            return os.path.getsize(path) / (1024**2)
        if n_vectors is not None:
            return (n_vectors * dim * 4) / (1024**2)
        return 0.0
