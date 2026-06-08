import time
from collections.abc import Callable

import numpy as np


class BenchmarkRunner:
    """Benchmark protocol: 1000 queries x 5 repetitions.

    Measures latency for each query across n_reps repetitions.
    The outer loop is repetitions, inner loop is queries.
    Results (indices) are saved only from the first repetition.
    No warmup is performed.
    """

    def run(
        self,
        search_fn: Callable,
        query_embeddings: np.ndarray,
        k: int = 10,
        n_reps: int = 5,
    ) -> tuple[list[float], list[list[int]]]:
        """Benchmark a search function.

        The search_fn must accept (query_2d, k) where query_2d
        has shape (1, 768) and return (distances, indices), both
        of shape (1, k). This matches the VectorIndex.search
        interface.

        Args:
            search_fn: Callable taking (ndarray, int) and
                returning (distances, indices).
            query_embeddings: Float32 array (n_queries, 768).
            k: Number of nearest neighbors to retrieve.
            n_reps: Number of repetitions per query.

        Returns:
            Tuple of (latencies_ms, results_rep1).
            latencies_ms is a list of n_reps * n_queries floats.
            results_rep1 is a list of n_queries lists of int IDs.
        """
        latencies: list[float] = []
        results_rep1: list[list[int]] | None = None

        for rep in range(n_reps):
            rep_results: list[list[int]] = []
            for query in query_embeddings:
                query_2d = query.reshape(1, -1)
                start = time.perf_counter()
                _, indices = search_fn(query_2d, k)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
                rep_results.append(indices[0].tolist())
            if rep == 0:
                results_rep1 = rep_results

        assert results_rep1 is not None
        return latencies, results_rep1
