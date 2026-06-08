import time
from collections.abc import Callable

import numpy as np

from src.logger import logger
from src.metrics import MetricsCalculator


class BenchmarkRunner:
    """Benchmark a search function and verify research hypotheses.

    Measures latency across n_reps repetitions (outer loop) over
    all queries (inner loop).  Results from the first repetition
    are used for accuracy; all repetitions are used for latency
    analysis.  No warmup is performed.
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

    @staticmethod
    def _is_better_tradeoff(
        curve_a: list[tuple[float, float]],
        curve_b: list[tuple[float, float]],
    ) -> bool:
        a = sorted(curve_a, key=lambda x: x[0])
        b = sorted(curve_b, key=lambda x: x[0])

        a_low_recall, a_low_lat = a[0]
        a_high_recall, a_high_lat = a[-1]
        b_low_recall, b_low_lat = b[0]
        b_high_recall, b_high_lat = b[-1]

        low_end_ok = a_low_recall >= b_low_recall and a_low_lat <= b_low_lat
        high_end_ok = a_high_recall >= b_high_recall and a_high_lat <= b_high_lat

        return low_end_ok and high_end_ok

    def verify_hypotheses(
        self,
        metrics: dict,
        curves: dict,
        n_passages: int,
        hnsw_memory_mb: float,
    ) -> None:
        """Verify all three research hypotheses (H1, H2, H3).

        H1: HNSW < IVF latency + recall@10 > 90% (Wilcoxon in
            pipeline; recall check here).
        H2: IVF-PQ memory >= 4x smaller than Flat + recall drop
            < 5pp.
        H3: IVF-PQ fits in 2GB and has better recall-latency
            tradeoff than HNSW.

        Args:
            metrics: Dict with recall values and medians.
            curves: Dict with sweep data for each index.
            n_passages: Number of passages in the corpus.
            hnsw_memory_mb: HNSW memory footprint in MB.
        """
        logger.info("Verifikacija hipoteza")

        recall_ivfpq = metrics["recall_ivfpq"]

        flat_size = MetricsCalculator.get_index_size_mb("", n_vectors=n_passages)
        ivfpq_size = MetricsCalculator.get_index_size_mb(
            "output/artifacts/index_ivfpq.faiss", n_vectors=n_passages
        )

        memory_ratio = flat_size / ivfpq_size if ivfpq_size > 0 else 0.0
        recall_drop = 1.0 - recall_ivfpq

        h2_mem_ok = memory_ratio >= 4.0
        h2_recall_ok = recall_drop < 0.05

        logger.info(
            f"H2 (memorija >=4x): flat={flat_size:.1f}MB / "
            f"ivfpq={ivfpq_size:.1f}MB = {memory_ratio:.2f}x >= 4x = "
            f"{h2_mem_ok}"
        )
        logger.info(
            f"H2 (recall drop < 5pp): drop={recall_drop:.4f} < 0.05 = {h2_recall_ok}"
        )

        h3_budget_ok = ivfpq_size < 2048
        logger.info(
            f"H3 (budzet < 2GB): IVFPQ={ivfpq_size:.1f}MB < 2048MB = {h3_budget_ok}"
        )

        if hnsw_memory_mb < 2048:
            ivfpq_curve = curves["IVF-PQ"]
            hnsw_curve = curves["HNSW"]
            h3_tradeoff_ok = self._is_better_tradeoff(ivfpq_curve, hnsw_curve)
            logger.info(f"H3 (IVF-PQ bolji tradeoff od HNSW): {h3_tradeoff_ok}")
        else:
            logger.info(
                f"H3 (IVF-PQ bolji tradeoff od HNSW): True "
                f"(HNSW={hnsw_memory_mb:.1f}MB ne staje u "
                f"budzet)"
            )
            h3_tradeoff_ok = True

        all_ok = (
            metrics.get("_h1_wilcoxon_ok", False)
            and metrics.get("_h1_recall_ok", False)
            and h2_mem_ok
            and h2_recall_ok
            and h3_budget_ok
            and h3_tradeoff_ok
        )
        logger.info(f"Sve hipoteze potvrdjene: {all_ok}")
