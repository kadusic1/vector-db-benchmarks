import numpy as np
from scipy.stats import wilcoxon


class StatisticsAnalyzer:
    """Descriptive statistics and Wilcoxon signed-rank test."""

    @staticmethod
    def describe(name: str, latencies_ms: list[float]) -> dict:
        """Compute descriptive statistics for latency measurements.

        Args:
            name: Configuration name (e.g., "Flat", "IVF").
            latencies_ms: List of latency values in milliseconds.

        Returns:
            Dictionary with descriptive statistics keys.
        """
        arr = np.array(latencies_ms)
        return {
            "konfiguracija": name,
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    @staticmethod
    def wilcoxon_test(
        medians_a: np.ndarray,
        medians_b: np.ndarray,
        alternative: str = "less",
    ) -> dict:
        """Perform the Wilcoxon signed-rank test.

        Tests whether medians_a is significantly less than
        medians_b.

        Args:
            medians_a: Per-query medians for configuration A.
            medians_b: Per-query medians for configuration B.
            alternative: Hypothesis ('less', 'greater',
                'two-sided').

        Returns:
            Dict with statistic, p_value, and significant flag.
        """
        result = wilcoxon(medians_a, medians_b, alternative=alternative)
        p_value = float(result.pvalue)
        return {
            "statistic": float(result.statistic),
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
