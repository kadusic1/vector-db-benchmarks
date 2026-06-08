import os
import time

import numpy as np
from datasets import load_dataset

from src.data import N_SHARDS
from src.benchmark import BenchmarkRunner
from src.indices import FlatIndex, HNSWIndex, IVFIndex, IVFPQIndex
from src.logger import logger
from src.metrics import MetricsCalculator
from src.plots import PlotGenerator
from src.statistics import StatisticsAnalyzer


_DATA_FILES = [f"data/passages_{i}.parquet" for i in range(N_SHARDS)] + [
    "data/queries.parquet"
]


class ExperimentPipeline:
    """Orchestrate all phases of the vector index benchmark.

    Assumes data/ has already been prepared by
    ``scripts/prepare_data.py``.  Executes 7 phases: load data,
    build indices, network overhead, benchmark, metrics,
    statistics, parameter sweep, and plots.
    """

    def __init__(self) -> None:
        self.benchmark_runner = BenchmarkRunner()
        self.metrics = MetricsCalculator()
        self.stats = StatisticsAnalyzer()

        self.passage_emb: np.ndarray | None = None
        self.query_emb: np.ndarray | None = None
        self.passages: list[str] | None = None

        self.index_flat: FlatIndex | None = None
        self.index_ivf: IVFIndex | None = None
        self.index_ivfpq: IVFPQIndex | None = None
        self.index_hnsw: HNSWIndex | None = None
        self.qdrant_overhead_ms: float = 0.0

        self.latencies: dict[str, list[float]] = {}
        self.results: dict[str, list[list[int]]] = {}

    @staticmethod
    def _check_data_exists() -> None:
        missing = [p for p in _DATA_FILES if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Nedostaju data fajlovi.")

    def _phase_1_load_data(self) -> None:
        self._check_data_exists()
        logger.info("FAZA 1: Ucitavanje podataka sa diska")

        data_files = [f"data/passages_{i}.parquet" for i in range(N_SHARDS)]
        ds = load_dataset("parquet", data_files=data_files, split="train")
        self.passages = list(ds["text"])
        self.passage_emb = np.array(ds["embedding"], dtype=np.float32)
        logger.info(f"Ucitano {len(self.passages)} pasusa")

        ds_q = load_dataset("parquet", data_files="data/queries.parquet", split="train")
        self.query_emb = np.array(ds_q["embedding"], dtype=np.float32)
        logger.info(f"Ucitano {len(self.query_emb)} embeddinga upita")

    def _phase_2_indices(self) -> None:
        assert self.passage_emb is not None
        logger.info("FAZA 2: Izgradnja indeksa")

        logger.info("Flat: izgradnja...")
        self.index_flat = FlatIndex()
        self.index_flat.build(self.passage_emb)

        logger.info("IVF: trening i izgradnja...")
        self.index_ivf = IVFIndex()
        self.index_ivf.build(self.passage_emb)

        logger.info("IVF-PQ: trening i izgradnja...")
        self.index_ivfpq = IVFPQIndex()
        self.index_ivfpq.build(self.passage_emb)

        assert self.passages is not None
        logger.info("HNSW: izgradnja kolekcije u Qdrant-u...")
        self.index_hnsw = HNSWIndex()
        self.index_hnsw.build(self.passage_emb, passages=self.passages)

    def _phase_3_overhead(self) -> None:
        logger.info("FAZA 3: Mjerenje Qdrant mreznog overhead-a")
        assert self.index_hnsw is not None
        self.qdrant_overhead_ms = self.index_hnsw.measure_network_overhead()
        logger.info(f"Qdrant network overhead: {self.qdrant_overhead_ms:.2f} ms")

    def _phase_4_benchmark(self) -> None:
        logger.info("FAZA 4: Benchmarking")
        assert self.index_flat is not None
        assert self.index_ivf is not None
        assert self.index_ivfpq is not None
        assert self.index_hnsw is not None
        assert self.query_emb is not None

        logger.info("Flat: benchmark...")
        lat_flat, res_flat = self.benchmark_runner.run(
            self.index_flat.search, self.query_emb
        )
        self.latencies["flat"] = lat_flat
        self.results["flat"] = res_flat

        logger.info("IVF (nprobe=32): benchmark...")
        self.index_ivf.nprobe = 32
        lat_ivf, res_ivf = self.benchmark_runner.run(
            self.index_ivf.search, self.query_emb
        )
        self.latencies["ivf"] = lat_ivf
        self.results["ivf"] = res_ivf

        logger.info("IVF-PQ (nprobe=32): benchmark...")
        self.index_ivfpq.nprobe = 32
        lat_ivfpq, res_ivfpq = self.benchmark_runner.run(
            self.index_ivfpq.search, self.query_emb
        )
        self.latencies["ivfpq"] = lat_ivfpq
        self.results["ivfpq"] = res_ivfpq

        logger.info("HNSW (ef_search=128): benchmark...")
        self.index_hnsw.ef_search = 128
        lat_hnsw, res_hnsw = self.benchmark_runner.run(
            self.index_hnsw.search, self.query_emb
        )
        self.latencies["hnsw"] = lat_hnsw
        self.results["hnsw"] = res_hnsw

    def _phase_5_metrics(self) -> dict:
        logger.info("FAZA 5: Metrike")
        ground_truth = self.results["flat"]

        recall_ivf, _ = self.metrics.compute_recall_at_k(
            self.results["ivf"], ground_truth
        )
        recall_ivfpq, _ = self.metrics.compute_recall_at_k(
            self.results["ivfpq"], ground_truth
        )
        recall_hnsw, _ = self.metrics.compute_recall_at_k(
            self.results["hnsw"], ground_truth
        )

        logger.info(
            f"Recall@10 - IVF: {recall_ivf:.4f}, "
            f"IVF-PQ: {recall_ivfpq:.4f}, "
            f"HNSW: {recall_hnsw:.4f}"
        )

        medians_flat = self.metrics.compute_per_query_medians(self.latencies["flat"])
        medians_ivf = self.metrics.compute_per_query_medians(self.latencies["ivf"])
        medians_ivfpq = self.metrics.compute_per_query_medians(self.latencies["ivfpq"])
        medians_hnsw = self.metrics.compute_per_query_medians(self.latencies["hnsw"])
        medians_hnsw_corrected = medians_hnsw - self.qdrant_overhead_ms

        return {
            "recall_ivf": recall_ivf,
            "recall_ivfpq": recall_ivfpq,
            "recall_hnsw": recall_hnsw,
            "medians_flat": medians_flat,
            "medians_ivf": medians_ivf,
            "medians_ivfpq": medians_ivfpq,
            "medians_hnsw": medians_hnsw,
            "medians_hnsw_corrected": medians_hnsw_corrected,
        }

    def _phase_6_statistics(self, metrics: dict) -> None:
        logger.info("FAZA 6: Statisticka analiza")
        stats_flat = self.stats.describe("Flat", self.latencies["flat"])
        stats_ivf = self.stats.describe("IVF", self.latencies["ivf"])
        stats_ivfpq = self.stats.describe("IVF-PQ", self.latencies["ivfpq"])
        stats_hnsw = self.stats.describe("HNSW", self.latencies["hnsw"])

        header = (
            f"{'Konfiguracija':<15} {'Mean(ms)':<10} "
            f"{'Median(ms)':<12} {'Std(ms)':<10} "
            f"{'P95(ms)':<10} {'P99(ms)':<10}"
        )
        logger.info(header)
        for s in [stats_flat, stats_ivf, stats_ivfpq, stats_hnsw]:
            logger.info(
                f"{s['konfiguracija']:<15} {s['mean_ms']:<10.3f} "
                f"{s['median_ms']:<12.3f} {s['std_ms']:<10.3f} "
                f"{s['p95_ms']:<10.3f} {s['p99_ms']:<10.3f}"
            )

        logger.info("Wilcoxon signed-rank testovi:")
        h1 = self.stats.wilcoxon_test(
            metrics["medians_hnsw_corrected"],
            metrics["medians_ivf"],
            alternative="less",
        )
        logger.info(
            f"H1 (HNSW < IVF): statistic={h1['statistic']:.2f}, "
            f"p={h1['p_value']:.6f}, "
            f"significant={h1['significant']}"
        )
        metrics["_h1_wilcoxon_ok"] = h1["significant"]

        recall_hnsw = metrics["recall_hnsw"]
        h1_recall_ok = recall_hnsw > 0.90
        metrics["_h1_recall_ok"] = h1_recall_ok
        logger.info(f"H1 (recall@10 > 90%): {recall_hnsw:.4f} > 0.90 = {h1_recall_ok}")

    def _phase_7_sweep(self) -> dict:
        logger.info("FAZA 7: Sweep parametara")
        assert self.query_emb is not None
        assert self.index_ivf is not None
        assert self.index_ivfpq is not None
        assert self.index_hnsw is not None

        curves: dict[str, list[tuple[float, float]]] = {
            "IVF": [],
            "IVF-PQ": [],
            "HNSW": [],
        }

        for nprobe in [8, 16, 32, 64, 128]:
            self.index_ivf.nprobe = nprobe
            latencies, results = self.benchmark_runner.run(
                self.index_ivf.search, self.query_emb, n_reps=3
            )
            recall, _ = self.metrics.compute_recall_at_k(results, self.results["flat"])
            mean_lat = float(np.mean(latencies))
            curves["IVF"].append((recall, mean_lat))
            logger.info(
                f"IVF nprobe={nprobe}: recall={recall:.4f}, latency={mean_lat:.4f} ms"
            )

        for nprobe in [8, 16, 32, 64, 128]:
            self.index_ivfpq.nprobe = nprobe
            latencies, results = self.benchmark_runner.run(
                self.index_ivfpq.search, self.query_emb, n_reps=3
            )
            recall, _ = self.metrics.compute_recall_at_k(results, self.results["flat"])
            mean_lat = float(np.mean(latencies))
            curves["IVF-PQ"].append((recall, mean_lat))
            logger.info(
                f"IVF-PQ nprobe={nprobe}: recall={recall:.4f}, "
                f"latency={mean_lat:.4f} ms"
            )

        for ef in [50, 100, 200, 400]:
            self.index_hnsw.ef_search = ef
            latencies, results = self.benchmark_runner.run(
                self.index_hnsw.search, self.query_emb, n_reps=3
            )
            recall, _ = self.metrics.compute_recall_at_k(results, self.results["flat"])
            mean_lat = float(np.mean(latencies))
            curves["HNSW"].append((recall, mean_lat))
            logger.info(
                f"HNSW ef_search={ef}: recall={recall:.4f}, latency={mean_lat:.4f} ms"
            )

        return curves

    def _phase_8_plots(self, curves: dict, n_passages: int) -> None:
        assert self.index_hnsw is not None
        logger.info("FAZA 8: Generisanje grafika")

        PlotGenerator.recall_latency_curves(curves, "output/figures/tradeoff_curve.pdf")
        logger.info("Generisan: output/figures/tradeoff_curve.pdf")

        latencies_dict = {
            "Flat": self.latencies["flat"],
            "IVF": self.latencies["ivf"],
            "IVF-PQ": self.latencies["ivfpq"],
            "HNSW": self.latencies["hnsw"],
        }
        PlotGenerator.latency_boxplots(
            latencies_dict, "output/figures/boxplot_latency.pdf"
        )
        logger.info("Generisan: output/figures/boxplot_latency.pdf")

        flat_size = MetricsCalculator.get_index_size_mb("", n_vectors=n_passages)
        ivf_size = MetricsCalculator.get_index_size_mb(
            "output/artifacts/index_ivf.faiss", n_vectors=n_passages
        )
        ivfpq_size = MetricsCalculator.get_index_size_mb(
            "output/artifacts/index_ivfpq.faiss", n_vectors=n_passages
        )
        hnsw_size = self.index_hnsw.get_memory_mb()

        memory_mb = {
            "Flat": flat_size,
            "IVF": ivf_size,
            "IVF-PQ": ivfpq_size,
            "HNSW": hnsw_size,
        }
        PlotGenerator.memory_comparison(
            memory_mb, "output/figures/memory_comparison.pdf"
        )
        logger.info("Generisan: output/figures/memory_comparison.pdf")

    def run(self) -> None:
        """Execute all phases of the benchmark pipeline.

        Prerequisites: data/passages_0..9.parquet, data/queries.parquet
        (created by ``scripts/create_embeddings.py`` or
        ``scripts/download_embeddings.py``)
        """
        start_time = time.perf_counter()
        for d in ["output/artifacts", "output/figures"]:
            os.makedirs(d, exist_ok=True)

        self._phase_1_load_data()

        assert self.passage_emb is not None
        n_passages = len(self.passage_emb)

        self._phase_2_indices()
        self._phase_3_overhead()
        self._phase_4_benchmark()
        metrics = self._phase_5_metrics()
        self._phase_6_statistics(metrics)

        curves = self._phase_7_sweep()
        self._phase_8_plots(curves, n_passages)

        assert self.index_hnsw is not None
        hnsw_memory_mb = self.index_hnsw.get_memory_mb()
        self.benchmark_runner.verify_hypotheses(
            metrics, curves, n_passages, hnsw_memory_mb
        )

        elapsed = time.perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(
            f"Ukupno vrijeme simulacije: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        logger.info("Sve faze zavrsene. Grafici su spremljeni u output/figures/.")
