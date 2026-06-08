import matplotlib.pyplot as plt


class PlotGenerator:
    """Generate all figures: tradeoff curves, box plots, bar
    charts."""

    @staticmethod
    def recall_latency_curves(curves: dict, save_path: str) -> None:
        """Plot recall-latency tradeoff curves.

        Args:
            curves: Dict mapping index name to list of
                (recall, latency) tuples.
            save_path: Path to save the PNG figure.
        """
        plt.figure(figsize=(10, 6))
        for name, points in curves.items():
            recalls = [p[0] for p in points]
            latencies = [p[1] for p in points]
            plt.plot(recalls, latencies, marker="o", label=name)

        plt.xlabel("Recall@10")
        plt.ylabel("Mean Latency (ms)")
        plt.title("Recall-Latency Tradeoff")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def latency_boxplots(latencies: dict, save_path: str) -> None:
        """Plot box plots of latency distributions (log scale).

        Args:
            latencies: Dict mapping config name to latency list.
            save_path: Path to save the PNG figure.
        """
        plt.figure(figsize=(10, 6))
        labels = list(latencies.keys())
        data = [latencies[lab] for lab in labels]

        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.yscale("log")
        plt.ylabel("Latency (ms) — log scale")
        plt.title("Latency Distribution by Index Configuration")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def memory_comparison(memory_mb: dict, save_path: str) -> None:
        """Plot a bar chart comparing memory footprints.

        Args:
            memory_mb: Dict mapping config name to memory in MB.
            save_path: Path to save the PNG figure.
        """
        plt.figure(figsize=(8, 5))
        labels = list(memory_mb.keys())
        values = [memory_mb[lab] for lab in labels]

        plt.bar(labels, values, color=["blue", "orange", "green", "red"])
        plt.ylabel("Memory (MB)")
        plt.title("Memory Footprint by Index Configuration")
        for i, v in enumerate(values):
            plt.text(i, v + max(values) * 0.01, f"{v:.1f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
