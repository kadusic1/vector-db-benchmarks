import os
import statistics
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from src.core.index import VectorIndex
from src.logger import logger


class HNSWIndex(VectorIndex):
    """Qdrant HNSW index accessed over the network.

    Builds a multi-layer graph structure inside the Qdrant vector
    database running as a Docker microservice. Network overhead is
    measured separately and can be subtracted for fair comparison
    with FAISS indices.
    """

    def __init__(
        self,
        collection_name: str = "ms_marco_hnsw",
        m: int = 16,
        ef_construct: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.collection_name = collection_name
        self.m = m
        self.ef_construct = ef_construct
        self.ef_search = ef_search
        self.client: QdrantClient | None = None

    def get_client(self) -> QdrantClient:
        """Create and return a QdrantClient connected to localhost."""
        if self.client is None:
            self.client = QdrantClient("localhost", port=6333)
        return self.client

    def build(self, embeddings: np.ndarray, passages: list[str] | None = None) -> None:
        if passages is None:
            raise ValueError("passages is required for HNSWIndex.build()")
        client = self.get_client()
        client.delete_collection(self.collection_name, timeout=60)
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embeddings.shape[1], distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(m=self.m, ef_construct=self.ef_construct),
        )
        batch_size = 500
        n_vectors = len(embeddings)
        for start in range(0, n_vectors, batch_size):
            end = min(start + batch_size, n_vectors)
            batch_embeds = embeddings[start:end]
            batch_texts = passages[start:end]
            points = [
                PointStruct(
                    id=start + i,
                    vector=batch_embeds[i].tolist(),
                    payload={"text": batch_texts[i]},
                )
                for i in range(len(batch_embeds))
            ]
            client.upsert(collection_name=self.collection_name, points=points)
            if end % 50_000 == 0 or end == n_vectors:
                logger.info(f"Uploaded {end}/{n_vectors} vectors")

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        client = self.get_client()
        if query.ndim == 2:
            query = query.reshape(-1)
        query_list = query.tolist()
        result = client.query_points(
            collection_name=self.collection_name,
            query=query_list,
            limit=k,
            search_params=SearchParams(hnsw_ef=self.ef_search),
        )
        indices = np.array([[p.id for p in result.points]], dtype=np.int64)
        distances = np.array([[p.score for p in result.points]], dtype=np.float32)
        return distances, indices

    def measure_network_overhead(self, n_measurements: int = 200) -> float:
        """Measure Qdrant network overhead latency.

        Sends lightweight get_collections requests and returns
        the median round-trip time in milliseconds.

        Args:
            n_measurements: Number of measurements to take.

        Returns:
            Median network overhead in milliseconds.
        """
        client = self.get_client()
        latencies: list[float] = []
        for _ in range(n_measurements):
            start = time.perf_counter()
            client.get_collections()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        return statistics.median(latencies)

    def get_memory_mb(self) -> float:
        qdrant_storage = "qdrant_data"
        if os.path.exists(qdrant_storage):
            total_bytes = 0
            for dirpath, _, filenames in os.walk(qdrant_storage):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_bytes += os.path.getsize(fp)
            return total_bytes / (1024**2)
        return 0.0
