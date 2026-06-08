"""Stream MS MARCO, compute embeddings, and save as local Parquet
files for the pipeline.

This script takes ~1 hour (500k passages encoded with
all-mpnet-base-v2). It serves as proof that the embeddings were
computed from the original dataset rather than downloaded from a
pre-built source.

Usage:
    python scripts/create_embeddings.py

Generates:
    data/passages_0..9.parquet  - passages text + 768-dim embeddings
                                  (50k rows per shard)
    data/queries.parquet        - queries text + 768-dim embeddings
"""

import os
import sys

from datasets import Dataset, Features, Sequence, Value

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import N_SHARDS, load_passages, load_queries
from src.embeddings import EmbeddingEngine
from src.logger import logger


def _save_parquet(
    texts: list[str], embeddings, path: str, name: str, id_offset: int = 0
) -> None:
    dim = embeddings.shape[1]
    features = Features(
        {
            "id": Value("int32"),
            "text": Value("string"),
            "embedding": Sequence(Value("float32"), length=dim),
        }
    )
    ds = Dataset.from_dict(
        {
            "id": list(range(id_offset, id_offset + len(texts))),
            "text": texts,
            "embedding": embeddings.tolist(),
        },
        features=features,
    )
    ds.to_parquet(path)
    mb = os.path.getsize(path) / 1024**2
    logger.info(f"  {name}: {len(texts)} redova, {mb:.1f} MB -> {path}")


def main() -> None:
    os.makedirs("data", exist_ok=True)

    passage_shards = [f"data/passages_{i}.parquet" for i in range(N_SHARDS)]
    if all(os.path.exists(p) for p in passage_shards) and os.path.exists(
        "data/queries.parquet"
    ):
        logger.info("Parquet fajlovi vec postoje. Skripta zavrsava.")
        return

    logger.info("Pokretanje pripreme podataka iz MS MARCO...")

    passages = load_passages()
    logger.info(f"Ucitano {len(passages)} pasusa")

    queries = load_queries(n=1000)
    logger.info(f"Ucitano {len(queries)} upita")

    engine = EmbeddingEngine()

    logger.info("Enkodiranje pasusa (ovo traje ~1h)...")
    passage_emb = engine.encode(passages)
    logger.info(f"Enkodirano {len(passage_emb)} pasusa")

    logger.info("Enkodiranje upita...")
    query_emb = engine.encode(queries)
    logger.info(f"Enkodirano {len(query_emb)} upita")

    EmbeddingEngine.verify_normalization(passage_emb)
    EmbeddingEngine.verify_normalization(query_emb)

    n = len(passages)
    shard_size = n // N_SHARDS
    for i in range(N_SHARDS):
        start = i * shard_size
        end = start + shard_size if i < N_SHARDS - 1 else n
        _save_parquet(
            passages[start:end],
            passage_emb[start:end],
            f"data/passages_{i}.parquet",
            f"passages_{i}",
            id_offset=start,
        )
    _save_parquet(queries, query_emb, "data/queries.parquet", "queries")

    logger.info(
        f"Gotovo. Spremljeno: {', '.join(passage_shards)}, data/queries.parquet"
    )


if __name__ == "__main__":
    main()
