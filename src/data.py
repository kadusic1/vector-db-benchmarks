from datasets import load_dataset

from src.logger import logger

N_SHARDS = 10


def load_passages(
    split: str = "train",
    n_passages: int = 500_000,
) -> list[str]:
    """Stream passage texts from MS MARCO and return unique texts.

    Args:
        split: Dataset split to load ("train" or "validation").
        n_passages: Maximum number of unique passages to return.

    Returns:
        List of passage strings.
    """
    logger.info(f"Streamanje MS MARCO ({split}) do {n_passages} unikatnih pasusa...")
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True)
    passages_set: set[str] = set()
    for example in dataset:
        for passage in example["passages"]["passage_text"]:
            passages_set.add(passage)
            if len(passages_set) >= n_passages:
                return list(passages_set)
    return list(passages_set)


def load_queries(split: str = "validation", n: int = 1000) -> list[str]:
    """Stream query texts from MS MARCO and return the first n.

    Args:
        split: Dataset split to load ("validation").
        n: Number of queries to return.

    Returns:
        List of n query strings.
    """
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True)
    queries: list[str] = []
    for i, example in enumerate(dataset):
        if i >= n:
            break
        queries.append(example["query"])
    return queries
