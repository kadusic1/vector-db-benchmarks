import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.logger import logger


class EmbeddingEngine:
    """Encode texts into vectors using sentence-transformers.

    Uses all-mpnet-base-v2 to produce 768-dimensional float32
    embeddings with optional L2 normalization. Automatically
    detects CUDA for GPU acceleration if available.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"EmbeddingEngine: device={self.device}, batch_size={batch_size}")
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into a float32 numpy array.

        Args:
            texts: List of text strings to encode.
            normalize: Whether to L2-normalize the embeddings.

        Returns:
            Float32 array of shape (len(texts), 768).
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def verify_normalization(embeddings: np.ndarray) -> bool:
        """Verify that all embeddings are L2-normalized.

        Args:
            embeddings: Float32 array of shape (n_vectors, dim).

        Returns:
            True if all norms are within [0.9999, 1.0001].

        Raises:
            AssertionError: If any norm is outside valid range.
        """
        norms = np.linalg.norm(embeddings, axis=1)
        min_norm = norms.min()
        max_norm = norms.max()
        logger.info(f"Normalization check: min={min_norm:.6f}, max={max_norm:.6f}")
        assert np.all((norms >= 0.9999) & (norms <= 1.0001)), (
            f"Embeddings not normalized: min={min_norm:.6f}, max={max_norm:.6f}"
        )
        return True
