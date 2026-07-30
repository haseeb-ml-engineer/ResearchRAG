"""
Sentence Transformers embedding provider for ResearchRAG.

This module implements `SentenceTransformerEmbedding`, a concrete
`BaseEmbedding` that generates document and query embeddings using the
`sentence-transformers` library. It encapsulates all interaction with
the underlying `SentenceTransformer` model, exposing only the
provider-agnostic interface defined by `BaseEmbedding` to the rest of
the system.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.base_embedding import BaseEmbedding, EmbeddingError

logger = get_logger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Embedding provider backed by a local Sentence Transformers model.

    The underlying model is loaded lazily on first use rather than at
    construction time, so instantiating this class is inexpensive and
    does not require the model weights to be loaded immediately.
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None) -> None:
        """
        Initialize the Sentence Transformers embedding provider.

        Args:
            model_name: Identifier of the Sentence Transformers model
                to load. Defaults to the value configured in
                application settings.
            batch_size: Number of texts to encode per batch. Defaults
                to the value configured in application settings.
        """
        self._model_name: str = model_name or settings.embedding.model
        self._batch_size: int = batch_size or settings.embedding.batch_size
        self._model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """
        Load the Sentence Transformers model if it has not already
        been loaded.

        This method is safe to call multiple times: subsequent calls
        are no-ops once the model has been successfully loaded.

        Raises:
            EmbeddingError: If the model fails to load, for example
                due to an invalid or unavailable model identifier.
        """
        if self._model is not None:
            return

        logger.info("Loading Sentence Transformers model: %s", self._model_name)
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as error:
            raise EmbeddingError(
                f"Failed to load Sentence Transformers model '{self._model_name}': {error}"
            ) from error

        logger.info("Successfully loaded Sentence Transformers model: %s", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of document texts.

        Args:
            texts: List of text strings to embed, typically the
                content of document chunks.

        Returns:
            A list of embedding vectors, one per input text, in the
            same order as `texts`.

        Raises:
            EmbeddingError: If `texts` is empty, contains non-string
                elements, or if embedding generation fails.
        """
        self._validate_texts(texts)
        self.load_model()
        assert self._model is not None

        logger.info("Generating embeddings for %d document(s)", len(texts))
        vectors = self._encode(texts)
        logger.info("Generated document embeddings with shape %s", vectors.shape)

        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text: The query text to embed.

        Returns:
            An embedding vector as a list of floats.

        Raises:
            EmbeddingError: If `text` is empty, is not a string, or if
                embedding generation fails.
        """
        self._validate_texts([text])
        self.load_model()
        assert self._model is not None

        logger.info("Generating embedding for query")
        vector = self._encode([text])[0]
        logger.info("Generated query embedding with shape %s", vector.shape)

        return [float(value) for value in vector]

    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this model.

        Returns:
            The number of dimensions in each embedding vector.

        Raises:
            EmbeddingError: If the model fails to load while
                determining its embedding dimension.
        """
        self.load_model()
        assert self._model is not None
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise EmbeddingError("Sentence Transformer model did not report an embedding dimension.")
        return int(dimension)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors using the loaded
        model.

        Args:
            texts: List of text strings to encode.

        Returns:
            A NumPy array of shape (len(texts), embedding_dimension).

        Raises:
            EmbeddingError: If encoding fails.
        """
        try:
            assert self._model is not None
            return self._model.encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError(f"Failed to generate embeddings: {error}") from error

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        """
        Validate that input texts are non-empty and correctly typed.

        Args:
            texts: List of text strings to validate.

        Raises:
            EmbeddingError: If `texts` is empty, is not a list, or
                contains any non-string or empty elements.
        """
        if not texts:
            raise EmbeddingError("Input texts must not be empty.")

        if not isinstance(texts, list):
            raise EmbeddingError(f"Expected a list of strings, got {type(texts)}.")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"Invalid text input for embedding: {text!r}")