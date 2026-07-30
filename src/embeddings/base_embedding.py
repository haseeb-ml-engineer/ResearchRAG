"""
Abstract base embedding interface for ResearchRAG.

This module defines the contract that every embedding provider in the
project must implement, regardless of the underlying model or vendor
(Sentence Transformers, OpenAI, Cohere, Voyage AI, Jina, HuggingFace,
or any future provider). Concrete embedding classes inherit from
`BaseEmbedding` and implement its abstract methods, allowing the rest
of the system to generate embeddings without any knowledge of which
specific provider is in use.

This module defines the interface only. No embedding generation logic
is implemented here.
"""

from abc import ABC, abstractmethod
from typing import List

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """
    Raised when an embedding provider fails to load its model or
    generate embeddings.

    This includes failures such as an unreachable embedding API, an
    invalid or missing model identifier, or malformed input text.
    """


class BaseEmbedding(ABC):
    """
    Abstract base class defining the common interface for all
    embedding providers in ResearchRAG.

    Every embedding provider (Sentence Transformers, OpenAI, Cohere,
    Voyage AI, Jina, HuggingFace, etc.) must inherit from this class
    and implement its abstract methods. This ensures that the
    embedding generation stage, retrieval stage, and vector store
    integration can all depend solely on this shared interface,
    remaining entirely agnostic to which concrete provider is
    configured.

    Subclasses are responsible for:
        - Loading their underlying model or client on demand (lazy
          loading), rather than at construction time.
        - Generating embeddings for batches of document text.
        - Generating an embedding for a single query string.
        - Reporting the dimensionality of the vectors they produce.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the active embedding model."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the underlying embedding model or initialize the provider
        client.

        Implementations should perform this loading lazily — that is,
        only when it is actually needed (e.g., on first use) rather
        than during object construction — so that instantiating an
        embedding provider is inexpensive and does not require network
        access or model weights to be available immediately.

        Calling this method more than once should be safe and should
        not reload an already-loaded model.

        Raises:
            EmbeddingError: If the model or client cannot be
                initialized, for example due to a missing model
                identifier, an unreachable provider API, or invalid
                credentials.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of document texts.

        Args:
            texts: List of text strings to embed, typically the
                content of document chunks.

        Returns:
            A list of embedding vectors, one per input text, in the
            same order as `texts`. Each vector is a list of floats
            whose length equals `embedding_dimension()`.

        Raises:
            EmbeddingError: If embedding generation fails for the
                given batch, for example due to a provider API error
                or invalid input.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text: The query text to embed.

        Returns:
            An embedding vector as a list of floats, whose length
            equals `embedding_dimension()`.

        Raises:
            EmbeddingError: If embedding generation fails for the
                given query, for example due to a provider API error
                or invalid input.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this provider.

        Returns:
            The number of dimensions in each embedding vector produced
            by this provider's model.

        Raises:
            EmbeddingError: If the dimensionality cannot be determined,
                for example because the model has not yet been loaded
                and its dimensionality cannot be inferred without
                loading it.
        """
        raise NotImplementedError