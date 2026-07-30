"""
Embedding manager for ResearchRAG.

This module defines `EmbeddingManager`, the single interface through
which the rest of the application performs embedding operations. It
selects and instantiates the configured embedding provider using a
factory pattern, and delegates all embedding work to that provider.
No other module should import or instantiate a concrete embedding
provider (e.g., `SentenceTransformerEmbedding`) directly.
"""

from typing import Dict, List, Optional, Type

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.base_embedding import BaseEmbedding, EmbeddingError
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding

logger = get_logger(__name__)


class UnsupportedProviderError(Exception):
    """Raised when the configured or requested embedding provider is not registered."""


class EmbeddingManager:
    """
    Manages embedding provider selection and delegates embedding
    operations to the active provider.

    `EmbeddingManager` is the only component in ResearchRAG that should
    interact with concrete embedding provider classes. All other
    modules (vector store, retriever, indexing pipeline, query
    pipeline) depend solely on this manager, which exposes a
    provider-agnostic API. Switching embedding providers requires only
    a configuration change, not a code change in any dependent module.
    """

    # Registry mapping provider names to their implementation class.
    # Adding a new provider requires only a new entry here (or a call
    # to `register_provider`); no other logic in this class changes.
    _PROVIDER_REGISTRY: Dict[str, Type[BaseEmbedding]] = {
        "sentence-transformers": SentenceTransformerEmbedding,
    }

    def __init__(self, provider_name: Optional[str] = None) -> None:
        """
        Initialize the embedding manager with the configured provider.

        Args:
            provider_name: Name of the embedding provider to use.
                Defaults to the provider configured in application
                settings.

        Raises:
            UnsupportedProviderError: If the requested provider is not
                registered.
            EmbeddingError: If the selected provider fails to
                initialize.
        """
        self._provider_name: str = provider_name or settings.embedding.provider
        self._provider: BaseEmbedding = self._create_provider(self._provider_name)

        logger.info("EmbeddingManager initialized with provider: %s", self._provider_name)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseEmbedding]) -> None:
        """
        Register a new embedding provider implementation.

        This allows new providers (e.g., OpenAI, Cohere, Voyage AI,
        Jina) to be added to the system without modifying any existing
        logic in `EmbeddingManager`.

        Args:
            name: Identifier used to select this provider via
                configuration.
            provider_class: A concrete `BaseEmbedding` subclass
                implementing the provider.

        Raises:
            TypeError: If `provider_class` does not subclass
                `BaseEmbedding`.
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise TypeError(
                f"provider_class must be a subclass of BaseEmbedding, got {provider_class}"
            )
        cls._PROVIDER_REGISTRY[name] = provider_class
        logger.info("Registered embedding provider: %s", name)

    def _create_provider(self, provider_name: str) -> BaseEmbedding:
        """
        Instantiate the embedding provider corresponding to the given
        name.

        Args:
            provider_name: Name of the embedding provider to
                instantiate.

        Returns:
            An initialized instance of the requested `BaseEmbedding`
            subclass.

        Raises:
            UnsupportedProviderError: If `provider_name` is not
                registered.
            EmbeddingError: If the provider class fails to initialize.
        """
        provider_class = self._PROVIDER_REGISTRY.get(provider_name)

        if provider_class is None:
            supported = sorted(self._PROVIDER_REGISTRY.keys())
            raise UnsupportedProviderError(
                f"Unsupported embedding provider '{provider_name}'. "
                f"Supported providers: {supported}"
            )

        logger.info("Creating embedding provider instance: %s", provider_name)
        try:
            return provider_class()
        except Exception as error:
            raise EmbeddingError(
                f"Failed to initialize embedding provider '{provider_name}': {error}"
            ) from error

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
            EmbeddingError: If `texts` is empty, contains invalid
                entries, or embedding generation fails.
        """
        self._validate_input(texts)
        logger.info(
            "Delegating embed_documents request for %d text(s) to provider: %s",
            len(texts),
            self._provider_name,
        )
        return self._provider.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text: The query text to embed.

        Returns:
            An embedding vector as a list of floats.

        Raises:
            EmbeddingError: If `text` is empty, is not a string, or
                embedding generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(f"Invalid query text for embedding: {text!r}")

        logger.info("Delegating embed_query request to provider: %s", self._provider_name)
        return self._provider.embed_query(text)

    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by the active
        embedding provider.

        Returns:
            The number of dimensions in each embedding vector.

        Raises:
            EmbeddingError: If the active provider cannot report its
                embedding dimension.
        """
        return self._provider.embedding_dimension()

    def get_provider_name(self) -> str:
        """
        Return the name of the currently active embedding provider.

        Returns:
            The provider name used to select this manager's active
            provider.
        """
        return self._provider_name

    def get_provider(self) -> BaseEmbedding:
        """Return the active embedding provider instance."""
        return self._provider

    @staticmethod
    def _validate_input(texts: List[str]) -> None:
        """
        Validate a batch of input texts before delegating to the
        active provider.

        Args:
            texts: List of text strings to validate.

        Raises:
            EmbeddingError: If `texts` is not a non-empty list of
                non-empty strings.
        """
        if not isinstance(texts, list) or not texts:
            raise EmbeddingError("Input texts must be a non-empty list of strings.")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"Invalid text input for embedding: {text!r}")