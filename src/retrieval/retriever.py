"""
Retriever for ResearchRAG.

This module defines `Retriever`, the entry point of the query
pipeline. It accepts a natural language query, generates its embedding
through an `EmbeddingManager`, and searches a `BaseVectorStore` for the
most relevant document chunks. `Retriever` performs retrieval only —
it contains no prompt construction, reranking, query rewriting, or
LLM interaction.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstores.base_vectorstore import BaseVectorStore, SimilaritySearchResult

logger = get_logger(__name__)


class RetrieverError(Exception):
    """Base exception for errors raised by `Retriever`."""


class InvalidQueryError(RetrieverError):
    """Raised when a query fails validation prior to retrieval."""


class InvalidRetrievalParameterError(RetrieverError):
    """Raised when `top_k` or `similarity_threshold` is invalid."""


@dataclass(frozen=True)
class RetrievedChunk:
    """
    A single, provider-independent retrieval result.

    Attributes:
        document_id: Unique identifier of the retrieved chunk.
        page_content: Textual content of the retrieved chunk.
        metadata: Metadata associated with the retrieved chunk.
        similarity_score: Similarity score of the match, where a higher
            value indicates greater similarity to the query.
    """

    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    similarity_score: float


class Retriever:
    """
    Retrieves the most relevant document chunks for a natural language
    query.

    `Retriever` depends only on the `EmbeddingManager` and
    `BaseVectorStore` abstractions, both supplied through the
    constructor. It knows nothing about prompt engineering, LLMs, or
    any serving layer — its sole responsibility is turning a query into
    a ranked, provider-independent list of relevant chunks.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_manager: EmbeddingManager,
        default_top_k: Optional[int] = None,
        default_similarity_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the retriever with its collaborating components.

        Args:
            vector_store: A `BaseVectorStore` implementation used to
                search for relevant document chunks.
            embedding_manager: An `EmbeddingManager` used to generate
                the query embedding.
            default_top_k: Default number of results to retrieve when
                not overridden by a method call. Defaults to the value
                configured in application settings.
            default_similarity_threshold: Default minimum similarity
                score a chunk must meet to be returned when not
                overridden by a method call. Defaults to the value
                configured in application settings.
        """
        self._vector_store = vector_store
        self._embedding_manager = embedding_manager
        self._default_top_k = default_top_k or settings.retrieval.top_k
        self._default_similarity_threshold = (
            default_similarity_threshold
            if default_similarity_threshold is not None
            else settings.retrieval.similarity_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a query using
        configured defaults.

        Args:
            query: Natural language query text.
            top_k: Maximum number of chunks to retrieve. Defaults to
                this retriever's configured default.
            filters: Optional metadata filters to constrain the
                search.

        Returns:
            A list of `RetrievedChunk` objects ordered from most to
            least relevant.

        Raises:
            InvalidQueryError: If `query` is empty or not a string.
            InvalidRetrievalParameterError: If `top_k` is not positive.
        """
        return self._search(
            query=query,
            top_k=top_k,
            filters=filters,
            similarity_threshold=self._default_similarity_threshold,
        )

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks with explicit control over
        the similarity threshold applied.

        Args:
            query: Natural language query text.
            top_k: Maximum number of chunks to retrieve. Defaults to
                this retriever's configured default.
            filters: Optional metadata filters to constrain the
                search.
            similarity_threshold: Minimum similarity score a chunk
                must meet to be returned. Defaults to this retriever's
                configured default.

        Returns:
            A list of `RetrievedChunk` objects ordered from most to
            least relevant, each carrying its similarity score.

        Raises:
            InvalidQueryError: If `query` is empty or not a string.
            InvalidRetrievalParameterError: If `top_k` or
                `similarity_threshold` is invalid.
        """
        resolved_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._default_similarity_threshold
        )
        return self._search(
            query=query,
            top_k=top_k,
            filters=filters,
            similarity_threshold=resolved_threshold,
        )

    def retrieve_by_metadata(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks constrained to those
        matching the given metadata filters.

        Args:
            query: Natural language query text.
            filters: Metadata filters the returned chunks must match
                (e.g., restricting results to a specific source or
                document type).
            top_k: Maximum number of chunks to retrieve. Defaults to
                this retriever's configured default.

        Returns:
            A list of `RetrievedChunk` objects ordered from most to
            least relevant, all matching the given filters.

        Raises:
            InvalidQueryError: If `query` is empty or not a string.
            InvalidRetrievalParameterError: If `top_k` is invalid.
            RetrieverError: If `filters` is empty.
        """
        if not filters:
            raise RetrieverError("filters must not be empty for retrieve_by_metadata.")

        return self._search(
            query=query,
            top_k=top_k,
            filters=filters,
            similarity_threshold=self._default_similarity_threshold,
        )

    def validate_query(self, query: str) -> None:
        """
        Validate a natural language query prior to retrieval.

        Args:
            query: Natural language query text to validate.

        Raises:
            InvalidQueryError: If `query` is not a string or is empty
                after stripping whitespace.
        """
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError(f"Query must be a non-empty string, got {query!r}")

    def _search(
        self,
        query: str,
        top_k: Optional[int],
        filters: Optional[Dict[str, Any]],
        similarity_threshold: float,
    ) -> List[RetrievedChunk]:
        """
        Execute the core retrieval sequence: validate, embed, search,
        and filter.

        Args:
            query: Natural language query text.
            top_k: Maximum number of chunks to retrieve, or None to
                use this retriever's configured default.
            filters: Optional metadata filters to constrain the
                search.
            similarity_threshold: Minimum similarity score a chunk
                must meet to be included in the results.

        Returns:
            A list of `RetrievedChunk` objects ordered from most to
            least relevant.

        Raises:
            InvalidQueryError: If `query` is empty or not a string.
            InvalidRetrievalParameterError: If `top_k` or
                `similarity_threshold` is invalid.
        """
        self.validate_query(query)
        resolved_top_k = top_k or self._default_top_k
        self._validate_retrieval_parameters(resolved_top_k, similarity_threshold)

        logger.info(
            "Retrieving for query (top_k=%d, threshold=%.3f, filters=%s): %r",
            resolved_top_k,
            similarity_threshold,
            filters,
            query,
        )

        start_time = time.perf_counter()

        try:
            query_embedding = self._embedding_manager.embed_query(query)
        except Exception as error:
            raise RetrieverError(f"Failed to generate query embedding: {error}") from error

        try:
            raw_results = self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=resolved_top_k,
                filters=filters,
            )
        except Exception as error:
            raise RetrieverError(f"Vector store search failed: {error}") from error

        elapsed_seconds = time.perf_counter() - start_time

        chunks = self._to_retrieved_chunks(raw_results, similarity_threshold)

        logger.info(
            "Retrieved %d chunk(s) in %.3fs (before threshold: %d)",
            len(chunks),
            elapsed_seconds,
            len(raw_results),
        )

        return chunks

    @staticmethod
    def _to_retrieved_chunks(
        raw_results: List[SimilaritySearchResult],
        similarity_threshold: float,
    ) -> List[RetrievedChunk]:
        """
        Convert vector store search results into provider-independent
        retrieval results, filtering out chunks below the similarity
        threshold.

        Args:
            raw_results: Search results returned by the vector store.
            similarity_threshold: Minimum similarity score a chunk
                must meet to be included.

        Returns:
            A list of `RetrievedChunk` objects meeting the similarity
            threshold, in the original relevance order.
        """
        return [
            RetrievedChunk(
                document_id=result.document_id,
                page_content=result.document.page_content,
                metadata=dict(result.document.metadata),
                similarity_score=result.score,
            )
            for result in raw_results
            if result.score >= similarity_threshold
        ]

    @staticmethod
    def _validate_retrieval_parameters(top_k: int, similarity_threshold: float) -> None:
        """
        Validate retrieval parameters prior to executing a search.

        Args:
            top_k: Maximum number of chunks to retrieve.
            similarity_threshold: Minimum similarity score a chunk
                must meet to be returned.

        Raises:
            InvalidRetrievalParameterError: If `top_k` is not positive
                or `similarity_threshold` is outside the range [0, 1].
        """
        if top_k <= 0:
            raise InvalidRetrievalParameterError(f"top_k must be greater than 0, got {top_k}")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise InvalidRetrievalParameterError(
                f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
            )