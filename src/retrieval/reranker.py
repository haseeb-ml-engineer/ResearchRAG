"""
Reranker for ResearchRAG.

This module defines `Reranker`, an optional secondary refinement stage
in the query pipeline. It accepts a list of `RetrievedChunk` objects
produced by `Retriever` and re-scores each chunk against the original
query using a cross-encoder model. Cross-encoders jointly encode the
query and each candidate chunk, producing a more precise relevance
score than the bi-encoder similarity used during initial retrieval,
at the cost of higher per-candidate latency.

`Reranker` performs relevance re-scoring only — it contains no
embedding generation, vector store access, prompt construction, or
LLM interaction.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)


class RerankerError(Exception):
    """Base exception for errors raised by `Reranker`."""


class RerankerModelLoadError(RerankerError):
    """Raised when the cross-encoder model fails to load."""


class InvalidRerankerInputError(RerankerError):
    """Raised when reranker input validation fails."""


@dataclass(frozen=True)
class RerankResult:
    """
    A single, provider-independent reranking result.

    Attributes:
        document_id: Unique identifier of the reranked chunk.
        page_content: Textual content of the reranked chunk.
        metadata: Metadata associated with the reranked chunk.
        original_score: Similarity score assigned during initial
            retrieval, preserved for auditability.
        rerank_score: Relevance score assigned by the cross-encoder
            model, where a higher value indicates greater relevance
            to the query.
    """

    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float


class Reranker:
    """
    Re-scores and re-ranks retrieved document chunks using a
    cross-encoder model.

    `Reranker` depends only on the `RetrievedChunk` data contract
    produced by `Retriever` and a cross-encoder model loaded from
    the `sentence-transformers` library. It knows nothing about
    embeddings, vector stores, prompts, or LLMs — its sole
    responsibility is turning a broad set of retrieval candidates
    into a precision-refined subset ordered by cross-encoder
    relevance.

    The underlying cross-encoder model is loaded lazily on first use
    rather than at construction time, so instantiating this class is
    inexpensive and does not require the model weights to be loaded
    immediately.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the reranker with its configuration.

        Args:
            model_name: Identifier of the cross-encoder model to
                use for reranking. Defaults to the value configured
                in application settings.
            default_top_k: Default number of top chunks to retain
                after reranking when not overridden by a method call.
                Defaults to the value configured in application
                settings.
        """
        self._model_name: str = model_name or settings.retrieval.reranker_model
        self._default_top_k: int = default_top_k or settings.retrieval.reranker_top_k
        self._model = None

        logger.info(
            "Reranker initialized (model=%s, default_top_k=%d)",
            self._model_name,
            self._default_top_k,
        )

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Re-score and re-rank retrieved chunks against a query.

        Each chunk is paired with the query and scored by the
        cross-encoder model. The chunks are then sorted by their
        new relevance scores in descending order and truncated to
        the requested `top_k`.

        Args:
            query: The original natural language query that produced
                the retrieved chunks.
            chunks: List of `RetrievedChunk` objects to rerank,
                typically the output of `Retriever.retrieve()`.
            top_k: Maximum number of chunks to retain after
                reranking. Defaults to this reranker's configured
                default.

        Returns:
            A list of `RerankResult` objects ordered from most to
            least relevant according to the cross-encoder, of
            length at most `top_k`.

        Raises:
            InvalidRerankerInputError: If `query` is empty, `chunks`
                is not a list of `RetrievedChunk`, or `top_k` is not
                positive.
            RerankerError: If cross-encoder scoring fails.
        """
        self._validate_inputs(query, chunks)
        resolved_top_k = top_k or self._default_top_k
        self._validate_top_k(resolved_top_k)

        logger.info(
            "Reranking %d chunk(s) for query (top_k=%d): %r",
            len(chunks),
            resolved_top_k,
            query,
        )

        # An empty chunk list is valid but trivially produces no results.
        if not chunks:
            logger.info("No chunks to rerank, returning empty results")
            return []

        self._load_model()
        assert self._model is not None

        start_time = time.perf_counter()

        try:
            query_chunk_pairs = [
                [query, chunk.page_content] for chunk in chunks
            ]
            scores = [float(score) for score in self._model.predict(query_chunk_pairs)]
        except Exception as error:
            raise RerankerError(
                f"Cross-encoder scoring failed: {error}"
            ) from error

        elapsed_seconds = time.perf_counter() - start_time

        results = self._build_results(chunks, scores)
        results.sort(key=lambda result: result.rerank_score, reverse=True)
        results = results[:resolved_top_k]

        logger.info(
            "Reranked %d chunk(s) to %d in %.3fs",
            len(chunks),
            len(results),
            elapsed_seconds,
        )

        return results

    def _load_model(self) -> None:
        """
        Load the cross-encoder model if it has not already been
        loaded.

        This method is safe to call multiple times: subsequent calls
        are no-ops once the model has been successfully loaded.

        Raises:
            RerankerModelLoadError: If the model fails to load, for
                example due to an invalid or unavailable model
                identifier.
        """
        if self._model is not None:
            return

        logger.info("Loading cross-encoder model: %s", self._model_name)
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception as error:
            raise RerankerModelLoadError(
                f"Failed to load cross-encoder model '{self._model_name}': {error}"
            ) from error

        logger.info(
            "Successfully loaded cross-encoder model: %s", self._model_name
        )

    @staticmethod
    def _build_results(
        chunks: List[RetrievedChunk],
        scores: List[float],
    ) -> List[RerankResult]:
        """
        Pair each retrieved chunk with its cross-encoder score to
        produce a list of `RerankResult` objects.

        Args:
            chunks: The original retrieved chunks, in the same order
                as `scores`.
            scores: Cross-encoder relevance scores, one per chunk,
                in the same order as `chunks`.

        Returns:
            A list of `RerankResult` objects, one per chunk, not yet
            sorted.
        """
        return [
            RerankResult(
                document_id=chunk.document_id,
                page_content=chunk.page_content,
                metadata=dict(chunk.metadata),
                original_score=chunk.similarity_score,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

    @staticmethod
    def _validate_inputs(query: str, chunks: List[RetrievedChunk]) -> None:
        """
        Validate reranker inputs prior to scoring.

        Args:
            query: The natural language query to validate.
            chunks: The list of retrieved chunks to validate.

        Raises:
            InvalidRerankerInputError: If `query` is not a non-empty
                string, or `chunks` is not a list of
                `RetrievedChunk` objects.
        """
        if not isinstance(query, str) or not query.strip():
            raise InvalidRerankerInputError(
                f"Query must be a non-empty string, got {query!r}"
            )

        if not isinstance(chunks, list):
            raise InvalidRerankerInputError(
                f"chunks must be a list of RetrievedChunk objects, "
                f"got {type(chunks)}"
            )

        for chunk in chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise InvalidRerankerInputError(
                    f"Every item in chunks must be a RetrievedChunk, "
                    f"got {type(chunk)}"
                )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """
        Validate the `top_k` parameter.

        Args:
            top_k: Maximum number of chunks to retain after
                reranking.

        Raises:
            InvalidRerankerInputError: If `top_k` is not a positive
                integer.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise InvalidRerankerInputError(
                f"top_k must be a positive integer, got {top_k}"
            )
