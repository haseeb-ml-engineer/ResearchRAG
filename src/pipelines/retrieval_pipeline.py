"""
Retrieval pipeline orchestrator for ResearchRAG.

This module defines `RetrievalPipeline`, an orchestration layer that
coordinates the full retrieval workflow. It accepts user queries,
retrieves document chunks, applies metadata and similarity filters,
optionally performs cross-encoder reranking, and produces a finalized,
formatted context ready for downstream consumption.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.reranker import RerankResult, Reranker
from src.retrieval.retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)


class RetrievalPipelineError(Exception):
    """Base exception for errors raised by `RetrievalPipeline`."""


class InvalidQueryError(RetrievalPipelineError):
    """Raised when a query fails validation."""


class InvalidRetrievalParameterError(RetrievalPipelineError):
    """Raised when Top-K or thresholds are invalid."""


class NoResultsFoundError(RetrievalPipelineError):
    """Raised when retrieval yields zero documents after filtering."""


@dataclass(frozen=True)
class RetrievedDocument:
    """
    A single, finalized document chunk returned by the pipeline.

    Attributes:
        document_id: Unique identifier of the chunk.
        page_content: The textual content.
        metadata: Associated source metadata.
        similarity_score: The original bi-encoder retrieval score.
        rerank_score: The cross-encoder rerank score, if applied.
        final_score: The decisive score used to rank this document.
    """

    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rerank_score: Optional[float] = None
    final_score: float = 0.0


@dataclass(frozen=True)
class RetrievalResponse:
    """
    Provider-independent structured response from the retrieval pipeline.

    Attributes:
        query: The original user query.
        documents: List of finalized, ranked documents.
        context_string: All document text formatted into a single string.
        retrieval_time_seconds: Time spent in initial retrieval.
        reranking_time_seconds: Time spent reranking, if applied.
        total_time_seconds: Total execution time of the pipeline.
        reranking_applied: Whether the reranking stage executed.
        filters_applied: Metadata filters applied during retrieval.
    """

    query: str
    documents: List[RetrievedDocument]
    context_string: str
    retrieval_time_seconds: float
    reranking_time_seconds: Optional[float]
    total_time_seconds: float
    reranking_applied: bool
    filters_applied: Dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """
    Orchestrates the complete document retrieval workflow.

    `RetrievalPipeline` sequences retrieval, filtering, reranking, and
    context assembly. It delegates algorithmic logic to injected components
    (`Retriever`, `Reranker`) and remains completely unaware of language
    models, generative pipelines, or frontend frameworks.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        default_top_k: Optional[int] = None,
        default_similarity_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the retrieval pipeline with its collaborating components.

        Args:
            retriever: Component responsible for vector similarity search.
            reranker: Optional component for precision re-scoring.
            default_top_k: Default maximum results to retrieve.
            default_similarity_threshold: Minimum score required to keep a chunk.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._default_top_k = default_top_k or settings.retrieval.top_k
        self._default_similarity_threshold = (
            default_similarity_threshold or settings.retrieval.similarity_threshold
        )

        logger.info(
            "RetrievalPipeline initialized (top_k=%d, threshold=%.2f, reranker=%s)",
            self._default_top_k,
            self._default_similarity_threshold,
            "enabled" if self._reranker else "disabled",
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """
        Execute the full retrieval workflow.

        Args:
            query: The user's search query.
            top_k: Maximum chunks to return. Defaults to configuration.
            similarity_threshold: Minimum score required. Defaults to config.
            filters: Metadata filters to constrain the initial search.

        Returns:
            A `RetrievalResponse` containing the ranked chunks, context string,
            and timing statistics.

        Raises:
            InvalidQueryError: If the query is empty or malformed.
            InvalidRetrievalParameterError: If parameters are out of bounds.
            NoResultsFoundError: If no documents pass the filters.
            RetrievalPipelineError: If an underlying component fails.
        """
        self._validate_inputs(query, top_k, similarity_threshold)

        resolved_top_k = top_k or self._default_top_k
        resolved_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._default_similarity_threshold
        )
        resolved_filters = filters or {}

        logger.info("Retrieval pipeline started for query: %r", query)
        total_start = time.perf_counter()

        # 1. Retrieval Stage
        retrieval_start = time.perf_counter()
        try:
            raw_chunks = self._retriever.retrieve(
                query=query,
                top_k=resolved_top_k,
                filters=resolved_filters,
            )
        except Exception as error:
            raise RetrievalPipelineError(f"Retriever failed: {error}") from error

        retrieval_time = time.perf_counter() - retrieval_start

        # 2. Similarity Filtering Stage
        filtered_chunks = self._filter_results(raw_chunks, resolved_threshold)

        if not filtered_chunks:
            logger.info(
                "No documents met the similarity threshold (%.2f)",
                resolved_threshold,
            )
            raise NoResultsFoundError("No relevant documents found after filtering.")

        # 3. Reranking Stage (Optional)
        rerank_time = None
        reranking_applied = False
        final_docs: List[RetrievedDocument] = []

        if self._reranker is not None:
            rerank_start = time.perf_counter()
            try:
                # Reranker internally handles sorting by new scores
                ranked_chunks = self._reranker.rerank(
                    query=query,
                    chunks=filtered_chunks,
                    top_k=resolved_top_k,
                )
            except Exception as error:
                raise RetrievalPipelineError(f"Reranker failed: {error}") from error

            rerank_time = time.perf_counter() - rerank_start
            reranking_applied = True

            # Map RerankResult back to standard representation
            final_docs = [
                RetrievedDocument(
                    document_id=chunk.document_id,
                    page_content=chunk.page_content,
                    metadata=dict(chunk.metadata),
                    similarity_score=chunk.original_score,
                    rerank_score=chunk.rerank_score,
                    final_score=chunk.rerank_score,
                )
                for chunk in ranked_chunks
            ]
        else:
            # No reranker; preserve order and map RetrievedChunk
            final_docs = [
                RetrievedDocument(
                    document_id=chunk.document_id,
                    page_content=chunk.page_content,
                    metadata=dict(chunk.metadata),
                    similarity_score=chunk.similarity_score,
                    rerank_score=None,
                    final_score=chunk.similarity_score,
                )
                for chunk in filtered_chunks
            ]

        # 4. Limit Final Top-K
        final_docs = final_docs[:resolved_top_k]

        if not final_docs:
            raise NoResultsFoundError("No relevant documents found after reranking.")

        # 5. Build Context
        context_string = self.build_context(final_docs)
        total_time = time.perf_counter() - total_start

        logger.info(
            "Retrieval pipeline completed: total=%.3fs docs=%d reranked=%s",
            total_time,
            len(final_docs),
            str(reranking_applied),
        )

        return RetrievalResponse(
            query=query,
            documents=final_docs,
            context_string=context_string,
            retrieval_time_seconds=round(retrieval_time, 4),
            reranking_time_seconds=round(rerank_time, 4) if rerank_time else None,
            total_time_seconds=round(total_time, 4),
            reranking_applied=reranking_applied,
            filters_applied=resolved_filters,
        )

    def build_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Assemble retrieved documents into a formatted context string.

        Args:
            documents: Finalized, ranked document chunks.

        Returns:
            A single string containing all chunks formatted with source info.
        """
        if not documents:
            return ""

        blocks = []
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            blocks.append(
                f"[Document {index}] (Source: {source})\n"
                f"{doc.page_content}"
            )

        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _filter_results(
        chunks: List[RetrievedChunk],
        threshold: float,
    ) -> List[RetrievedChunk]:
        """
        Filter retrieved chunks by minimum similarity threshold.

        Args:
            chunks: The raw chunks returned by the retriever.
            threshold: Minimum required similarity score.

        Returns:
            Chunks that meet or exceed the threshold.
        """
        return [chunk for chunk in chunks if chunk.similarity_score >= threshold]

    @staticmethod
    def _validate_inputs(
        query: str,
        top_k: Optional[int],
        similarity_threshold: Optional[float],
    ) -> None:
        """
        Validate input parameters before executing the pipeline.

        Args:
            query: The search query.
            top_k: Requested maximum results.
            similarity_threshold: Requested minimum score.

        Raises:
            InvalidQueryError: If query is empty.
            InvalidRetrievalParameterError: If parameters are out of bounds.
        """
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError(f"Query must be a non-empty string, got {query!r}")

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise InvalidRetrievalParameterError(
                f"top_k must be a positive integer, got {top_k}"
            )

        if similarity_threshold is not None and not isinstance(
            similarity_threshold, (int, float)
        ):
            raise InvalidRetrievalParameterError(
                f"similarity_threshold must be a number, got {type(similarity_threshold)}"
            )
