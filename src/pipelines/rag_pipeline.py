"""
RAG pipeline orchestrator for ResearchRAG.

This module defines `RAGPipeline`, the central orchestration layer of
the query path. It coordinates the complete Retrieval-Augmented
Generation workflow — query validation, document retrieval, optional
reranking, context assembly, LLM generation, and structured response
construction — without implementing any of that business logic itself.

`RAGPipeline` delegates every capability to purpose-built components
(`Retriever`, `Reranker`, and a `BaseLLM` implementation), all supplied
through the constructor. It never generates embeddings, accesses a
vector database, constructs prompts, or interacts with a model provider
directly; its sole responsibility is sequencing and connecting the
stages that do.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.llms.base_llm import BaseLLM
from src.retrieval.reranker import RerankResult, Reranker
from src.retrieval.retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RAGPipelineError(Exception):
    """Base exception for errors raised by `RAGPipeline`."""


class InvalidQueryError(RAGPipelineError):
    """Raised when a query fails validation prior to pipeline execution."""


class RetrievalFailedError(RAGPipelineError):
    """Raised when the retrieval stage fails to produce results."""


class GenerationFailedError(RAGPipelineError):
    """Raised when the LLM generation stage fails."""


class NoDocumentsRetrievedError(RAGPipelineError):
    """Raised when retrieval returns zero relevant documents."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """
    A single source citation linking part of an answer to its
    originating document chunk.

    Attributes:
        document_id: Unique identifier of the cited chunk.
        page_content: Textual content of the cited chunk.
        metadata: Metadata associated with the cited chunk.
        relevance_score: The score used to rank this chunk, either
            the retrieval similarity score or the rerank score when
            reranking was applied.
    """

    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    relevance_score: float


@dataclass(frozen=True)
class RAGResponse:
    """
    Provider-independent structured response from the RAG pipeline.

    Attributes:
        answer: The generated answer text produced by the LLM.
        retrieved_documents: The document chunks used to construct
            the context for generation, in relevance order.
        citations: Source citations derived from the retrieved (and
            optionally reranked) documents.
        confidence_score: An aggregate relevance score reflecting
            how well the retrieved documents matched the query,
            computed as the mean relevance score of the citations.
        retrieval_time: Wall-clock time in seconds spent on the
            retrieval stage (including reranking, when applied).
        generation_time: Wall-clock time in seconds spent on the
            LLM generation stage.
        total_time: Wall-clock time in seconds for the entire
            pipeline execution from query receipt to response
            construction.
        reranking_applied: Whether the reranking stage was applied
            during this pipeline execution.
    """

    answer: str
    retrieved_documents: List[Dict[str, Any]]
    citations: List[Citation]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_time: float
    reranking_applied: bool = field(default=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Type alias for the union of chunk types that flow through the
# pipeline after the retrieval (and optional reranking) stage.
_RankedChunk = Union[RetrievedChunk, RerankResult]


class RAGPipeline:
    """
    Orchestrates the complete Retrieval-Augmented Generation workflow.

    `RAGPipeline` sequences the query path stages — validation,
    retrieval, optional reranking, context assembly, LLM generation,
    and response construction — by delegating each capability to
    purpose-built components received through the constructor:

    - `Retriever` for document retrieval.
    - `Reranker` (optional) for precision refinement.
    - `BaseLLM` for answer generation.

    The pipeline contains no retrieval algorithms, no embedding logic,
    no vector database operations, no prompt engineering, and no model
    provider code. It only coordinates the components that do.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        reranker: Optional[Reranker] = None,
        retrieval_top_k: Optional[int] = None,
        reranker_top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the RAG pipeline with its collaborating components.

        Args:
            retriever: A `Retriever` instance used to retrieve
                relevant document chunks for a query.
            llm: A `BaseLLM` implementation used to generate answers
                from retrieved context.
            reranker: An optional `Reranker` instance used to refine
                retrieval results before context assembly. When None,
                the reranking stage is skipped.
            retrieval_top_k: Number of chunks to retrieve from the
                vector store. Defaults to the value configured in
                application settings.
            reranker_top_k: Number of chunks to retain after
                reranking. Defaults to the value configured in
                application settings. Ignored when no reranker is
                provided.
        """
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k or settings.retrieval.top_k
        self._reranker_top_k = reranker_top_k or settings.retrieval.reranker_top_k

        logger.info(
            "RAGPipeline initialized (retrieval_top_k=%d, reranker=%s)",
            self._retrieval_top_k,
            "enabled" if self._reranker is not None else "disabled",
        )

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Validates the query, retrieves relevant documents, optionally
        reranks them, assembles context, generates an answer through
        the LLM, and returns a structured response with citations and
        timing metrics.

        Args:
            query: Natural language query from the user.
            top_k: Maximum number of chunks to retrieve. Defaults to
                this pipeline's configured retrieval top_k.
            filters: Optional metadata filters to constrain the
                retrieval search.

        Returns:
            A `RAGResponse` containing the generated answer, source
            citations, confidence score, and timing breakdown.

        Raises:
            InvalidQueryError: If the query is empty or not a string.
            RetrievalFailedError: If the retrieval stage fails.
            NoDocumentsRetrievedError: If no relevant documents are
                found for the query.
            GenerationFailedError: If the LLM generation stage fails.
        """
        self._validate_query(query)

        logger.info("RAG pipeline started for query: %r", query)
        pipeline_start = time.perf_counter()

        # --- Retrieval stage ---
        ranked_chunks, retrieval_time, reranking_applied = (
            self.retrieve_context(query, top_k=top_k, filters=filters)
        )

        # --- Context assembly ---
        context = self.build_context(ranked_chunks)

        # --- Generation stage ---
        generation_start = time.perf_counter()

        try:
            answer_text = self._llm.generate(prompt=query, system_prompt=context).content
        except Exception as error:
            raise GenerationFailedError(
                f"LLM generation failed for query: {error}"
            ) from error

        generation_time = time.perf_counter() - generation_start
        total_time = time.perf_counter() - pipeline_start

        logger.info(
            "LLM generation completed in %.3fs", generation_time
        )

        # --- Response construction ---
        response = self.build_response(
            answer_text=answer_text,
            ranked_chunks=ranked_chunks,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            reranking_applied=reranking_applied,
        )

        logger.info(
            "RAG pipeline completed: total=%.3fs retrieval=%.3fs "
            "generation=%.3fs confidence=%.3f docs=%d",
            response.total_time,
            response.retrieval_time,
            response.generation_time,
            response.confidence_score,
            len(response.citations),
        )

        return response

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Execute the retrieval and optional reranking stages.

        Args:
            query: Natural language query text.
            top_k: Maximum number of chunks to retrieve. Defaults to
                this pipeline's configured retrieval top_k.
            filters: Optional metadata filters to constrain the
                retrieval search.

        Returns:
            A three-element tuple of:
            - The ranked chunks (as `RetrievedChunk` or
              `RerankResult` objects).
            - The total wall-clock retrieval time in seconds
              (including reranking).
            - Whether reranking was applied.

        Raises:
            RetrievalFailedError: If the retrieval stage fails.
            NoDocumentsRetrievedError: If no relevant documents are
                found.
        """
        resolved_top_k = top_k or self._retrieval_top_k
        retrieval_start = time.perf_counter()

        try:
            chunks = self._retriever.retrieve(
                query=query, top_k=resolved_top_k, filters=filters
            )
        except Exception as error:
            raise RetrievalFailedError(
                f"Retrieval failed for query: {error}"
            ) from error

        retrieval_elapsed = time.perf_counter() - retrieval_start
        logger.info(
            "Retrieval returned %d chunk(s) in %.3fs",
            len(chunks),
            retrieval_elapsed,
        )

        if not chunks:
            raise NoDocumentsRetrievedError(
                "No relevant documents found for the given query."
            )

        # --- Optional reranking ---
        reranking_applied = False
        ranked_chunks: List[_RankedChunk] = list(chunks)

        if self._reranker is not None:
            rerank_start = time.perf_counter()
            try:
                reranked = self._reranker.rerank(
                    query=query,
                    chunks=chunks,
                    top_k=self._reranker_top_k,
                )
            except Exception as error:
                raise RetrievalFailedError(
                    f"Reranking failed for query: {error}"
                ) from error

            rerank_elapsed = time.perf_counter() - rerank_start
            logger.info(
                "Reranking refined %d chunk(s) to %d in %.3fs",
                len(chunks),
                len(reranked),
                rerank_elapsed,
            )
            ranked_chunks = list(reranked)
            reranking_applied = True

        total_retrieval_time = time.perf_counter() - retrieval_start
        return ranked_chunks, total_retrieval_time, reranking_applied

    def build_context(self, ranked_chunks: List[_RankedChunk]) -> str:
        """
        Assemble a formatted context string from ranked document
        chunks for LLM consumption.

        Each chunk is rendered as a numbered block containing its
        source metadata and text content, separated by visual
        delimiters for readability.

        Args:
            ranked_chunks: Ordered list of retrieved (and optionally
                reranked) document chunks.

        Returns:
            A single string containing all chunks formatted as
            numbered context passages ready for prompt inclusion.

        Raises:
            RAGPipelineError: If context assembly fails.
        """
        if not ranked_chunks:
            return ""

        context_blocks: List[str] = []

        for index, chunk in enumerate(ranked_chunks, start=1):
            source = chunk.metadata.get("source", "Unknown")
            context_blocks.append(
                f"[Document {index}] (Source: {source})\n"
                f"{chunk.page_content}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        logger.info(
            "Built context from %d chunk(s) (%d characters)",
            len(ranked_chunks),
            len(context),
        )

        return context

    def build_response(
        self,
        answer_text: str,
        ranked_chunks: List[_RankedChunk],
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        reranking_applied: bool,
    ) -> RAGResponse:
        """
        Construct a structured `RAGResponse` from pipeline outputs.

        Args:
            answer_text: The generated answer text from the LLM.
            ranked_chunks: The document chunks used for context, in
                relevance order.
            retrieval_time: Wall-clock time spent on retrieval
                (including reranking) in seconds.
            generation_time: Wall-clock time spent on LLM generation
                in seconds.
            total_time: Wall-clock time for the entire pipeline
                execution in seconds.
            reranking_applied: Whether the reranking stage was applied.

        Returns:
            A fully populated `RAGResponse` instance.
        """
        citations = self._build_citations(ranked_chunks)
        confidence_score = self._compute_confidence(citations)
        retrieved_documents = self._chunks_to_dicts(ranked_chunks)

        return RAGResponse(
            answer=answer_text,
            retrieved_documents=retrieved_documents,
            citations=citations,
            confidence_score=confidence_score,
            retrieval_time=round(retrieval_time, 4),
            generation_time=round(generation_time, 4),
            total_time=round(total_time, 4),
            reranking_applied=reranking_applied,
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Verify the operational status of all pipeline components.

        Returns:
            A dictionary reporting the health of the retriever, the
            LLM, and the reranker (when configured), plus the
            overall pipeline status.
        """
        status: Dict[str, Any] = {
            "pipeline": "healthy",
            "retriever": "healthy",
            "llm": "unknown",
            "reranker": "not_configured",
        }

        try:
            is_llm_healthy = self._llm.health_check()
            status["llm"] = "healthy" if is_llm_healthy else "unhealthy"
        except Exception as error:
            status["llm"] = f"unhealthy: {error}"
            status["pipeline"] = "degraded"
            logger.warning("LLM health check failed: %s", error)

        if self._reranker is not None:
            status["reranker"] = "configured"

        if status["llm"] != "healthy":
            status["pipeline"] = "degraded"

        logger.info("Health check result: %s", status)
        return status

    @staticmethod
    def _build_citations(
        ranked_chunks: List[_RankedChunk],
    ) -> List[Citation]:
        """
        Extract citations from ranked document chunks.

        Args:
            ranked_chunks: Ordered list of retrieved (and optionally
                reranked) document chunks.

        Returns:
            A list of `Citation` objects, one per chunk, preserving
            the relevance ordering.
        """
        citations: List[Citation] = []
        for chunk in ranked_chunks:
            if isinstance(chunk, RerankResult):
                score = chunk.rerank_score
            else:
                score = chunk.similarity_score

            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    page_content=chunk.page_content,
                    metadata=dict(chunk.metadata),
                    relevance_score=score,
                )
            )
        return citations

    @staticmethod
    def _compute_confidence(citations: List[Citation]) -> float:
        """
        Compute an aggregate confidence score from citation relevance
        scores.

        The confidence score is the arithmetic mean of all citation
        relevance scores, providing a single numeric indicator of how
        well the retrieved documents matched the query.

        Args:
            citations: List of citations with relevance scores.

        Returns:
            A float representing the mean relevance score, or 0.0 if
            no citations are present.
        """
        if not citations:
            return 0.0
        total = sum(citation.relevance_score for citation in citations)
        return round(total / len(citations), 4)

    @staticmethod
    def _chunks_to_dicts(
        ranked_chunks: List[_RankedChunk],
    ) -> List[Dict[str, Any]]:
        """
        Convert ranked chunks into plain dictionaries suitable for
        serialization.

        Args:
            ranked_chunks: Ordered list of retrieved (and optionally
                reranked) document chunks.

        Returns:
            A list of dictionaries, each containing the chunk's
            document ID, text content, metadata, and relevance score.
        """
        documents: List[Dict[str, Any]] = []
        for chunk in ranked_chunks:
            if isinstance(chunk, RerankResult):
                score = chunk.rerank_score
            else:
                score = chunk.similarity_score

            documents.append({
                "document_id": chunk.document_id,
                "page_content": chunk.page_content,
                "metadata": dict(chunk.metadata),
                "relevance_score": score,
            })
        return documents

    @staticmethod
    def _validate_query(query: str) -> None:
        """
        Validate a user query before pipeline execution.

        Args:
            query: The user's natural language query.

        Raises:
            InvalidQueryError: If `query` is not a string or is empty
                after stripping whitespace.
        """
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError(
                f"Query must be a non-empty string, got {query!r}"
            )
