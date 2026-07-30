"""
API Routes for ResearchRAG.

This module defines all REST API endpoints. It acts as the HTTP interface,
routing incoming requests to the appropriate backend pipelines via
dependency injection. These routes contain absolutely zero business logic;
they exist solely to validate incoming JSON, invoke the pipelines, handle
domain exceptions, and return strictly formatted response models.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_indexing_pipeline,
    get_rag_pipeline,
    get_settings,
    get_vector_store,
)
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.llms.base_llm import LLMConfigurationError, LLMGenerationError
from src.pipelines.indexing_pipeline import IndexingPipeline, IndexingPipelineError
from src.pipelines.rag_pipeline import RAGPipeline, RAGPipelineError
from src.pipelines.retrieval_pipeline import NoResultsFoundError
from src.vectorstores.base_vectorstore import BaseVectorStore

logger = get_logger(__name__)

# Create the main API router
router = APIRouter()


from src.api.schemas import (
    ConfigResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    StatisticsResponse,
)


# ---------------------------------------------------------------------------
# System Endpoints
# ---------------------------------------------------------------------------

@router.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    """Root endpoint verifying API accessibility."""
    return {"message": "Welcome to the ResearchRAG API"}


@router.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and orchestrators.
    Returns basic API status.
    """
    return {
        "status": "healthy",
        "service": "ResearchRAG",
        "timestamp": time.time(),
    }


@router.get("/configuration", response_model=ConfigResponse, tags=["System"])
async def get_configuration(
    config: Settings = Depends(get_settings),
) -> ConfigResponse:
    """
    Retrieve the current active configuration for the RAG system.
    Safely exposes provider selections without revealing API keys.
    """
    return ConfigResponse(
        llm_provider=config.llm.provider,
        embedding_provider=config.embedding.provider,
        vector_store=config.vector_store.provider,
    )


@router.get("/statistics", response_model=StatisticsResponse, tags=["System"])
async def get_statistics(
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> StatisticsResponse:
    """
    Retrieve macroscopic statistics about the current vector database.
    """
    try:
        # Assuming the vector store provides a way to get counts
        # This delegates to the vector store implementation
        stats = vector_store.get_collection_statistics() if hasattr(vector_store, 'get_collection_statistics') else {"total_documents": 0, "total_chunks": 0}
        return StatisticsResponse(**stats)
    except Exception as error:
        logger.error("Failed to fetch vector database statistics: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database statistics.",
        )


# ---------------------------------------------------------------------------
# Generation Endpoints
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_system(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """
    Execute a Retrieval-Augmented Generation (RAG) query.
    
    This endpoint validates the user's question, triggers the retrieval
    and generation pipelines, and formats the latency and sources into
    a consistent response.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty.",
        )

    logger.info("API received query: %r", request.question)

    try:
        # Execute the core business logic via the injected pipeline
        # The pipeline internally handles overrides if passed
        result = pipeline.answer(query=request.question)

        retrieved_sources = []
        similarity_scores = []
        metadata_list = []

        for document in result.retrieved_documents:
            metadata = dict(document.get("metadata", {}))
            score = float(document.get("relevance_score", 0.0) or 0.0)
            similarity_scores.append(score)
            metadata_list.append(metadata)
            retrieved_sources.append(
                {
                    "content": document.get("page_content", ""),
                    "metadata": metadata,
                    "score": score,
                    "similarity_score": score,
                    "filename": metadata.get("filename") or metadata.get("source_file_name") or metadata.get("source"),
                    "page_number": metadata.get("page_number") or metadata.get("page"),
                    "source_path": metadata.get("source_path") or metadata.get("source_file_path") or metadata.get("absolute_path"),
                }
            )

        # Map pipeline generic response to API-specific Response Model
        return QueryResponse(
            answer=result.answer,
            retrieved_sources=retrieved_sources,
            similarity_scores=similarity_scores,
            metadata=metadata_list,
            timings={
                "retrieval": result.retrieval_time,
                "generation": result.generation_time,
                "total": result.total_time,
            },
            retrieval_latency=result.retrieval_time,
            generation_latency=result.generation_time,
            total_latency=result.total_time,
        )

    except NoResultsFoundError:
        logger.warning("No context found for query: %r", request.question)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found to answer the query.",
        )
    except (LLMConfigurationError, LLMGenerationError) as error:
        logger.error("LLM failure during API query: %s", error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Language model provider failed: {error}",
        )
    except RAGPipelineError as error:
        logger.error("Pipeline error during API query: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal pipeline execution failed.",
        )
    except Exception as error:
        logger.critical("Unexpected error during API query: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected critical error occurred.",
        )


# ---------------------------------------------------------------------------
# Ingestion Endpoints
# ---------------------------------------------------------------------------

@router.post("/index", response_model=IndexResponse, tags=["Ingestion"])
async def index_documents(
    request: IndexRequest,
    pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
) -> IndexResponse:
    """
    Ingest documents into the vector database.
    
    Triggers the ETL pipeline to load, chunk, embed, and store
    documents from the provided source URI.
    """
    if not request.source or not request.source.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'source' field cannot be empty.",
        )

    logger.info("API requested document indexing for source: %r", request.source)

    try:
        report = pipeline.run(source=request.source)

        return IndexResponse(
            indexed_documents=report.indexed_documents,
            indexed_chunks=report.indexed_chunks,
            processing_time=report.elapsed_time_seconds,
            failures=report.failed_documents,
        )

    except IndexingPipelineError as error:
        logger.error("Indexing failed for source %r: %s", request.source, error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing pipeline failed: {error}",
        )
    except Exception as error:
        logger.critical("Unexpected error during indexing: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected critical error occurred during indexing.",
        )


@router.post("/rebuild-index", response_model=Dict[str, str], tags=["Ingestion"])
async def rebuild_index(
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> Dict[str, str]:
    """
    Dangerously clear the existing index and prepare for a clean rebuild.
    """
    logger.warning("API requested full index rebuild (deletion).")
    try:
        reset_collection = getattr(vector_store, "reset_collection", None)
        if callable(reset_collection):
            reset_collection()
            logger.info("Vector store successfully reset via API.")
            return {"status": "success", "message": "Index has been cleared and rebuilt."}
        else:
            raise NotImplementedError("The current vector store does not support resets.")
    except Exception as error:
        logger.error("Failed to rebuild index: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset the vector database.",
        )
