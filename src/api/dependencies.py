"""
Centralized dependency injection layer for the FastAPI application.

This module provides reusable factory functions (dependencies) that
instantiate and manage the lifecycle of core ResearchRAG services.
API routes use `Depends()` to request these services, ensuring that
business logic, database connections, and model clients are decoupled
from HTTP transport logic.
"""

from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.base_embedding import BaseEmbedding
from src.embeddings.embedding_manager import EmbeddingManager
from src.llms.base_llm import BaseLLM, LLMConfigurationError
from src.llms.llm_factory import LLMFactory
from src.loaders.pdf_loader import PDFLoader
from src.pipelines.indexing_pipeline import IndexingPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from src.pipelines.retrieval_pipeline import RetrievalPipeline
from src.preprocessing.splitter import DocumentSplitter
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.vectorstores.index_manager import IndexManager

# Module-level logger for initialization events
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core Configuration & Logging
# ---------------------------------------------------------------------------

@lru_cache()
def get_settings():
    """Provide the global application settings as a singleton."""
    return settings


def get_api_logger():
    """Provide the centralized logger."""
    return get_logger("research_rag.api")


# ---------------------------------------------------------------------------
# Infrastructure Singletons (Heavy Services)
# ---------------------------------------------------------------------------

@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    """Provide a singleton EmbeddingManager."""
    logger.debug("Initializing EmbeddingManager singleton...")
    try:
        return EmbeddingManager()
    except Exception as error:
        logger.error("Failed to initialize EmbeddingManager: %s", error)
        raise HTTPException(
            status_code=500, detail="Embedding service initialization failed."
        )


@lru_cache()
def get_embedding_model(
    manager: EmbeddingManager = Depends(get_embedding_manager)
) -> BaseEmbedding:
    """Provide the configured embedding model client."""
    try:
        return manager.get_provider()
    except Exception as error:
        logger.error("Failed to load embedding model: %s", error)
        raise HTTPException(
            status_code=500, detail="Failed to load embedding model."
        )


from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.base_vectorstore import BaseVectorStore

@lru_cache()
def get_vector_store() -> BaseVectorStore:
    """
    Provide a singleton Vector Store connection.
    
    Database connections are heavy and must be initialized exactly once
    and reused across all incoming API requests.
    """
    logger.debug("Initializing Vector Store singleton...")
    try:
        store = ChromaVectorStore()
        store.initialize()
        store.create_collection(settings.vector_store.collection_name)
        return store
    except Exception as error:
        logger.error("Failed to initialize Vector Store: %s", error)
        raise HTTPException(
            status_code=500, detail="Vector database connection failed."
        )

# For backward compatibility with existing injections
get_index_manager = get_vector_store


@lru_cache()
def get_llm_factory() -> LLMFactory:
    """Provide a singleton LLMFactory."""
    logger.debug("Initializing LLMFactory singleton...")
    try:
        return LLMFactory()
    except LLMConfigurationError as error:
        logger.error("LLM configuration error: %s", error)
        raise HTTPException(
            status_code=500, detail="LLM configuration is invalid."
        )
    except Exception as error:
        logger.error("Failed to initialize LLM Factory: %s", error)
        raise HTTPException(
            status_code=500, detail="LLM service initialization failed."
        )


@lru_cache()
def get_llm_client(factory: LLMFactory = Depends(get_llm_factory)) -> BaseLLM:
    """Provide the active LLM client instance."""
    try:
        return factory.get_client()
    except Exception as error:
        logger.error("Failed to retrieve LLM client: %s", error)
        raise HTTPException(
            status_code=500, detail="Failed to load language model client."
        )


@lru_cache()
def get_reranker() -> Optional[Reranker]:
    """Provide a singleton Reranker, if enabled in settings."""
    # Check if a reranker model is defined instead of a non-existent use_reranker flag
    if not settings.retrieval.reranker_model or settings.retrieval.reranker_model.lower() == "none":
        return None
        
    logger.debug("Initializing Reranker singleton...")
    try:
        return Reranker()
    except Exception as error:
        logger.error("Failed to initialize Reranker: %s", error)
        raise HTTPException(
            status_code=500, detail="Cross-encoder reranker initialization failed."
        )


# ---------------------------------------------------------------------------
# Request-Scoped Services (Lightweight Pipelines)
# ---------------------------------------------------------------------------

def get_retriever(
    vector_store: BaseVectorStore = Depends(get_index_manager),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
) -> Retriever:
    """Construct a Retriever instance using injected singletons."""
    return Retriever(vector_store=vector_store, embedding_manager=embedding_manager)


def get_retrieval_pipeline(
    retriever: Retriever = Depends(get_retriever),
    reranker: Optional[Reranker] = Depends(get_reranker),
) -> RetrievalPipeline:
    """Construct the RetrievalPipeline orchestrator."""
    return RetrievalPipeline(retriever=retriever, reranker=reranker)


def get_rag_pipeline(
    retriever: Retriever = Depends(get_retriever),
    llm: BaseLLM = Depends(get_llm_client),
    reranker: Optional[Reranker] = Depends(get_reranker),
) -> RAGPipeline:
    """
    Construct the RAGPipeline orchestrator.
    
    Pipelines are lightweight orchestration objects. They are safely created
    per-request, relying on the heavy, cached singletons (LLM, Database)
    passed in via dependency injection.
    """
    return RAGPipeline(retriever=retriever, llm=llm, reranker=reranker)


def get_indexing_pipeline(
    embedding_model: BaseEmbedding = Depends(get_embedding_model),
    vector_store: BaseVectorStore = Depends(get_index_manager),
) -> IndexingPipeline:
    """
    Construct the IndexingPipeline.
    
    Instantiates standard ETL components (Loaders, Splitters) on demand
    and injects the heavy singleton persistence and embedding models.
    """
    # Lightweight instances created per-request
    loader = PDFLoader()
    splitter = DocumentSplitter()
    
    return IndexingPipeline(
        loader=loader,
        splitter=splitter,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
