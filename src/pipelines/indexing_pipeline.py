"""
Indexing pipeline orchestrator for ResearchRAG.

This module defines `IndexingPipeline`, the primary orchestration layer
for ingesting raw documents into the vector database. It coordinates
loading, metadata enrichment, cleaning, chunking, embedding generation,
and storage, without implementing any of the underlying algorithms itself.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.documents import Document as LangChainDocument

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class IndexingPipelineError(Exception):
    """Base exception for errors raised during the indexing pipeline."""


Document = LangChainDocument


@dataclass(frozen=True)
class IndexingReport:
    """
    Structured report containing statistics from the indexing run.
    """
    indexed_documents: int
    indexed_chunks: int
    failed_documents: int
    elapsed_time_seconds: float
    embedding_model: str
    vector_store: str
    indexing_statistics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dependency Protocols (Dependency Inversion Principle)
# ---------------------------------------------------------------------------

class LoaderProtocol(Protocol):
    def load(self, source: str | Path) -> List[Document]:
        """Load documents from a source."""
        ...

class MetadataProcessorProtocol(Protocol):
    def process(self, documents: List[Document]) -> List[Document]:
        """Enrich document metadata."""
        ...

class CleanerProtocol(Protocol):
    def clean(self, documents: List[Document]) -> List[Document]:
        """Clean document text content."""
        ...

class SplitterProtocol(Protocol):
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        ...

class EmbeddingModelProtocol(Protocol):
    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...

class VectorStoreProtocol(Protocol):
    @property
    def store_name(self) -> str:
        """Return the name of the vector store."""
        ...

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> List[str]:
        """Store documents and their corresponding embeddings."""
        ...


class IndexingPipeline:
    """
    Orchestrates the complete document ingestion and indexing workflow.
    
    This class coordinates the extraction, transformation, and loading (ETL)
    of documents into the vector database. It relies entirely on injected
    dependencies to perform the actual work, ensuring the pipeline itself
    remains decoupled from specific file formats, chunking algorithms, or
    database technologies.
    """

    def __init__(
        self,
        loader: LoaderProtocol,
        splitter: SplitterProtocol,
        embedding_model: EmbeddingModelProtocol,
        vector_store: VectorStoreProtocol,
        metadata_processor: Optional[MetadataProcessorProtocol] = None,
        cleaner: Optional[CleanerProtocol] = None,
    ) -> None:
        """
        Initialize the indexing pipeline with required services.

        Args:
            loader: Service to extract documents from a source.
            splitter: Service to divide documents into chunks.
            embedding_model: Service to generate vector embeddings.
            vector_store: Service to persist chunks and vectors.
            metadata_processor: Optional service to enrich metadata.
            cleaner: Optional service to sanitize document text.
        """
        self._loader = loader
        self._splitter = splitter
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._metadata_processor = metadata_processor
        self._cleaner = cleaner

        store_name = getattr(self._vector_store, 'store_name', self._vector_store.__class__.__name__)
        logger.info(
            "IndexingPipeline initialized (loader=%s, store=%s)",
            self._loader.__class__.__name__,
            store_name,
        )

    def run(self, source: str) -> IndexingReport:
        """
        Execute the full document indexing workflow.

        Args:
            source: The URI, path, or identifier of the documents to load.

        Returns:
            An `IndexingReport` containing the results and statistics
            of the indexing run.

        Raises:
            IndexingPipelineError: If a critical pipeline stage fails.
            ValueError: If the source is invalid.
        """
        if not source or not isinstance(source, str):
            raise ValueError("Source must be a valid, non-empty string.")

        logger.info("Starting indexing pipeline for source: %r", source)
        start_time = time.perf_counter()
        
        # 1. Load Documents
        try:
            documents = self._loader.load(source)
            logger.info("Loaded %d documents from source.", len(documents))
        except Exception as error:
            logger.error("Failed to load documents from %r: %s", source, error)
            raise IndexingPipelineError(f"Document loading failed: {error}") from error

        if not documents:
            logger.warning("No documents found at source: %r", source)
            return self._create_empty_report(start_time)

        # 2. Metadata Processing (Optional)
        if self._metadata_processor:
            try:
                documents = self._metadata_processor.process(documents)
                logger.info("Metadata processing completed.")
            except Exception as error:
                logger.error("Metadata processing failed: %s", error)
                raise IndexingPipelineError(f"Metadata processing failed: {error}") from error

        # 3. Document Cleaning (Optional)
        if self._cleaner:
            try:
                documents = self._cleaner.clean(documents)
                logger.info("Document cleaning completed.")
            except Exception as error:
                logger.error("Document cleaning failed: %s", error)
                raise IndexingPipelineError(f"Document cleaning failed: {error}") from error

        # 4. Chunking
        try:
            chunks = self._splitter.split_documents(documents)
            logger.info("Number of chunks: %d", len(chunks))
        except Exception as error:
            logger.error("Document splitting failed: %s", error)
            raise IndexingPipelineError(f"Document splitting failed: {error}") from error

        if not chunks:
            logger.warning("Chunking resulted in 0 chunks. Aborting indexing.")
            return self._create_empty_report(start_time)

        # 5. Embedding Generation
        try:
            logger.info("Generating embeddings for %d chunks...", len(chunks))
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self._embedding_model.embed_documents(texts)
            logger.info("Embeddings generated: %d", len(embeddings))
            
            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count ({len(embeddings)}) does not match "
                    f"chunk count ({len(chunks)})."
                )
        except Exception as error:
            logger.error("Embedding generation failed: %s", error)
            raise IndexingPipelineError(f"Embedding generation failed: {error}") from error

        # 6. Vector Store Persistence
        try:
            logger.info("Storing %d chunks in vector database...", len(chunks))
            self._vector_store.add_documents(chunks, embeddings)
        except Exception as error:
            logger.error("Vector store persistence failed: %s", error)
            raise IndexingPipelineError(f"Vector storage failed: {error}") from error

        # 7. Collect Statistics
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Indexing pipeline completed successfully in %.2f seconds. "
            "Indexed %d documents (%d chunks).",
            elapsed_time,
            len(documents),
            len(chunks),
        )

        model_name = getattr(self._embedding_model, "model_name", "Unknown")
        store_name = getattr(self._vector_store, "store_name", "Unknown")

        return IndexingReport(
            indexed_documents=len(documents),
            indexed_chunks=len(chunks),
            failed_documents=0,
            elapsed_time_seconds=round(elapsed_time, 4),
            embedding_model=model_name,
            vector_store=store_name,
            indexing_statistics={
                "average_chunk_length": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
                "source": source,
            },
        )

    def _create_empty_report(self, start_time: float) -> IndexingReport:
        """Helper to return a zeroed report when pipeline aborts early."""
        model_name = getattr(self._embedding_model, "model_name", "Unknown")
        store_name = getattr(self._vector_store, "store_name", "Unknown")
        return IndexingReport(
            indexed_documents=0,
            indexed_chunks=0,
            failed_documents=0,
            elapsed_time_seconds=round(time.perf_counter() - start_time, 4),
            embedding_model=model_name,
            vector_store=store_name,
        )
