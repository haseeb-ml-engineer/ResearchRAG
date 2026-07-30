"""
Index manager for ResearchRAG.

This module defines `IndexManager`, responsible for orchestrating the
indexing pipeline: accepting documents, generating their embeddings
through an `EmbeddingManager`, and persisting them through a
`BaseVectorStore`. `IndexManager` contains no embedding logic and no
database logic of its own — it only coordinates the two, both of which
are received via dependency injection.
"""

import time
from typing import Any, Dict, List, Set

from langchain_core.documents import Document

from src.config.logging_config import get_logger
from src.embeddings.embedding_manager import EmbeddingManager
from src.preprocessing.metadata import MetadataFields
from src.vectorstores.base_vectorstore import BaseVectorStore

logger = get_logger(__name__)


class IndexManagerError(Exception):
    """Base exception for errors raised by `IndexManager`."""


class InvalidDocumentError(IndexManagerError):
    """Raised when a document fails basic structural validation."""


class MissingMetadataError(IndexManagerError):
    """Raised when a document is missing metadata required for indexing."""


class DuplicateDocumentError(IndexManagerError):
    """Raised when a batch of documents contains duplicate document IDs."""


class IndexManager:
    """
    Orchestrates the indexing pipeline by coordinating embedding
    generation and vector store persistence.

    `IndexManager` depends only on the `EmbeddingManager` and
    `BaseVectorStore` abstractions, both supplied through the
    constructor. It never generates embeddings itself and never speaks
    to a specific vector database directly, so swapping either
    dependency (a different embedding provider, a different vector
    database) requires no changes to this class.
    """

    def __init__(self, vector_store: BaseVectorStore, embedding_manager: EmbeddingManager) -> None:
        """
        Initialize the index manager with its collaborating
        components.

        Args:
            vector_store: A `BaseVectorStore` implementation used to
                persist and query document embeddings.
            embedding_manager: An `EmbeddingManager` used to generate
                embeddings for document text.
        """
        self._vector_store = vector_store
        self._embedding_manager = embedding_manager
        self._indexed_ids: Set[str] = set()

    def index_documents(
        self,
        documents: List[Document],
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Index a batch of documents: validate, embed, and persist them.

        Args:
            documents: List of LangChain `Document` objects to index.
                Each document's metadata must include a
                `document_id` field.
            skip_existing: If True, documents whose IDs have already
                been indexed in this manager's lifetime are skipped
                rather than re-indexed.

        Returns:
            A dictionary of indexing statistics: `documents_received`,
            `documents_indexed`, `documents_skipped`, and
            `duration_seconds`.

        Raises:
            InvalidDocumentError: If `documents` is empty or any
                document has empty content.
            MissingMetadataError: If any document is missing a
                `document_id` in its metadata.
            DuplicateDocumentError: If the batch contains duplicate
                document IDs.
        """
        logger.info("Indexing started for %d document(s)", len(documents))
        start_time = time.perf_counter()

        self._validate_documents(documents)

        documents_to_index = (
            self._filter_existing_documents(documents) if skip_existing else documents
        )
        skipped_count = len(documents) - len(documents_to_index)
        stored_count = 0

        if documents_to_index:
            stored_ids = self._embed_and_store(documents_to_index)
            stored_count = len(stored_ids)

        duration_seconds = time.perf_counter() - start_time
        statistics = {
            "documents_received": len(documents),
            "documents_indexed": stored_count,
            "documents_skipped": len(documents_to_index) - stored_count + skipped_count,
            "duration_seconds": duration_seconds,
        }

        logger.info(
            "Indexing completed: received=%d indexed=%d skipped=%d duration=%.3fs",
            statistics["documents_received"],
            statistics["documents_indexed"],
            statistics["documents_skipped"],
            statistics["duration_seconds"],
        )

        return statistics

    def index_document(self, document: Document) -> str:
        """
        Index a single document.

        Args:
            document: The LangChain `Document` to index. Its metadata
                must include a `document_id` field.

        Returns:
            The document ID under which the document was indexed.

        Raises:
            InvalidDocumentError: If the document has empty content.
            MissingMetadataError: If the document is missing a
                `document_id` in its metadata.
        """
        self.index_documents([document], skip_existing=False)
        return str(document.metadata[MetadataFields.DOCUMENT_ID])

    def rebuild_index(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Rebuild the index from scratch using the given documents.

        Clears all existing documents from the vector store and then
        indexes the provided documents as a fresh batch.

        Args:
            documents: List of LangChain `Document` objects to index
                after clearing the existing index.

        Returns:
            The indexing statistics produced by `index_documents` for
            the rebuilt index.

        Raises:
            InvalidDocumentError: If `documents` is empty or any
                document has empty content.
            MissingMetadataError: If any document is missing a
                `document_id` in its metadata.
            DuplicateDocumentError: If the batch contains duplicate
                document IDs.
        """
        logger.info("Rebuilding index with %d document(s)", len(documents))
        self.clear_index()
        return self.index_documents(documents, skip_existing=False)

    def clear_index(self) -> None:
        """
        Remove all documents from the vector store and reset local
        indexing state.

        Raises:
            IndexManagerError: If the underlying vector store fails to
                clear its collection.
        """
        logger.info("Clearing index")
        try:
            self._vector_store.reset_collection()
        except Exception as error:
            raise IndexManagerError(f"Failed to clear index: {error}") from error

        self._indexed_ids.clear()
        logger.info("Index cleared successfully")

    def document_exists(self, document_id: str) -> bool:
        """
        Check whether a document has already been indexed by this
        manager.

        Args:
            document_id: Unique identifier of the document to check.

        Returns:
            True if the document ID has been indexed during this
            manager's lifetime, False otherwise.
        """
        return document_id in self._indexed_ids

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Retrieve summary statistics about the current index.

        Returns:
            A dictionary containing the total number of documents
            stored in the vector store, the number of documents
            indexed during this manager's lifetime, the active
            embedding provider name, and its embedding dimension.

        Raises:
            IndexManagerError: If the underlying vector store or
                embedding manager fails to report its statistics.
        """
        try:
            total_documents = self._vector_store.count_documents()
        except Exception as error:
            raise IndexManagerError(f"Failed to retrieve document count: {error}") from error

        try:
            embedding_dimension = self._embedding_manager.embedding_dimension()
        except Exception as error:
            raise IndexManagerError(f"Failed to retrieve embedding dimension: {error}") from error

        return {
            "total_documents_in_store": total_documents,
            "documents_indexed_this_session": len(self._indexed_ids),
            "embedding_provider": self._embedding_manager.get_provider_name(),
            "embedding_dimension": embedding_dimension,
        }

    def _embed_and_store(self, documents: List[Document]) -> List[str]:
        """
        Generate embeddings for the given documents and persist them
        to the vector store.

        Args:
            documents: List of LangChain `Document` objects to embed
                and store.

        Raises:
            IndexManagerError: If embedding generation or storage
                fails.
        """
        texts = [document.page_content for document in documents]
        ids = [str(document.metadata[MetadataFields.DOCUMENT_ID]) for document in documents]

        try:
            embeddings = self._embedding_manager.embed_documents(texts)
        except Exception as error:
            raise IndexManagerError(f"Embedding generation failed: {error}") from error

        logger.info("Embedding generation completed for %d document(s)", len(documents))

        try:
            stored_ids = self._vector_store.add_documents(documents=documents, embeddings=embeddings, ids=ids)
        except Exception as error:
            raise IndexManagerError(f"Failed to store documents in vector store: {error}") from error

        self._indexed_ids.update(stored_ids)
        return stored_ids

    def _filter_existing_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter out documents whose IDs have already been indexed.

        Args:
            documents: List of LangChain `Document` objects to filter.

        Returns:
            The subset of `documents` whose document IDs have not yet
            been indexed by this manager.
        """
        return [
            document
            for document in documents
            if str(document.metadata[MetadataFields.DOCUMENT_ID]) not in self._indexed_ids
        ]

    @staticmethod
    def _validate_documents(documents: List[Document]) -> None:
        """
        Validate a batch of documents prior to indexing.

        Args:
            documents: List of LangChain `Document` objects to
                validate.

        Raises:
            InvalidDocumentError: If `documents` is empty or any
                document has empty content.
            MissingMetadataError: If any document is missing a
                `document_id` in its metadata.
            DuplicateDocumentError: If the batch contains duplicate
                document IDs.
        """
        if not documents:
            raise InvalidDocumentError("Cannot index an empty list of documents.")

        document_ids: List[str] = []
        for document in documents:
            if not isinstance(document, Document) or not document.page_content.strip():
                raise InvalidDocumentError(f"Invalid or empty document content: {document!r}")

            document_id = document.metadata.get(MetadataFields.DOCUMENT_ID)
            if not document_id:
                raise MissingMetadataError(
                    f"Document is missing required '{MetadataFields.DOCUMENT_ID}' metadata: "
                    f"{document.metadata}"
                )
            document_ids.append(str(document_id))

        duplicates = {doc_id for doc_id in document_ids if document_ids.count(doc_id) > 1}
        if duplicates:
            raise DuplicateDocumentError(f"Batch contains duplicate document IDs: {duplicates}")