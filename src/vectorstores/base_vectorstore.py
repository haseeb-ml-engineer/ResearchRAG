"""
Abstract base vector store interface for ResearchRAG.

This module defines the contract that every vector database backend in
the project must implement, regardless of the underlying technology
(ChromaDB, FAISS, Pinecone, Qdrant, Weaviate, Milvus, or any future
provider). Concrete vector store classes inherit from
`BaseVectorStore` and implement its abstract methods, allowing the
rest of the system (index management, retrieval, the RAG pipeline) to
persist and query embedded documents without any knowledge of which
specific backend is configured.

This module defines the interface only. No database-specific logic is
implemented here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from src.config.logging_config import get_logger

logger = get_logger(__name__)

# An embedding vector may be represented as a NumPy array or a plain
# list of floats, depending on the caller's context.
EmbeddingVector = List[float]


class VectorStoreError(Exception):
    """
    Raised when a vector store operation fails.

    This includes failures such as an unreachable database, an invalid
    or missing collection, a malformed query, or a backend-specific
    error surfaced during a read or write operation.
    """


class CollectionNotFoundError(VectorStoreError):
    """Raised when an operation references a collection that does not exist."""


@dataclass(frozen=True)
class SimilaritySearchResult:
    """
    A single, provider-independent result from a similarity search.

    Every `BaseVectorStore` implementation returns search results in
    this shape, regardless of the underlying database, so that callers
    never need to know whether the backend is ChromaDB, Pinecone,
    FAISS, or any other provider.

    Attributes:
        document_id: Unique identifier of the matched document.
        document: The matched content and metadata as a LangChain
            `Document`.
        score: Similarity score of the match, where a higher value
            indicates greater similarity to the query.
    """

    document_id: str
    document: Document
    score: float


class BaseVectorStore(ABC):
    """
    Abstract base class defining the common interface for all vector
    database backends in ResearchRAG.

    Every vector store implementation (ChromaDB, FAISS, Pinecone,
    Qdrant, Weaviate, Milvus, etc.) must inherit from this class and
    implement its abstract methods. This ensures that index
    management, retrieval, and the RAG pipeline can all depend solely
    on this shared interface, remaining entirely agnostic to which
    concrete backend is configured. Switching vector databases requires
    only a configuration change, never a change to any dependent
    module.
    """

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Return the name of the active vector store implementation."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the connection or client required to interact with
        the underlying vector database.

        Implementations should perform any necessary setup here (e.g.,
        opening a client connection, establishing an on-disk index
        handle) so that subsequent operations can assume the store is
        ready for use.

        Raises:
            VectorStoreError: If the underlying database cannot be
                reached or initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Create a new collection (or equivalent index/namespace) in the
        vector database.

        Args:
            collection_name: Name of the collection to create.

        Raises:
            VectorStoreError: If the collection already exists and
                cannot be created, or if creation otherwise fails.
        """
        raise NotImplementedError

    @abstractmethod
    def load_collection(self, collection_name: str) -> None:
        """
        Load an existing collection, making it the active target for
        subsequent operations.

        Args:
            collection_name: Name of the collection to load.

        Raises:
            CollectionNotFoundError: If no collection with the given
                name exists.
            VectorStoreError: If the collection exists but cannot be
                loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[EmbeddingVector],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents and their corresponding embeddings to the active
        collection.

        Args:
            documents: List of LangChain `Document` objects to store,
                each carrying its textual content and metadata.
            embeddings: List of embedding vectors corresponding
                positionally to `documents`.
            ids: Optional list of unique identifiers corresponding
                positionally to `documents`. If not provided,
                implementations are expected to generate identifiers.

        Returns:
            The list of unique identifiers under which the documents
            were stored, in the same order as `documents`.

        Raises:
            VectorStoreError: If the number of documents, embeddings,
                and IDs (when provided) do not match, or if the write
                operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the active collection by their unique
        identifiers.

        Args:
            ids: List of unique identifiers of the documents to
                delete.

        Raises:
            VectorStoreError: If deletion fails for the given
                identifiers.
        """
        raise NotImplementedError

    @abstractmethod
    def update_documents(
        self,
        ids: List[str],
        documents: List[Document],
        embeddings: List[EmbeddingVector],
    ) -> None:
        """
        Update existing documents and their embeddings in the active
        collection.

        Args:
            ids: List of unique identifiers of the documents to
                update.
            documents: List of updated LangChain `Document` objects,
                corresponding positionally to `ids`.
            embeddings: List of updated embedding vectors,
                corresponding positionally to `ids`.

        Raises:
            VectorStoreError: If any of the given identifiers do not
                exist, or if the update operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SimilaritySearchResult]:
        """
        Retrieve the most similar documents to a query embedding.

        Args:
            query_embedding: Embedding vector representing the query.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters to constrain the search
                (e.g., restricting results to a specific source or
                document type).

        Returns:
            A list of `SimilaritySearchResult` objects ordered from
            most to least similar, of length at most `top_k`.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def count_documents(self) -> int:
        """
        Return the number of documents stored in the active
        collection.

        Returns:
            The total document count in the active collection.

        Raises:
            VectorStoreError: If the count cannot be determined.
        """
        raise NotImplementedError

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a collection with the given name exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.

        Raises:
            VectorStoreError: If existence cannot be determined due to
                a backend failure.
        """
        raise NotImplementedError

    def get_collection_statistics(self) -> Dict[str, int]:
        """Return basic statistics for the active collection when supported."""
        total_documents = self.count_documents()
        return {
            "total_documents": total_documents,
            "total_chunks": total_documents,
        }

    @abstractmethod
    def reset_collection(self) -> None:
        """
        Remove all documents from the active collection, leaving the
        collection itself in place but empty.

        Raises:
            VectorStoreError: If the reset operation fails.
        """
        raise NotImplementedError