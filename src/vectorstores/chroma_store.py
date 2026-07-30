"""
ChromaDB vector store implementation for ResearchRAG.

This module implements `ChromaVectorStore`, a concrete
`BaseVectorStore` that persists document embeddings using ChromaDB. It
encapsulates all ChromaDB-specific details (client initialization,
collection management, query syntax, distance-to-score conversion),
exposing only the provider-agnostic interface defined by
`BaseVectorStore` to the rest of the system.
"""

import uuid
from typing import Any, Dict, List, Optional, cast

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.vectorstores.base_vectorstore import (
    BaseVectorStore,
    CollectionNotFoundError,
    EmbeddingVector,
    SimilaritySearchResult,
    VectorStoreError,
)

logger = get_logger(__name__)

# Default distance metric used when none is configured. ChromaDB
# supports "cosine", "l2", and "ip" (inner product).
_DEFAULT_DISTANCE_METRIC = "cosine"

# Default batch size used for insert operations when none is
# configured elsewhere in application settings.
_DEFAULT_BATCH_SIZE = 100


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store implementation backed by a persistent ChromaDB
    client.

    All ChromaDB-specific behavior — client initialization, collection
    management, insert/query syntax, and distance-to-similarity
    conversion — is contained within this class. Callers interact only
    with the `BaseVectorStore` interface and the provider-independent
    `SimilaritySearchResult` type, remaining unaware that ChromaDB is
    the underlying backend.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        distance_metric: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the ChromaDB vector store configuration.

        The underlying client and collection are not created until
        `initialize()` and `create_collection()` / `load_collection()`
        are called.

        Args:
            collection_name: Name of the ChromaDB collection to use.
                Defaults to the value configured in application
                settings.
            persist_directory: Filesystem directory where ChromaDB
                persists its data. Defaults to the value configured in
                application settings.
            distance_metric: Distance metric used for similarity
                search (e.g., "cosine", "l2", "ip"). Defaults to
                `_DEFAULT_DISTANCE_METRIC`.
            batch_size: Maximum number of documents inserted per batch
                operation. Defaults to `_DEFAULT_BATCH_SIZE`.
        """
        self._collection_name = collection_name or settings.vector_store.collection_name
        self._persist_directory = persist_directory or str(settings.paths.vector_db_dir)
        self._distance_metric = distance_metric or getattr(
            settings.vector_store, "distance_metric", _DEFAULT_DISTANCE_METRIC
        )
        self._batch_size = batch_size or getattr(
            settings.vector_store, "batch_size", _DEFAULT_BATCH_SIZE
        )

        self._client: Optional[Any] = None
        self._collection: Optional[Collection] = None

    @property
    def store_name(self) -> str:
        return "chromadb"

    def initialize(self) -> None:
        """
        Initialize the persistent ChromaDB client.

        Raises:
            VectorStoreError: If the client fails to initialize, for
                example due to an inaccessible persist directory.
        """
        logger.info(
            "Initializing ChromaDB client at persist directory: %s", self._persist_directory
        )
        try:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        except Exception as error:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {error}") from error

        logger.info("ChromaDB client initialized successfully")

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new ChromaDB collection and set it as active.

        If a collection with the given name already exists, it is
        loaded rather than recreated, since ChromaDB treats collection
        creation as idempotent by name.

        Args:
            collection_name: Name of the collection to create.

        Raises:
            VectorStoreError: If the client has not been initialized
                or collection creation fails.
        """
        self._ensure_client_initialized()
        assert self._client is not None

        logger.info("Creating ChromaDB collection: %s", collection_name)
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self._distance_metric},
            )
        except Exception as error:
            raise VectorStoreError(
                f"Failed to create collection '{collection_name}': {error}"
            ) from error

        self._collection_name = collection_name
        logger.info("Collection ready: %s", collection_name)

    def load_collection(self, collection_name: str) -> None:
        """
        Load an existing ChromaDB collection and set it as active.

        Args:
            collection_name: Name of the collection to load.

        Raises:
            CollectionNotFoundError: If no collection with the given
                name exists.
            VectorStoreError: If the client has not been initialized
                or the collection cannot be loaded.
        """
        self._ensure_client_initialized()
        assert self._client is not None

        if not self.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection does not exist: {collection_name}")

        logger.info("Loading ChromaDB collection: %s", collection_name)
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception as error:
            raise VectorStoreError(
                f"Failed to load collection '{collection_name}': {error}"
            ) from error

        self._collection_name = collection_name
        logger.info("Collection loaded: %s", collection_name)

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[EmbeddingVector],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents and their embeddings to the active collection in
        batches.

        Args:
            documents: List of LangChain `Document` objects to store.
            embeddings: List of embedding vectors corresponding
                positionally to `documents`.
            ids: Optional list of unique identifiers corresponding
                positionally to `documents`. If not provided,
                identifiers are generated automatically.

        Returns:
            The list of unique identifiers under which the documents
            were stored, in the same order as `documents`.

        Raises:
            VectorStoreError: If no collection is active, if input
                lengths are mismatched, if embeddings are malformed, or
                if the insert operation fails.
        """
        self._ensure_collection_loaded()
        assert self._collection is not None

        resolved_ids = ids or [self._generate_document_id() for _ in documents]
        self._validate_add_inputs(documents, embeddings, resolved_ids)

        existing_ids = self._get_existing_ids(resolved_ids)
        if existing_ids:
            filtered = [
                (document, embedding, document_id)
                for document, embedding, document_id in zip(documents, embeddings, resolved_ids)
                if document_id not in existing_ids
            ]
            if not filtered:
                logger.info("Skipping insertion because all %d document(s) already exist in collection: %s", len(documents), self._collection_name)
                return []
            documents = [item[0] for item in filtered]
            embeddings = [item[1] for item in filtered]
            resolved_ids = [item[2] for item in filtered]

        logger.info("Adding %d document(s) to collection: %s", len(documents), self._collection_name)

        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata) for document in documents]

        for batch_start in range(0, len(documents), self._batch_size):
            batch_end = batch_start + self._batch_size
            try:
                self._collection.add(
                    ids=resolved_ids[batch_start:batch_end],
                    embeddings=cast(Any, self._as_plain_lists(embeddings[batch_start:batch_end])),
                    documents=texts[batch_start:batch_end],
                    metadatas=cast(Any, metadatas[batch_start:batch_end]),
                )
            except Exception as error:
                raise VectorStoreError(f"Failed to add documents to collection: {error}") from error

        logger.info("Number of vectors inserted into ChromaDB: %d", len(documents))
        logger.info("Collection size after insertion: %d", self.count_documents())
        return resolved_ids

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the active collection by their unique
        identifiers.

        Args:
            ids: List of unique identifiers of the documents to
                delete.

        Raises:
            VectorStoreError: If no collection is active, `ids` is
                empty, or the delete operation fails.
        """
        self._ensure_collection_loaded()
        assert self._collection is not None

        if not ids:
            raise VectorStoreError("ids must not be empty for delete_documents.")

        logger.info("Deleting %d document(s) from collection: %s", len(ids), self._collection_name)
        try:
            self._collection.delete(ids=ids)
        except Exception as error:
            raise VectorStoreError(f"Failed to delete documents: {error}") from error

        logger.info("Successfully deleted %d document(s)", len(ids))

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
            VectorStoreError: If no collection is active, input
                lengths are mismatched, or the update operation fails.
        """
        self._ensure_collection_loaded()
        assert self._collection is not None
        self._validate_add_inputs(documents, embeddings, ids)

        logger.info("Updating %d document(s) in collection: %s", len(ids), self._collection_name)

        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata) for document in documents]

        try:
            self._collection.update(
                ids=ids,
                embeddings=cast(Any, self._as_plain_lists(embeddings)),
                documents=texts,
                metadatas=cast(Any, metadatas),
            )
        except Exception as error:
            raise VectorStoreError(f"Failed to update documents: {error}") from error

        logger.info("Successfully updated %d document(s)", len(ids))

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
            filters: Optional ChromaDB "where" metadata filter used to
                constrain the search.

        Returns:
            A list of `SimilaritySearchResult` objects ordered from
            most to least similar.

        Raises:
            VectorStoreError: If no collection is active, `top_k` is
                not positive, or the query fails.
        """
        self._ensure_collection_loaded()
        assert self._collection is not None

        if top_k <= 0:
            raise VectorStoreError(f"top_k must be greater than 0, got {top_k}")

        logger.info(
            "Running similarity search on collection: %s (top_k=%d)",
            self._collection_name,
            top_k,
        )

        try:
            raw_results = self._collection.query(
                query_embeddings=cast(Any, [self._as_plain_list(query_embedding)]),
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise VectorStoreError(f"Similarity search failed: {error}") from error

        results = self._parse_query_results(cast(Any, raw_results))
        logger.info("Similarity search returned %d result(s)", len(results))
        return results

    def count_documents(self) -> int:
        """
        Return the number of documents stored in the active
        collection.

        Returns:
            The total document count in the active collection.

        Raises:
            VectorStoreError: If no collection is active or the count
                cannot be determined.
        """
        self._ensure_collection_loaded()
        assert self._collection is not None
        try:
            return self._collection.count()
        except Exception as error:
            raise VectorStoreError(f"Failed to count documents: {error}") from error

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a ChromaDB collection with the given name
        exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.

        Raises:
            VectorStoreError: If the client has not been initialized
                or existence cannot be determined.
        """
        self._ensure_client_initialized()
        assert self._client is not None
        try:
            existing_names = {collection.name for collection in self._client.list_collections()}
        except Exception as error:
            raise VectorStoreError(f"Failed to check collection existence: {error}") from error

        return collection_name in existing_names

    def reset_collection(self) -> None:
        """
        Remove all documents from the active collection by deleting
        and recreating it.

        Raises:
            VectorStoreError: If no collection is active or the reset
                operation fails.
        """
        self._ensure_client_initialized()
        self._ensure_collection_loaded()
        assert self._client is not None

        logger.info("Resetting collection: %s", self._collection_name)
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as error:
            raise VectorStoreError(
                f"Failed to reset collection '{self._collection_name}': {error}"
            ) from error

        self.create_collection(self._collection_name)
        logger.info("Collection reset successfully: %s", self._collection_name)

    def _ensure_client_initialized(self) -> None:
        """
        Ensure the ChromaDB client has been initialized.

        Raises:
            VectorStoreError: If `initialize()` has not been called.
        """
        if self._client is None:
            raise VectorStoreError("ChromaDB client is not initialized. Call initialize() first.")

    def _ensure_collection_loaded(self) -> None:
        """
        Ensure an active collection has been created or loaded.

        Raises:
            VectorStoreError: If no collection is currently active.
        """
        self._ensure_client_initialized()
        if self._collection is None:
            raise VectorStoreError(
                "No active collection. Call create_collection() or load_collection() first."
            )

    @staticmethod
    def _validate_add_inputs(
        documents: List[Document],
        embeddings: List[EmbeddingVector],
        ids: List[str],
    ) -> None:
        """
        Validate that documents, embeddings, and identifiers are
        non-empty, consistently sized, and well-formed.

        Args:
            documents: List of LangChain `Document` objects.
            embeddings: List of embedding vectors.
            ids: List of unique identifiers.

        Raises:
            VectorStoreError: If any input is empty, lengths are
                mismatched, identifiers are duplicated, or embeddings
                have inconsistent dimensionality.
        """
        if not documents:
            raise VectorStoreError("documents must not be empty.")

        if not (len(documents) == len(embeddings) == len(ids)):
            raise VectorStoreError(
                "documents, embeddings, and ids must be the same length: "
                f"{len(documents)}, {len(embeddings)}, {len(ids)}"
            )

        if len(set(ids)) != len(ids):
            raise VectorStoreError("ids must not contain duplicate values.")

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) > 1:
            raise VectorStoreError(f"Embeddings have inconsistent dimensions: {dimensions}")

    @staticmethod
    def _as_plain_list(embedding: EmbeddingVector) -> List[float]:
        """
        Convert a single embedding vector to a plain list of floats.

        Args:
            embedding: An embedding vector as a list or NumPy array.

        Returns:
            The embedding vector as a plain list of floats.
        """
        return list(embedding)

    @classmethod
    def _as_plain_lists(cls, embeddings: List[EmbeddingVector]) -> List[List[float]]:
        """
        Convert a batch of embedding vectors to plain lists of floats.

        Args:
            embeddings: List of embedding vectors as lists or NumPy
                arrays.

        Returns:
            The embedding vectors as plain lists of floats.
        """
        return [cls._as_plain_list(embedding) for embedding in embeddings]

    @staticmethod
    def _generate_document_id() -> str:
        """
        Generate a unique identifier for a document without a supplied
        ID.

        Returns:
            A unique identifier string.
        """
        return str(uuid.uuid4())

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """
        Convert a ChromaDB distance value into a similarity score.

        Args:
            distance: Raw distance value returned by ChromaDB, where
                smaller values indicate greater similarity.

        Returns:
            A similarity score where higher values indicate greater
            similarity.
        """
        return 1.0 - distance

    def _get_existing_ids(self, ids: List[str]) -> set[str]:
        self._ensure_collection_loaded()
        assert self._collection is not None
        try:
            existing = self._collection.get(ids=ids, include=[])
        except Exception:
            return set()

        existing_ids = existing.get("ids", [])
        if existing_ids and isinstance(existing_ids[0], list):
            return set(existing_ids[0])
        return set(existing_ids)

    @classmethod
    def _parse_query_results(cls, raw_results: Dict[str, Any]) -> List[SimilaritySearchResult]:
        """
        Convert raw ChromaDB query output into provider-independent
        search results.

        Args:
            raw_results: The dictionary returned by ChromaDB's
                `Collection.query`.

        Returns:
            A list of `SimilaritySearchResult` objects.
        """
        ids = raw_results.get("ids", [[]])[0]
        texts = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: List[SimilaritySearchResult] = []
        for document_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            document = Document(page_content=text, metadata=dict(metadata or {}))
            results.append(
                SimilaritySearchResult(
                    document_id=document_id,
                    document=document,
                    score=cls._distance_to_score(distance),
                )
            )

        return results