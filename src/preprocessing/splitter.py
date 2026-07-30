"""
Document splitting for ResearchRAG.

This module defines `DocumentSplitter`, responsible for splitting
LangChain `Document` objects into smaller, retrieval-sized chunks. It
performs text splitting only; it does not load documents, generate or
validate metadata beyond chunk-level bookkeeping, generate embeddings,
or interact with a vector store or LLM.
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

# Default separators used when none are explicitly provided, ordered
# from largest to smallest semantic boundary.
_DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


class InvalidSplitterConfigurationError(Exception):
    """Raised when the splitter is configured with invalid parameters."""


class EmptyDocumentListError(Exception):
    """Raised when splitting is attempted on an empty document list."""


class DocumentSplitter:
    """
    Splits LangChain `Document` objects into smaller chunks suitable
    for embedding and retrieval.

    `DocumentSplitter` wraps a configurable `TextSplitter` (defaulting
    to `RecursiveCharacterTextSplitter`) and enriches every resulting
    chunk with chunk-level metadata (chunk ID, chunk index, total chunk
    count, and chunk size) while preserving all metadata carried by the
    parent document.

    The underlying text splitter is injected rather than hardcoded,
    allowing future chunking strategies (e.g., semantic or
    structure-aware splitting) to be introduced by supplying a
    different `TextSplitter` implementation, without modifying this
    class.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        length_function: Optional[Callable[[str], int]] = None,
        text_splitter: Optional[TextSplitter] = None,
    ) -> None:
        """
        Initialize the document splitter.

        Args:
            chunk_size: Maximum number of characters per chunk. Defaults
                to the value configured in application settings.
            chunk_overlap: Number of overlapping characters between
                consecutive chunks. Defaults to the value configured in
                application settings.
            separators: Ordered list of separators used to split text,
                from largest to smallest semantic boundary. Defaults to
                a sensible general-purpose set.
            length_function: Function used to measure text length when
                splitting. Defaults to `len`.
            text_splitter: A pre-configured `TextSplitter` instance to
                use instead of the default
                `RecursiveCharacterTextSplitter`. Supplying this allows
                alternate chunking strategies to be plugged in without
                modifying this class.

        Raises:
            InvalidSplitterConfigurationError: If `chunk_size` is not
                positive or `chunk_overlap` is not smaller than
                `chunk_size`.
        """
        self._chunk_size = chunk_size or settings.chunking.chunk_size
        self._chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunking.chunk_overlap
        )
        self._separators = separators or list(_DEFAULT_SEPARATORS)
        self._length_function = length_function or len

        self._validate_configuration(self._chunk_size, self._chunk_overlap)

        self._text_splitter: TextSplitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
            length_function=self._length_function,
        )

    @staticmethod
    def _validate_configuration(chunk_size: int, chunk_overlap: int) -> None:
        """
        Validate chunk size and overlap configuration.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between
                consecutive chunks.

        Raises:
            InvalidSplitterConfigurationError: If `chunk_size` is not
                positive or `chunk_overlap` is not smaller than
                `chunk_size`.
        """
        if chunk_size <= 0:
            raise InvalidSplitterConfigurationError(
                f"chunk_size must be greater than 0, got {chunk_size}"
            )
        if chunk_overlap < 0:
            raise InvalidSplitterConfigurationError(
                f"chunk_overlap must be non-negative, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise InvalidSplitterConfigurationError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})"
            )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunked `Document` objects.

        Args:
            documents: List of LangChain `Document` objects to split.

        Returns:
            A list of chunked `Document` objects, each carrying the
            original document's metadata plus chunk-level metadata.

        Raises:
            EmptyDocumentListError: If `documents` is empty.
        """
        if not documents:
            raise EmptyDocumentListError("Cannot split an empty list of documents.")

        logger.info("Splitting %d document(s) into chunks", len(documents))
        start_time = time.perf_counter()

        chunks: List[Document] = []
        for document in documents:
            chunks.extend(self.split_single_document(document))

        elapsed_seconds = time.perf_counter() - start_time
        statistics = self.get_statistics(chunks)

        logger.info(
            "Produced %d chunk(s) from %d document(s) in %.3fs "
            "(average chunk size: %.1f characters)",
            statistics["total_chunks"],
            len(documents),
            elapsed_seconds,
            statistics["average_chunk_size"],
        )

        return chunks

    def split_single_document(self, document: Document) -> List[Document]:
        """
        Split a single document into chunked `Document` objects.

        Args:
            document: The LangChain `Document` to split.

        Returns:
            A list of chunked `Document` objects derived from the given
            document, each preserving the parent document's metadata
            and enriched with chunk-level metadata.
        """
        raw_chunks = self._text_splitter.split_text(document.page_content)
        total_chunks = len(raw_chunks)

        parent_id = str(document.metadata.get("document_id", ""))

        chunked_documents: List[Document] = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_metadata = dict(document.metadata)
            chunk_metadata.update(
                {
                    "chunk_id": self._generate_chunk_id(parent_id, index, chunk_text),
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "chunk_size": self._length_function(chunk_text),
                }
            )
            chunked_documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunked_documents

    @staticmethod
    def _generate_chunk_id(parent_id: str, index: int, content: str) -> str:
        """
        Generate a deterministic, unique identifier for a chunk.

        The identifier is derived from the parent document's ID, the
        chunk's position, and its content, so identical input always
        produces the same chunk ID across runs.

        Args:
            parent_id: Identifier of the parent document.
            index: Position of the chunk within its parent document.
            content: Textual content of the chunk.

        Returns:
            A hexadecimal string uniquely identifying the chunk.
        """
        hasher = hashlib.sha256()
        hasher.update(parent_id.encode("utf-8"))
        hasher.update(str(index).encode("utf-8"))
        hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()

    def get_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Compute summary statistics for a list of chunked documents.

        Args:
            chunks: List of chunked `Document` objects.

        Returns:
            A dictionary containing the total chunk count, average
            chunk size, minimum chunk size, and maximum chunk size.
            All size values are zero when `chunks` is empty.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0.0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [self._length_function(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }