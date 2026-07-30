"""
Abstract base loader interface for ResearchRAG.

This module defines the contract that every document loader in the
project must implement, regardless of source format (PDF, plain text,
Markdown, DOCX, HTML, web pages, or video transcripts). Concrete
loaders inherit from `BaseLoader` and implement its abstract methods,
allowing the ingestion pipeline to work with any loader interchangeably
without knowledge of its internal implementation.

This module defines the interface only. No loading logic is implemented
here.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class UnsupportedSourceError(Exception):
    """
    Raised when a source provided to a loader is not supported.

    This includes sources with an unsupported file extension, sources
    that do not exist, or sources that otherwise fail validation before
    loading is attempted.
    """


class BaseLoader(ABC):
    """
    Abstract base class defining the common interface for all document
    loaders in ResearchRAG.

    Every loader (PDF, text, Markdown, DOCX, HTML, web, YouTube
    transcript, etc.) must inherit from this class and implement its
    abstract methods. This ensures that the ingestion pipeline can treat
    all loaders uniformly, regardless of source format, by depending
    only on this shared interface rather than on any concrete loader
    implementation.

    Subclasses are responsible for:
        - Declaring which file extensions or source types they support.
        - Validating that a given source is accessible and supported
          before attempting to load it.
        - Loading the source and returning its content as a list of
          LangChain `Document` objects.
    """

    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from the given source.

        Args:
            source: Path or identifier of the source to load. This may
                be a filesystem path, a URL, or another source
                identifier, depending on the concrete loader
                implementation.

        Returns:
            A list of LangChain `Document` objects extracted from the
            source. Each `Document` is expected to carry its textual
            content along with relevant source metadata (e.g., file
            name, page number, or origin URL).

        Raises:
            UnsupportedSourceError: If the source fails validation
                (see `validate_source`) prior to loading.
            IOError: If the source exists and is supported but cannot
                be read due to a filesystem, network, or parsing
                failure.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate whether the given source exists and is supported by
        this loader.

        Implementations should check that the source is accessible
        (e.g., the file exists, the URL is reachable) and that its
        format or extension is one of the values returned by
        `supported_extensions`.

        Args:
            source: Path or identifier of the source to validate.

        Returns:
            True if the source is valid and supported by this loader,
            False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Return the list of file extensions or source types supported
        by this loader.

        Returns:
            A list of supported extensions (e.g., [".pdf"]) or source
            identifiers (e.g., ["youtube.com"]) for non-file-based
            loaders. Extensions should be lowercase and include the
            leading dot for file-based loaders.
        """
        raise NotImplementedError