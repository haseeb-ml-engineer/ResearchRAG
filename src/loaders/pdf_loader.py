"""
PDF document loader for ResearchRAG.

This module implements `PDFLoader`, a concrete `BaseLoader` responsible
solely for loading PDF files (individually or recursively from a
directory) and converting them into LangChain `Document` objects. It
performs no cleaning, chunking, embedding, or storage — those
responsibilities belong to their own dedicated modules further along
the ingestion pipeline.
"""

from pathlib import Path
import hashlib
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from src.config.logging_config import get_logger
from src.loaders.base_loader import BaseLoader, UnsupportedSourceError
from src.preprocessing.metadata import MetadataFields, MetadataManager

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = [".pdf"]


class PDFLoader(BaseLoader):
    """
    Loader responsible for extracting text and metadata from PDF files.

    Supports loading a single PDF file or recursively loading every PDF
    file within a directory. Each page of a PDF becomes a separate
    LangChain `Document`, enriched with source metadata such as file
    name, absolute path, file type, and page number.
    """

    def __init__(self) -> None:
        self._metadata_manager = MetadataManager()

    def supported_extensions(self) -> List[str]:
        """
        Return the file extensions supported by this loader.

        Returns:
            A list containing the supported extension(s).
        """
        return list(_SUPPORTED_EXTENSIONS)

    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate whether the given source is a valid, supported PDF
        source.

        A source is considered valid if it is either:
            - An existing file with a supported extension, or
            - An existing directory (validity of its contents is
              determined at load time).

        Args:
            source: Path to a PDF file or a directory containing PDF
                files.

        Returns:
            True if the source exists and is a supported type, False
            otherwise.
        """
        path = Path(source)

        if not path.exists():
            logger.error("Source does not exist: %s", path)
            return False

        if path.is_dir():
            return True

        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            return True

        logger.error("Unsupported file extension for source: %s", path)
        return False

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load one or more PDF files into LangChain `Document` objects.

        If `source` points to a single PDF file, that file is loaded.
        If `source` points to a directory, every PDF file within it
        (searched recursively) is loaded.

        Args:
            source: Path to a PDF file or a directory containing PDF
                files.

        Returns:
            A list of LangChain `Document` objects, one per page,
            enriched with source metadata.

        Raises:
            UnsupportedSourceError: If the source does not exist or is
                not a supported type.
            FileNotFoundError: If a directory source contains no PDF
                files.
            IOError: If a single PDF file cannot be parsed.
        """
        path = Path(source)
        logger.info("Starting PDF load for source: %s", path)

        if not self.validate_source(path):
            raise UnsupportedSourceError(f"Invalid or unsupported source: {path}")

        if path.is_dir():
            documents = self._load_directory(path)
        else:
            documents = self._load_single_pdf(path)

        logger.info("Completed PDF load for source: %s (%d pages)", path, len(documents))
        return documents

    def _load_directory(self, directory: Path) -> List[Document]:
        """
        Recursively load every PDF file found within a directory.

        Files that fail to load (e.g., corrupted PDFs) are logged and
        skipped so that a single bad file does not abort loading the
        rest of the directory.

        Args:
            directory: Directory to search for PDF files.

        Returns:
            A list of LangChain `Document` objects aggregated across
            all successfully loaded PDF files.

        Raises:
            FileNotFoundError: If no PDF files are found in the
                directory.
        """
        pdf_paths = sorted(directory.rglob("*.pdf"))

        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in directory: {directory}")

        documents: List[Document] = []
        for pdf_path in pdf_paths:
            try:
                documents.extend(self._load_single_pdf(pdf_path))
            except IOError as error:
                logger.warning("Skipping file due to load failure: %s (%s)", pdf_path, error)

        return documents

    def _load_single_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file and enrich its resulting documents with
        metadata.

        Args:
            file_path: Path to the PDF file to load.

        Returns:
            A list of LangChain `Document` objects, one per page.

        Raises:
            IOError: If the file cannot be parsed as a valid PDF.
        """
        try:
            loader = PyMuPDFLoader(str(file_path))
            documents = loader.load()
        except Exception as error:
            raise IOError(f"Failed to load PDF file '{file_path}': {error}") from error

        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        for document in documents:
            self._enrich_metadata(document, file_path, file_hash)

        logger.info("PDF loaded: %s", file_path)
        logger.info("Number of pages: %d", len(documents))
        return documents

    def _enrich_metadata(self, document: Document, file_path: Path, file_hash: str) -> None:
        """
        Attach standard source metadata to a document without
        overwriting metadata already populated by the underlying
        loader.

        Args:
            document: The LangChain `Document` to enrich in place.
            file_path: Path of the source PDF file the document was
                extracted from.
        """
        page_number = int(document.metadata.get("page", 0))
        enriched_metadata = self._metadata_manager.process(
            metadata=document.metadata,
            content=document.page_content,
            source=file_hash,
            file_type=file_path.suffix.lower(),
            page=page_number,
            include_checksum=True,
        )

        resolved_path = str(file_path.resolve())
        enriched_metadata[MetadataFields.SOURCE] = file_path.name
        enriched_metadata["filename"] = file_path.name
        enriched_metadata["page_number"] = page_number + 1
        enriched_metadata["source_hash"] = file_hash
        enriched_metadata["source_path"] = resolved_path
        enriched_metadata[MetadataFields.SOURCE_FILE] = file_path.name
        enriched_metadata[MetadataFields.ABSOLUTE_PATH] = resolved_path
        enriched_metadata["source_file_name"] = file_path.name
        enriched_metadata["source_file_path"] = resolved_path

        document.metadata = enriched_metadata