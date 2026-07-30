"""
Metadata management for ResearchRAG.

This module defines `MetadataManager`, responsible for standardizing,
enriching, validating, and merging document metadata before documents
enter the chunking stage of the ingestion pipeline. It operates purely
on metadata dictionaries and document content strings; it performs no
document loading, chunking, embedding, or storage.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class InvalidMetadataError(Exception):
    """Raised when document metadata fails required-field validation."""


class MetadataFields:
    """
    Canonical metadata field names used throughout ResearchRAG.

    Centralizing field names as constants avoids hardcoded strings
    scattered across the codebase and gives a single place to add new
    standard fields as the project grows.
    """

    DOCUMENT_ID = "document_id"
    SOURCE = "source"
    SOURCE_FILE = "source_file"
    ABSOLUTE_PATH = "absolute_path"
    FILE_TYPE = "file_type"
    PAGE = "page"
    CREATED_AT = "created_at"
    CONTENT_LENGTH = "content_length"
    CHECKSUM = "checksum"
    LANGUAGE = "language"


# Fields that every document's metadata must contain after enrichment.
_REQUIRED_FIELDS: List[str] = [
    MetadataFields.DOCUMENT_ID,
    MetadataFields.SOURCE,
    MetadataFields.FILE_TYPE,
]

# Maps known aliases produced by different loaders to the canonical
# field name expected downstream. New aliases can be added here without
# touching any other logic in this module.
_FIELD_ALIASES: Dict[str, str] = {
    "source_file_name": MetadataFields.SOURCE_FILE,
    "source_file_path": MetadataFields.ABSOLUTE_PATH,
    "file_path": MetadataFields.ABSOLUTE_PATH,
    "path": MetadataFields.ABSOLUTE_PATH,
}


class MetadataManager:
    """
    Manages the lifecycle of document metadata prior to chunking.

    `MetadataManager` standardizes metadata keys produced by different
    loaders, enriches metadata with project-defined fields, validates
    that required fields are present, and merges loader-provided
    metadata with automatically generated metadata. Downstream pipeline
    stages depend only on this class's public methods and require no
    knowledge of how metadata is internally derived.
    """

    def standardize_keys(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata keys to their canonical field names.

        Keys recognized as aliases (see `_FIELD_ALIASES`) are renamed to
        their canonical equivalent. Unrecognized keys are preserved
        as-is, allowing loaders or future metadata sources to introduce
        new fields without requiring changes to this method.

        Args:
            metadata: Raw metadata dictionary, typically produced by a
                document loader.

        Returns:
            A new dictionary with standardized key names.
        """
        standardized: Dict[str, Any] = {}
        for key, value in metadata.items():
            canonical_key = _FIELD_ALIASES.get(key, key)
            standardized[canonical_key] = value
        return standardized

    def generate_document_id(
        self,
        source: str,
        content: str,
        page: Optional[int] = None,
    ) -> str:
        """
        Generate a deterministic, unique document identifier.

        The identifier is derived from a hash of the source identifier,
        page number, and content, so that identical content from the
        same source and page always produces the same identifier
        across ingestion runs.

        Args:
            source: Identifier of the document's origin (e.g., file
                path or URL).
            content: Textual content of the document.
            page: Page number, when applicable.

        Returns:
            A hexadecimal string uniquely identifying the document.
        """
        hasher = hashlib.sha256()
        hasher.update(source.encode("utf-8"))
        hasher.update(str(page).encode("utf-8"))
        hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()

    def compute_checksum(self, content: str) -> str:
        """
        Compute a checksum of the document content.

        Args:
            content: Textual content of the document.

        Returns:
            A hexadecimal SHA-256 checksum of the content.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def enrich_metadata(
        self,
        metadata: Dict[str, Any],
        content: str,
        source: str,
        file_type: str,
        page: Optional[int] = None,
        include_checksum: bool = False,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enrich metadata with project-defined fields.

        Adds a document identifier, source, file type, page number,
        creation timestamp, and content length. Existing values already
        present in `metadata` are preserved and take precedence over
        generated values.

        Args:
            metadata: Standardized metadata dictionary to enrich.
            content: Textual content of the document, used to compute
                content length, checksum, and the document identifier.
            source: Identifier of the document's origin (e.g., file
                path or URL).
            file_type: File type or extension of the source document.
            page: Page number, when applicable.
            include_checksum: Whether to compute and attach a content
                checksum.
            language: Optional language code for the document content.

        Returns:
            A new dictionary containing the enriched metadata.
        """
        generated: Dict[str, Any] = {
            MetadataFields.DOCUMENT_ID: self.generate_document_id(source, content, page),
            MetadataFields.SOURCE: source,
            MetadataFields.FILE_TYPE: file_type,
            MetadataFields.PAGE: page,
            MetadataFields.CREATED_AT: datetime.now(timezone.utc).isoformat(),
            MetadataFields.CONTENT_LENGTH: len(content),
        }

        if include_checksum:
            generated[MetadataFields.CHECKSUM] = self.compute_checksum(content)

        if language is not None:
            generated[MetadataFields.LANGUAGE] = language

        enriched = self.merge_metadata(existing=metadata, generated=generated)

        logger.info(
            "Enriched metadata for document_id=%s source=%s page=%s",
            enriched.get(MetadataFields.DOCUMENT_ID),
            enriched.get(MetadataFields.SOURCE),
            enriched.get(MetadataFields.PAGE),
        )

        return enriched

    def merge_metadata(
        self,
        existing: Dict[str, Any],
        generated: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge automatically generated metadata into existing metadata.

        Existing metadata values take precedence: a generated value is
        only applied when the corresponding key is absent or `None` in
        the existing metadata. This preserves any metadata already
        supplied earlier in the pipeline rather than overwriting it.

        Args:
            existing: Metadata dictionary already associated with the
                document.
            generated: Metadata dictionary produced automatically by
                this manager.

        Returns:
            A new dictionary containing the merged metadata.
        """
        merged = dict(existing)
        for key, value in generated.items():
            if merged.get(key) is None:
                merged[key] = value
        return merged

    def validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate that required metadata fields are present and
        non-empty.

        Args:
            metadata: Metadata dictionary to validate.

        Raises:
            InvalidMetadataError: If one or more required fields are
                missing or empty.
        """
        missing_fields = [
            field
            for field in _REQUIRED_FIELDS
            if metadata.get(field) in (None, "")
        ]

        if missing_fields:
            logger.error("Metadata validation failed. Missing fields: %s", missing_fields)
            raise InvalidMetadataError(
                f"Metadata is missing required field(s): {missing_fields}"
            )

    def process(
        self,
        metadata: Dict[str, Any],
        content: str,
        source: str,
        file_type: str,
        page: Optional[int] = None,
        include_checksum: bool = False,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full metadata processing sequence: standardize,
        enrich, and validate.

        This is the primary entry point downstream pipeline stages
        should use, so they do not need to know the individual steps
        performed internally.

        Args:
            metadata: Raw metadata dictionary, typically produced by a
                document loader.
            content: Textual content of the document.
            source: Identifier of the document's origin (e.g., file
                path or URL).
            file_type: File type or extension of the source document.
            page: Page number, when applicable.
            include_checksum: Whether to compute and attach a content
                checksum.
            language: Optional language code for the document content.

        Returns:
            A fully standardized, enriched, and validated metadata
            dictionary.

        Raises:
            InvalidMetadataError: If the resulting metadata is missing
                required fields.
        """
        standardized = self.standardize_keys(metadata)
        enriched = self.enrich_metadata(
            metadata=standardized,
            content=content,
            source=source,
            file_type=file_type,
            page=page,
            include_checksum=include_checksum,
            language=language,
        )
        self.validate_metadata(enriched)
        return enriched