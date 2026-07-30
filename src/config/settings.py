"""
Centralized configuration for ResearchRAG.

This module defines a single, typed configuration object that every
component of the system (loaders, chunkers, embedders, vector store,
retriever, generation layer, API, and frontend) reads from. Configuration
values are loaded from environment variables and/or a local `.env` file,
with sensible defaults provided for local development.

Usage:
    from src.config.settings import settings

    print(settings.embedding.model)
    print(settings.paths.vector_db_dir)
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """General application metadata and runtime mode."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
    )

    name: str = Field(
        default="ResearchRAG",
        description="Human-readable name of the application.",
    )
    version: str = Field(
        default="0.1.0",
        description="Current application version, following semantic versioning.",
    )
    environment: str = Field(
        default="development",
        description="Deployment environment, e.g. 'development', 'staging', or 'production'.",
    )
    debug: bool = Field(
        default=False,
        description="Enables verbose logging and developer-only behavior when true.",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Ensure the environment is one of the recognized deployment stages."""
        allowed = {"development", "staging", "production"}
        normalized = value.lower()
        if normalized not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}, got '{value}'")
        return normalized


class PathSettings(BaseSettings):
    """Filesystem locations used for document storage, persistence, and logs."""

    model_config = SettingsConfigDict(
        env_prefix="PATH_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for all locally stored data.",
    )
    raw_documents_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory containing unprocessed source documents awaiting ingestion.",
    )
    processed_documents_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory containing cleaned and chunked document artifacts.",
    )
    vector_db_dir: Path = Field(
        default=Path("data/vectorstore"),
        description="Directory where the persisted vector database is stored.",
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory where application log files are written.",
    )


class EmbeddingSettings(BaseSettings):
    """Configuration for the embedding generation stage."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = Field(
        default="sentence-transformers",
        description="Name of the embedding provider (e.g. 'sentence-transformers', 'openai').",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Identifier of the embedding model to use.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Number of text chunks embedded per batch.",
    )


class VectorStoreSettings(BaseSettings):
    """Configuration for the vector database backend."""

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_STORE_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = Field(
        default="chromadb",
        description="Name of the vector database provider.",
    )
    collection_name: str = Field(
        default="research_documents",
        description="Name of the collection/index used to store document embeddings.",
    )


class ChunkingSettings(BaseSettings):
    """Configuration for splitting cleaned documents into retrieval-sized units."""

    model_config = SettingsConfigDict(
        env_prefix="CHUNKING_",
        env_file=".env",
        extra="ignore",
    )

    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of characters per chunk.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Number of overlapping characters between consecutive chunks.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info) -> int:
        """Ensure the chunk overlap is smaller than the chunk size."""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class RetrievalSettings(BaseSettings):
    """Configuration for query-time retrieval behavior."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=".env",
        extra="ignore",
    )

    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of most relevant chunks to retrieve per query.",
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score a chunk must meet to be considered relevant.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Identifier of the cross-encoder model used for reranking retrieved chunks.",
    )
    reranker_top_k: int = Field(
        default=3,
        gt=0,
        description="Number of top chunks to retain after reranking.",
    )


class LLMSettings(BaseSettings):
    """Configuration for the generation (language model) stage."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = Field(
        default="groq",
        description="Name of the LLM provider (e.g. 'groq', 'openai').",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Identifier of the language model used for answer generation.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling output randomness.",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum number of tokens the model may generate in a single response.",
    )


class APIKeySettings(BaseSettings):
    """
    External provider API keys.

    These values must never have default values. They are optional at the
    type level and are expected to be supplied exclusively through
    environment variables or a local `.env` file, never hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI, loaded from the OPENAI_API_KEY environment variable.",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="API key for Groq, loaded from the GROQ_API_KEY environment variable.",
    )


class Settings(BaseSettings):
    """
    Root configuration object aggregating every configuration section
    used across ResearchRAG.

    Each section is modeled as its own settings class so that related
    values stay grouped, independently documented, and independently
    testable, while still being accessible from a single object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)


# Global singleton configuration instance.
# Every module in the project should import this object rather than
# instantiating Settings() directly, ensuring a single consistent
# configuration is shared across the entire application.
settings = Settings()