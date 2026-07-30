"""
CLI entry point for document ingestion in ResearchRAG.

This script orchestrates the complete document indexing workflow. It acts
as the composition root, parsing command-line arguments, initializing
concrete service implementations, injecting them into the `IndexingPipeline`,
and reporting the final statistics. It handles exit codes to integrate cleanly
with CI/CD pipelines or cron jobs.
"""

import argparse
import sys
from pathlib import Path

# Ensure the 'src' module can be found when running from the 'scripts' directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.embedding_manager import EmbeddingManager
from src.loaders.pdf_loader import PDFLoader
from src.pipelines.indexing_pipeline import IndexingPipeline, IndexingPipelineError
from src.preprocessing.splitter import DocumentSplitter
from src.vectorstores.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the ingestion script.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into the ResearchRAG vector database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path or URI to the document(s) to ingest (e.g., /data/papers.pdf)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an optional custom configuration (.env) file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution flow for the ingestion script.

    Initializes dependencies, constructs the pipeline, processes the
    document source, and handles process termination.
    """
    args = parse_args()

    # In a full implementation, the --config flag would trigger a reload
    # of the `settings` object before components are instantiated.
    if args.config:
        logger.info("Custom config path provided: %s", args.config)

    logger.info("Starting ResearchRAG ingestion script...")
    logger.info("Target source: %s", args.source)

    try:
        # 1. Dependency Resolution
        # Instantiate the concrete services required by the pipeline.
        # These pull their default configurations directly from `settings`.
        logger.debug("Initializing pipeline components...")
        
        loader = PDFLoader()
        splitter = DocumentSplitter()
        
        # Retrieve the configured embedding provider (e.g., SentenceTransformers)
        embedding_manager = EmbeddingManager()
        embedding_model = embedding_manager.get_provider()
        
        vector_store_backend = ChromaVectorStore()
        vector_store_backend.initialize()
        vector_store_backend.create_collection(settings.vector_store.collection_name)

        # 2. Pipeline Construction (Dependency Injection)
        pipeline = IndexingPipeline(
            loader=loader,
            splitter=splitter,
            embedding_model=embedding_model,
            vector_store=vector_store_backend,
        )

        # 3. Pipeline Execution
        report = pipeline.run(source=args.source)

    except IndexingPipelineError as error:
        logger.error("Ingestion failed during pipeline execution: %s", error)
        sys.exit(1)
    except Exception as error:
        logger.critical("An unexpected critical error occurred: %s", error)
        sys.exit(1)

    # 4. Result Formatting
    print("\n" + "=" * 50)
    print("           RESEARCH RAG - INGESTION SUMMARY")
    print("=" * 50)
    print(f"Source:           {args.source}")
    print(f"Status:           SUCCESS")
    print(f"Time Elapsed:     {report.elapsed_time_seconds:.2f} seconds")
    print(f"Embedding Model:  {report.embedding_model}")
    print(f"Vector Database:  {report.vector_store}")
    print("-" * 50)
    print(f"Documents Loaded: {report.indexed_documents}")
    print(f"Chunks Indexed:   {report.indexed_chunks}")
    if report.failed_documents > 0:
        print(f"Failed Documents: {report.failed_documents}")
    print("=" * 50 + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
