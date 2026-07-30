"""
CLI entry point for querying the ResearchRAG system.

This script acts as the composition root for the question-answering
flow. It parses user input from the command line, initializes the
required retrieval and generation services via dependency injection,
executes the pipeline, and safely formats the final output to standard out.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the 'src' module can be found when running from the 'scripts' directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.embeddings.embedding_manager import EmbeddingManager
from src.llms.base_llm import LLMConfigurationError, LLMGenerationError
from src.llms.llm_factory import LLMFactory
from src.pipelines.rag_pipeline import RAGPipeline, RAGPipelineError
from src.pipelines.retrieval_pipeline import (
    NoResultsFoundError,
    RetrievalPipeline,
    RetrievalPipelineError,
)
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.vectorstores.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the query script.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query the ResearchRAG system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="The question to ask the RAG system.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of documents to retrieve. Overrides settings.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM generation temperature. Overrides settings.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Display the retrieved context snippets used for the answer.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Display full metadata and similarity scores for sources.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output to the console.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution flow for the query script.

    Initializes all dependencies, executes the query via the pipelines,
    handles domain-specific exceptions, and formats the output for the user.
    """
    args = parse_args()

    # Optional overrides
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Initializing ResearchRAG query script...")
    
    if not args.question or not args.question.strip():
        print("\n[ERROR] The question cannot be empty.\n", file=sys.stderr)
        sys.exit(1)

    # Track total script execution time
    script_start_time = time.perf_counter()

    try:
        # 1. Dependency Resolution
        logger.debug("Loading embedding model...")
        embedding_manager = EmbeddingManager()
        embedding_model = embedding_manager.get_provider()

        logger.debug("Connecting to vector database...")
        vector_store_backend = ChromaVectorStore()
        vector_store_backend.initialize()
        vector_store_backend.create_collection(settings.vector_store.collection_name)
        
        logger.debug("Initializing LLM factory...")
        llm_factory = LLMFactory()
        llm = llm_factory.get_client()

        logger.debug("Initializing retrieval components...")
        retriever = Retriever(vector_store=vector_store_backend, embedding_manager=embedding_manager)
        
        reranker = None
        if getattr(settings.retrieval, "use_reranker", False):
            reranker = Reranker()

        # 2. Pipeline Construction (Dependency Injection)
        # Note: Depending on the exact iteration of the RAGPipeline architecture, 
        # it may use RetrievalPipeline internally or the raw components. 
        # We initialize both to adhere to architectural requirements.
        retrieval_pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
            default_top_k=args.top_k,
        )

        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
        )

        # Apply temperature override to settings temporarily if requested
        if args.temperature is not None:
            settings.llm.temperature = args.temperature

        # 3. Execution
        logger.info("Executing query: %r", args.question)
        
        # Execute query. (Assuming RAGPipeline manages the end-to-end flow)
        response = rag_pipeline.answer(query=args.question)

    except NoResultsFoundError:
        print("\n[INFO] No relevant documents were found in the database to answer your query.")
        print("Please try rephrasing your question or ingesting more documents.\n")
        sys.exit(0)
    except (RetrievalPipelineError, RAGPipelineError) as error:
        print(f"\n[ERROR] Pipeline execution failed: {error}\n", file=sys.stderr)
        logger.error("Pipeline failure: %s", error)
        sys.exit(1)
    except (LLMConfigurationError, LLMGenerationError) as error:
        print(f"\n[ERROR] Language Model failed: {error}\n", file=sys.stderr)
        print("Please check your API keys and model configuration in settings.", file=sys.stderr)
        logger.error("LLM failure: %s", error)
        sys.exit(1)
    except Exception as error:
        print(f"\n[CRITICAL] An unexpected system error occurred: {error}\n", file=sys.stderr)
        logger.critical("Unexpected error: %s", error, exc_info=True)
        sys.exit(1)

    script_total_time = time.perf_counter() - script_start_time

    # 4. Result Formatting (User Output)
    print("\n" + "=" * 60)
    print("                      RESEARCH RAG                      ")
    print("=" * 60)
    print(f"\nQUESTION:\n{args.question}\n")
    print("-" * 60)
    print(f"\nANSWER:\n{response.answer}\n")
    print("=" * 60)

    # 5. Optional Sources and Metadata Output
    if args.show_sources or args.show_metadata:
        print("\n[RETRIEVED SOURCES]\n")
        for idx, doc in enumerate(response.retrieved_documents, start=1):
            
            source_name = getattr(doc, 'metadata', {}).get('source', 'Unknown')
            sim_score = getattr(doc, 'similarity_score', 0.0)
            rerank_score = getattr(doc, 'rerank_score', None)
            
            print(f"--- Document {idx} | Source: {source_name} ---")
            
            if args.show_metadata:
                print(f"Similarity Score: {sim_score:.4f}")
                if rerank_score is not None:
                    print(f"Rerank Score:     {rerank_score:.4f}")
                print(f"Full Metadata:    {getattr(doc, 'metadata', {})}")
                
            if args.show_sources:
                content = getattr(doc, 'page_content', str(doc))
                print(f"\nContent Snippet:\n{content.strip()}...\n")
                
        print("=" * 60)

    # 6. Performance Statistics
    print("\n[PERFORMANCE STATISTICS]")
    
    # We attempt to pull granular latency metrics if the response object supports them,
    # otherwise fallback to total script execution time.
    retrieval_latency = getattr(response, "retrieval_latency_seconds", None)
    generation_latency = getattr(response, "generation_latency_seconds", None)
    
    if retrieval_latency is not None:
        print(f"Retrieval Latency:  {retrieval_latency:.3f}s")
    if generation_latency is not None:
        print(f"Generation Latency: {generation_latency:.3f}s")
        
    print(f"Total Script Time:  {script_total_time:.3f}s")
    print("\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
