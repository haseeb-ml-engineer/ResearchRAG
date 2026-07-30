# ResearchRAG — System Architecture

> Status: Living document. This file is the canonical architectural reference for the ResearchRAG codebase and should be updated alongside any structural change to the system.

---

# Project Overview

ResearchRAG is a production-grade Retrieval-Augmented Generation (RAG) system designed to answer questions over large, evolving collections of research documents (papers, technical reports, internal knowledge bases, and long-form PDFs). Rather than relying solely on the parametric knowledge of a large language model (LLM), ResearchRAG grounds every generated answer in retrieved, verifiable source material, reducing hallucination and enabling citation-backed responses.

The system is built to serve as a reference implementation of RAG engineering best practices: clear module boundaries, deterministic and testable pipeline stages, explicit interfaces between components, and a design that anticipates the operational demands of a real product rather than a one-off notebook experiment. Where most RAG tutorials collapse ingestion, retrieval, and generation into a single script, ResearchRAG treats each of these as an independently deployable, independently testable subsystem connected by well-defined contracts.

ResearchRAG is intended for two audiences:

- **Engineers** who need a maintainable, extensible foundation for building retrieval-augmented applications in production.
- **Researchers and reviewers** who need transparency into how an answer was produced — what was retrieved, why it was retrieved, and how it was used to construct the final response.

---

# Goals

The primary objectives guiding every design decision in ResearchRAG are:

1. **Correctness over cleverness.** Every stage of the pipeline should behave predictably and be independently verifiable. Retrieval quality and answer grounding take priority over exotic prompting tricks.
2. **Modularity.** Each pipeline stage (loading, cleaning, chunking, embedding, storage, retrieval, generation) must be swappable without requiring changes to unrelated components. A new vector store or a new embedding provider should be a configuration change, not a rewrite.
3. **Observability.** Every retrieval and generation event should be traceable: which documents were retrieved, which chunks were used, what prompt was constructed, and what the model returned.
4. **Testability.** Every module exposes a narrow, well-typed interface so it can be unit tested in isolation, with integration tests validating the composed pipeline.
5. **Operational readiness.** The architecture should scale from a local CLI prototype to a served API without requiring a structural rewrite — the same core pipeline should be usable in both contexts.
6. **Extensibility.** New retrieval strategies (hybrid search, reranking, multi-query expansion) and new agentic behaviors should be additive, not disruptive, to the existing pipeline.

---

# High-Level Architecture

ResearchRAG is organized as a pipeline of discrete, composable stages, orchestrated by a central pipeline controller. Each stage consumes a well-defined input contract and produces a well-defined output contract, allowing stages to be developed, tested, and replaced independently.

At the highest level, the system has two operational paths:

- **Ingestion Path (offline / batch):** Raw documents are loaded, cleaned, chunked, embedded, and persisted into a vector database. This path runs whenever the knowledge base is built or updated.
- **Query Path (online / real-time):** A user query is embedded, relevant chunks are retrieved from the vector database, a prompt is constructed from the retrieved context, and the LLM generates a grounded response.

```
                              ┌───────────────────────────────────────────┐
                              │              INGESTION PATH                │
                              │              (offline / batch)             │
                              └───────────────────────────────────────────┘

   ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌────────────────┐     ┌───────────────┐
   │  Document  │────▶│  Cleaning  │────▶│  Chunking  │────▶│    Embedding    │────▶│  Vector Store  │
   │  Loaders   │     │            │     │            │     │   Generation    │     │   (persisted)  │
   └───────────┘     └───────────┘     └───────────┘     └────────────────┘     └───────┬───────┘
                                                                                          │
                                                                                          │
                              ┌───────────────────────────────────────────┐              │
                              │               QUERY PATH                   │              │
                              │              (online / real-time)          │              │
                              └───────────────────────────────────────────┘              │
                                                                                          │
   ┌───────────┐     ┌────────────────┐     ┌────────────┐     ┌────────────────┐        │
   │    User    │────▶│  Query Embed-   │────▶│  Retrieval  │◀───────────────────────────┘
   │   Query    │     │     ding        │     │   Engine    │
   └───────────┘     └────────────────┘     └─────┬──────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │  Prompt Construc-  │
                                          │      tion          │
                                          └─────────┬─────────┘
                                                     │
                                                     ▼
                                             ┌───────────────┐
                                             │      LLM       │
                                             │  (Generation)  │
                                             └───────┬───────┘
                                                     │
                                                     ▼
                                             ┌───────────────┐
                                             │ Final Response │
                                             │ (+ citations)  │
                                             └───────────────┘
```

The **Ingestion Path** and **Query Path** share the vector store as their point of integration but otherwise have no runtime coupling — ingestion can be re-run at any time (e.g., nightly, on document upload) without interrupting query serving, and query serving never mutates the underlying store.

A central `Pipeline` orchestrator coordinates stage execution, handles cross-cutting concerns (logging, timing, error propagation), and exposes a single entry point (`run_ingestion()` / `run_query()`) that downstream consumers — a CLI, a batch job, or an API layer — can call without needing to understand internal stage wiring.

---

# Data Flow

The system's core value is produced by a linear, auditable data flow. Each stage below is described in terms of its input, its transformation, and its output, so that any stage can be reasoned about, tested, or replaced in isolation.

```
Document Loading
      ↓
   Cleaning
      ↓
   Chunking
      ↓
Embedding Generation
      ↓
 Vector Database
      ↓
   Retrieval
      ↓
Prompt Construction
      ↓
      LLM
      ↓
 Final Response
```

### 1. Document Loading

- **Enters:** Raw source files (PDF, HTML, Markdown, plain text, DOCX) or references to external sources (URLs, cloud storage paths).
- **Transformation:** Format-specific loaders extract raw textual content and preserve source-level metadata (file name, page number, section headers, source URL, ingestion timestamp).
- **Leaves:** A normalized in-memory `RawDocument` object containing unstructured text plus a metadata dictionary. No cleaning or structural interpretation has occurred yet — this stage is purely concerned with extraction.

### 2. Cleaning

- **Enters:** `RawDocument` objects from the loading stage.
- **Transformation:** Removal of boilerplate (headers, footers, page numbers), normalization of whitespace and encoding artifacts, de-hyphenation of line-wrapped text, and optional filtering of non-informative content (tables of contents, reference lists, if configured).
- **Leaves:** A `CleanedDocument` — text suitable for semantic chunking, with original metadata preserved and augmented with cleaning provenance (e.g., which normalization rules were applied).

### 3. Chunking

- **Enters:** `CleanedDocument` objects.
- **Transformation:** Text is split into retrieval-sized units using a configurable strategy (fixed-size with overlap, sentence-boundary aware, semantic/topic-boundary aware, or structure-aware for documents with headings). Each chunk retains a back-reference to its parent document and position.
- **Leaves:** A list of `Chunk` objects, each with chunk text, a stable chunk ID, parent document ID, and positional metadata (page, offset, section).

### 4. Embedding Generation

- **Enters:** `Chunk` objects.
- **Transformation:** Each chunk's text is passed through an embedding model to produce a fixed-dimensional dense vector representation. This stage is provider-agnostic: the embedding model is injected as a dependency, not hardcoded.
- **Leaves:** `EmbeddedChunk` objects pairing each chunk with its vector representation and the embedding model identifier/version used (critical for future re-embedding and consistency checks).

### 5. Vector Database

- **Enters:** `EmbeddedChunk` objects (write path) or a query vector (read path).
- **Transformation:** On the write path, vectors and associated metadata are persisted into an index optimized for approximate nearest-neighbor search. On the read path, a similarity search is performed against the index.
- **Leaves:** On write, a persisted, queryable index. On read, an ordered list of candidate chunks with similarity scores.

### 6. Retrieval

- **Enters:** A user query (embedded into a vector) and access to the vector database.
- **Transformation:** The retrieval engine executes similarity search, applies any configured filtering (metadata constraints, recency, source restrictions), and selects the top-k most relevant chunks. This stage is also the extension point for future strategies such as hybrid (sparse + dense) search and reranking.
- **Leaves:** A ranked `RetrievalResult` — an ordered list of chunks with scores and full source metadata, ready for prompt assembly.

### 7. Prompt Construction

- **Enters:** The original user query and the `RetrievalResult`.
- **Transformation:** Retrieved chunks are formatted into a structured context block, deduplicated, truncated to fit the model's context window budget, and combined with the system instructions and the user query according to a defined prompt template.
- **Leaves:** A fully assembled prompt payload (system message, context block, user query) ready to be sent to the LLM, along with a record of exactly which chunks were included (for later citation and auditability).

### 8. LLM

- **Enters:** The assembled prompt payload.
- **Transformation:** The language model generates a response conditioned on the provided context. This stage is provider-agnostic and abstracted behind a common generation interface so that the underlying model can be swapped without touching upstream stages.
- **Leaves:** A raw model response, including any generation metadata (token usage, latency, finish reason).

### 9. Final Response

- **Enters:** The raw model response and the retrieval provenance from earlier stages.
- **Transformation:** The response is post-processed — citations are attached by mapping referenced content back to source chunks, formatting is normalized, and safety/quality checks are applied.
- **Leaves:** The final, user-facing answer object: response text, cited sources, and confidence/telemetry metadata.

---

# Project Modules

The `src` directory is organized so that each folder maps to exactly one responsibility in the data flow described above. No folder should need to know the internal implementation details of another; folders communicate only through the typed interfaces defined in `src/core`.

- **`src/core`** — Defines the shared abstractions and data contracts used across the entire system: base classes/interfaces for loaders, chunkers, embedders, vector stores, retrievers, and generators, along with the core data models (`RawDocument`, `Chunk`, `EmbeddedChunk`, `RetrievalResult`). This is the "contract layer" — every other module depends on it, but it depends on nothing else in `src`.

- **`src/loaders`** — Responsible solely for extracting raw text and metadata from source files or external sources. Contains one implementation per supported format (PDF, HTML, Markdown, plain text, etc.), all conforming to the loader interface defined in `core`.

- **`src/cleaning`** — Responsible for normalizing and sanitizing raw extracted text. Contains the individual cleaning rules (whitespace normalization, boilerplate removal, encoding fixes) and a configurable pipeline that composes them.

- **`src/chunking`** — Responsible for splitting cleaned documents into retrieval-sized units. Contains multiple chunking strategies (fixed-size, sentence-aware, semantic, structure-aware) behind a common interface, selectable via configuration.

- **`src/embeddings`** — Responsible for converting text into vector representations. Wraps one or more embedding providers behind a common interface, so the rest of the system is agnostic to which embedding model is in use.

- **`src/vectorstore`** — Responsible for persistence and similarity search over embedded chunks. Wraps the underlying vector database (e.g., FAISS, Chroma, Pinecone, Weaviate) behind a common interface, isolating the rest of the system from vendor-specific APIs.

- **`src/retrieval`** — Responsible for translating a user query into a ranked set of relevant chunks. Houses retrieval strategy logic (similarity search, filtering, and — in future iterations — hybrid search and reranking).

- **`src/generation`** — Responsible for prompt construction and interaction with the LLM. Contains prompt templates, context-window budget management, and the generation interface that wraps the underlying LLM provider.

- **`src/pipeline`** — Responsible for orchestration. Composes the modules above into the ingestion pipeline and the query pipeline, handling stage sequencing, error propagation, and logging, without containing business logic of its own.

- **`src/config`** — Responsible for centralized configuration management (model selection, chunk sizes, retrieval parameters, storage backends), typically loaded from environment variables or configuration files, so that behavior can change without code changes.

- **`src/utils`** — Responsible for small, shared, stateless helper functionality (logging setup, timing decorators, ID generation) that does not belong to any single pipeline stage.

- **`tests`** — Mirrors the `src` structure. Contains unit tests for each module in isolation and integration tests that validate the composed pipeline end-to-end.

---

# Design Principles

ResearchRAG's structure is deliberately shaped by a small set of software engineering principles. These are not academic exercises — each one directly addresses a failure mode commonly seen in RAG systems that are built quickly and then become unmaintainable.

**Single Responsibility Principle.** Every module in `src` does exactly one job — a loader loads, a chunker chunks, a retriever retrieves. This matters because RAG systems accumulate complexity fast: when chunking logic is entangled with embedding logic, a change to one silently breaks the other. Isolating responsibility means a bug or a change request maps cleanly to a single module.

**Separation of Concerns.** Ingestion, retrieval, and generation are architecturally separate paths that only meet at the vector store. This matters because these three concerns evolve on different timelines and have different operational profiles — ingestion is batch and infrequent, retrieval and generation are real-time and latency-sensitive. Coupling them makes it impossible to scale or deploy them independently.

**Modular Design.** Every stage is defined by an interface in `src/core` before it is implemented. This matters because it allows multiple implementations (e.g., three different chunking strategies) to coexist and be selected via configuration, rather than via conditional branches scattered through the codebase.

**Extensibility.** New capabilities — hybrid search, reranking, multi-query retrieval — are designed to slot into existing interfaces (primarily `src/retrieval`) rather than requiring changes to the pipeline orchestrator or upstream stages. This matters because a RAG system's retrieval strategy is its most actively researched component, and the architecture should not resist iteration on it.

**Readability.** Code in ResearchRAG favors explicitness over cleverness — clear names, explicit types, and small functions over dense one-liners. This matters because a system meant to be studied and extended by other engineers must be legible on first read, not only to its original author.

**Testability.** Every module's narrow interface and explicit inputs/outputs make it possible to unit test each stage with synthetic data, without standing up a real vector database or calling a real LLM. This matters because RAG systems that can only be tested end-to-end are slow to iterate on and prone to regressions slipping through unnoticed.

---

# Folder Structure Explanation

```
ResearchRAG/
├── src/
│   ├── core/           # Shared interfaces and data contracts
│   ├── loaders/         # Document ingestion from raw sources
│   ├── cleaning/         # Text normalization and sanitization
│   ├── chunking/         # Chunking strategies
│   ├── embeddings/       # Embedding model wrappers
│   ├── vectorstore/      # Vector database integrations
│   ├── retrieval/        # Query-time retrieval logic
│   ├── generation/       # Prompt construction and LLM interaction
│   ├── pipeline/         # Ingestion and query orchestration
│   ├── config/           # Centralized configuration
│   └── utils/            # Shared stateless utilities
├── tests/
│   ├── unit/            # Per-module isolated tests
│   └── integration/      # End-to-end pipeline tests
├── docs/
│   └── architecture.md   # This document
├── scripts/              # CLI entry points for ingestion/querying
└── data/                 # Local sample/test documents (not for production data)
```

- **`src/`** exists as the single source of truth for application logic, structured so that the folder layout mirrors the pipeline's data flow — reading the folder names top to bottom approximates reading the pipeline stage by stage.
- **`tests/`** mirrors `src/` exactly, so that for any module a contributor can immediately locate its corresponding tests without searching.
- **`docs/`** exists to keep architectural intent, decisions, and rationale co-located with the code they describe, rather than living in an external wiki that drifts out of sync.
- **`scripts/`** exists to separate operational entry points (how the system is *run*) from the system's internal logic (how the system *works*), keeping `src` free of CLI argument-parsing concerns.
- **`data/`** exists as a clearly-labeled sandbox for local development and test fixtures, explicitly not intended to hold production document sets, to avoid accidental coupling between code and a specific dataset.

---

# Coding Standards

Consistency across the codebase is treated as a first-class requirement, not a stylistic preference, because it directly affects how quickly a contributor can trust and extend unfamiliar code.

**Naming Conventions.** Modules, files, and functions use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Names should describe intent, not implementation (`Retriever` rather than `FaissWrapper`). Boolean variables and functions are prefixed with `is_`, `has_`, or `should_`.

**Type Hints.** All public functions and methods carry complete type hints for parameters and return values. Data contracts between pipeline stages (`Chunk`, `EmbeddedChunk`, `RetrievalResult`, etc.) are defined as typed data classes rather than loosely-structured dictionaries, so that a stage's expected input/output shape is enforced by the type system rather than by convention.

**Docstrings.** Every public class and function includes a docstring describing its purpose, arguments, return value, and any exceptions it may raise. Docstrings follow a consistent format (Google-style) throughout the codebase so that documentation can be mechanically generated from source.

**Logging.** Logging is used in place of print statements throughout the pipeline. Each stage logs entry, exit, and key metrics (e.g., number of chunks produced, retrieval latency, number of candidates returned) at an appropriate level (`DEBUG` for detail, `INFO` for stage completion, `WARNING`/`ERROR` for recoverable/unrecoverable issues), enabling production observability without code changes.

**Error Handling.** Each module defines and raises specific, descriptive exceptions rather than allowing generic exceptions to propagate. Pipeline orchestration code is responsible for catching stage-level exceptions, logging sufficient context to diagnose the failure, and deciding whether to fail fast or degrade gracefully (e.g., skipping an unparseable document rather than aborting an entire ingestion run).

**Comments.** Comments explain *why*, not *what* — the code itself should make the "what" self-evident through naming and structure. Comments are reserved for non-obvious tradeoffs, workarounds for external library quirks, or references to relevant design decisions in `docs/`.

**Import Ordering.** Imports are grouped and ordered as: standard library, third-party packages, then local application imports, with a blank line separating each group and alphabetical ordering within each group. This ordering is enforced automatically by tooling rather than by manual review.

**Formatting.** Code formatting (line length, quote style, whitespace) is enforced automatically via an autoformatter as part of pre-commit checks and CI, so that formatting is never a subject of code review discussion.

---

# Future Roadmap

ResearchRAG's architecture is intentionally structured so that the following capabilities can be introduced additively, primarily by extending `src/retrieval` and `src/generation`, without requiring changes to the core pipeline contracts.

- **Hybrid Search.** Combine dense vector similarity with sparse keyword-based retrieval (e.g., BM25) to improve recall on queries with rare terms, proper nouns, or exact-match requirements that dense embeddings handle poorly in isolation.

- **Metadata Filtering.** Extend the retrieval interface to support structured filters (date ranges, document source, author, document type) applied alongside vector similarity, enabling more precise, scoped retrieval.

- **Multi-Query Retrieval.** Generate multiple reformulations of a single user query to retrieve a broader and more diverse candidate set, then merge and deduplicate results before ranking.

- **Query Expansion.** Automatically enrich a user's query with related terms, synonyms, or clarifying context prior to embedding, improving retrieval recall for underspecified queries.

- **Reranking.** Introduce a secondary, more computationally expensive reranking stage (e.g., a cross-encoder model) applied to the top-N retrieved candidates to improve precision before prompt construction.

- **LangGraph Integration.** Model more complex, stateful retrieval-and-reasoning workflows (e.g., iterative retrieval, self-correction loops) as explicit graphs, enabling conditional branching and multi-step reasoning beyond a linear pipeline.

- **AI Agents.** Introduce agentic capabilities that allow the system to decide when to retrieve, when to ask a clarifying question, and when to invoke external tools, moving beyond a single fixed retrieve-then-generate pattern.

- **Evaluation Framework.** Build a systematic evaluation harness measuring retrieval quality (precision/recall at k), answer faithfulness, and citation accuracy against curated benchmark datasets, integrated into CI to catch regressions.

- **FastAPI Service Layer.** Expose the query pipeline as a versioned HTTP API, with request/response schemas, authentication, and rate limiting, enabling ResearchRAG to be consumed as a backend service rather than only a CLI tool.

- **Docker Packaging.** Containerize the application and its dependencies (including the vector database, where applicable) for reproducible local development and deployment parity across environments.

- **Deployment.** Define production deployment targets (e.g., container orchestration on Kubernetes or a managed container service), including health checks, horizontal scaling policies for the query path, and a separate scheduling mechanism for the ingestion path.

---

*This document is expected to evolve alongside the codebase. Any architectural change that affects module boundaries, data contracts, or the pipeline flow described above should be accompanied by an update to this file in the same pull request.*