# ResearchRAG

**A modular, production-oriented Retrieval-Augmented Generation system for grounded question answering over research documents.**

---

## Overview

ResearchRAG is a Retrieval-Augmented Generation (RAG) system designed to answer questions over large, evolving collections of research documents — papers, technical reports, and long-form PDFs — by grounding every response in retrieved source material rather than relying solely on a language model's parametric knowledge.

Large language models are prone to hallucination when asked about specialized, private, or rapidly changing content that falls outside or is underrepresented in their training data. Retrieval-Augmented Generation addresses this by retrieving relevant passages from a trusted document collection at query time and conditioning the model's generation on that retrieved context. This produces answers that are traceable to a source, reduces fabrication, and allows a knowledge base to be updated without retraining or fine-tuning any model.

ResearchRAG is built as a reference implementation of RAG engineering practices: clearly separated pipeline stages, explicit interfaces between components, and a structure designed to scale from local experimentation to a served application without architectural rework. The full technical design is documented in [`docs/architecture.md`](docs/architecture.md).

---

## Key Features

- **Multiple document format support** — ingestion of PDF, HTML, Markdown, and plain text sources through a common loader interface.
- **Modular architecture** — ingestion, retrieval, and generation are independently testable and independently replaceable components.
- **Configurable chunking** — multiple chunking strategies (fixed-size, sentence-aware, structure-aware) selectable via configuration.
- **Embedding generation** — provider-agnostic embedding layer supporting local and hosted models.
- **Vector database integration** — persistent similarity search backed by a dedicated vector store abstraction.
- **Semantic retrieval** — top-k similarity search over embedded document chunks at query time.
- **Citation-backed responses** — generated answers are linked back to the source chunks used to produce them.
- **Extensible design** — new retrieval strategies and generation providers can be added without modifying upstream pipeline stages.
- **FastAPI backend** *(Planned)* — a versioned HTTP API for serving the query pipeline as a backend service.
- **Streamlit frontend** *(Planned)* — a lightweight interactive interface for document upload and question answering.
- **Evaluation framework** *(Planned)* — systematic measurement of retrieval quality and answer faithfulness against benchmark datasets.

---

## System Architecture

ResearchRAG operates as two independent pipelines that share a common vector store: an offline **Document Ingestion Pipeline** and an online **User Query Pipeline**. A complete architectural breakdown, including module responsibilities and data contracts, is available in [`docs/architecture.md`](docs/architecture.md).

### Document Ingestion Pipeline

```
Documents ──▶ Loading ──▶ Cleaning ──▶ Chunking ──▶ Embedding ──▶ Vector Store
```

### User Query Pipeline

```
User Query ──▶ Query Embedding ──▶ Retrieval ──▶ Prompt Construction ──▶ LLM ──▶ Response + Citations
                                        ▲
                                        │
                                  Vector Store
```

---

## Folder Structure

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
│   └── architecture.md   # Full architectural reference
├── scripts/              # CLI entry points for ingestion and querying
└── data/                 # Local sample/test documents (not for production data)
```

- **`src/core`** defines the shared contracts every other module depends on.
- **`src/loaders`** through **`src/generation`** implement one pipeline stage each, following the flow described above.
- **`src/pipeline`** composes the individual stages into the ingestion and query pipelines.
- **`tests/`** mirrors `src/` so that every module's tests are easy to locate.
- **`docs/`** holds architectural documentation kept in sync with the codebase.

---

## Technology Stack

| Category | Technology |
|---|---|
| Core language | Python |
| Orchestration | LangChain |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| LLM providers | Groq / OpenAI |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Numerical / tensor computation | PyTorch, NumPy |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-organization>/ResearchRAG.git
cd ResearchRAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Populate it with the required values:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
VECTOR_DB_PATH=./data/vectorstore
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5. Run document ingestion

```bash
python scripts/ingest.py --source ./data/documents
```

### 6. Run the API server

```bash
uvicorn src.api.main:app --reload
```

### 7. Run the Streamlit frontend

```bash
streamlit run scripts/app.py
```

---

## Usage

A typical ResearchRAG workflow follows four steps:

1. **Upload documents** — place source files (PDF, HTML, Markdown, plain text) into the configured ingestion directory or upload them through the frontend.
2. **Build the index** — run the ingestion pipeline to clean, chunk, embed, and persist the documents into the vector store.
3. **Ask a question** — submit a natural language query through the API or frontend.
4. **Receive a grounded answer** — the system retrieves relevant passages, constructs a context-aware prompt, and returns an answer accompanied by citations to the source chunks used.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What method does the paper use to reduce hallucination?"}'
```

---

## Project Roadmap

- [x] Modular architecture
- [x] Configurable chunking strategies
- [x] Vector store abstraction
- [ ] FastAPI backend
- [ ] Streamlit frontend
- [ ] Hybrid Search
- [ ] Metadata Filtering
- [ ] Multi-query Retrieval
- [ ] Query Expansion
- [ ] Reranking
- [ ] LangGraph integration
- [ ] AI Agents
- [ ] Evaluation Dashboard
- [ ] Docker Deployment

---

## Repository Standards

ResearchRAG follows a consistent set of engineering standards across the codebase:

- **Type hints** on all public functions, methods, and data contracts.
- **Structured logging** at each pipeline stage in place of ad hoc print statements.
- **Automated testing**, with unit tests per module and integration tests for the composed pipelines.
- **Docstrings** on all public classes and functions, following a consistent format.
- **Clean Architecture**, with a clear separation between core contracts, pipeline stages, and orchestration logic.
- **SOLID principles**, applied throughout module design to keep components single-purpose and independently extensible.

---

## Contributing

Contributions are welcome. Before submitting a change:

1. Open an issue describing the proposed change or bug, unless one already exists.
2. Fork the repository and create a feature branch from `main`.
3. Follow the coding standards described above and in [`docs/architecture.md`](docs/architecture.md).
4. Add or update tests for any behavioral change.
5. Ensure the full test suite and linter pass locally before opening a pull request.
6. Submit a pull request with a clear description of the change and its motivation.

Architectural changes (new modules, altered data contracts, changes to pipeline flow) should be accompanied by a corresponding update to `docs/architecture.md` in the same pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

ResearchRAG draws on ideas and patterns established across the broader open-source machine learning and information retrieval community, including work in dense retrieval, vector search, and retrieval-augmented generation research. This project is independently developed and is not affiliated with any of the organizations whose tools or research it builds upon.