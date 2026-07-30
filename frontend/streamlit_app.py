"""
Streamlit frontend for ResearchRAG.

This module provides a modern, interactive web interface for users to
upload research documents, query the knowledge base, and view retrieved
sources. It acts strictly as a presentation layer and delegates all
heavy lifting to the FastAPI backend via HTTP requests.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

ENDPOINT_HEALTH = f"{API_URL}/health"
ENDPOINT_QUERY = f"{API_URL}/query"
ENDPOINT_INDEX = f"{API_URL}/index"
ENDPOINT_RESET = f"{API_URL}/rebuild-index"
ENDPOINT_STATS = f"{API_URL}/statistics"
ENDPOINT_CONFIG = f"{API_URL}/configuration"

UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject custom CSS to create a polished, modern appearance."""
    st.markdown(
        """
        <style>
        /* ---------- Global ---------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ---------- Header ---------- */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.8rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.3rem 0 0 0;
            opacity: 0.85;
            font-size: 0.95rem;
        }

        /* ---------- Cards ---------- */
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1.6rem;
            color: #2d3748;
        }
        .metric-card p {
            margin: 0.2rem 0 0 0;
            font-size: 0.8rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ---------- Status Badges ---------- */
        .status-online {
            display: inline-flex; align-items: center; gap: 6px;
            background: #c6f6d5; color: #22543d;
            padding: 4px 12px; border-radius: 20px;
            font-size: 0.8rem; font-weight: 600;
        }
        .status-offline {
            display: inline-flex; align-items: center; gap: 6px;
            background: #fed7d7; color: #742a2a;
            padding: 4px 12px; border-radius: 20px;
            font-size: 0.8rem; font-weight: 600;
        }

        /* ---------- Source Cards ---------- */
        .source-card {
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 0.8rem 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.6rem;
        }
        .source-card .source-header {
            font-weight: 600;
            color: #2d3748;
            font-size: 0.85rem;
        }
        .source-card .source-score {
            font-size: 0.75rem;
            color: #667eea;
            font-weight: 500;
        }
        .source-card .source-content {
            font-size: 0.82rem;
            color: #4a5568;
            margin-top: 0.4rem;
            line-height: 1.5;
        }

        /* ---------- Latency Bar ---------- */
        .latency-bar {
            display: flex;
            gap: 1.2rem;
            font-size: 0.78rem;
            color: #718096;
            padding-top: 0.3rem;
        }
        .latency-bar span {
            display: inline-flex; align-items: center; gap: 4px;
        }

        /* ---------- Section Dividers ---------- */
        .section-divider {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API Communication Helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> bool:
    """Check if the FastAPI backend is online."""
    try:
        response = requests.get(ENDPOINT_HEALTH, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def fetch_statistics() -> Dict[str, Any]:
    """Retrieve vector database statistics from the backend."""
    try:
        response = requests.get(ENDPOINT_STATS, timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return {"total_documents": 0, "total_chunks": 0}


def fetch_configuration() -> Optional[Dict[str, str]]:
    """Retrieve the active backend configuration (provider names only)."""
    try:
        response = requests.get(ENDPOINT_CONFIG, timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def execute_query(question: str, top_k: int, temperature: float) -> Dict[str, Any]:
    """
    Send a question to the RAG backend and return the response.

    Args:
        question: The user's natural-language question.
        top_k: Maximum number of retrieved context chunks.
        temperature: LLM sampling temperature.

    Returns:
        The JSON response body from the backend.

    Raises:
        requests.exceptions.RequestException: On any HTTP-level failure.
    """
    payload = {
        "question": question,
        "top_k": top_k,
        "temperature": temperature,
    }
    response = requests.post(ENDPOINT_QUERY, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def trigger_indexing(source_path: str) -> Dict[str, Any]:
    """Tell the backend to ingest a document at the given path."""
    payload = {"source": source_path}
    response = requests.post(ENDPOINT_INDEX, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def trigger_reset() -> bool:
    """Tell the backend to completely clear the vector index."""
    try:
        response = requests.post(ENDPOINT_RESET, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_latency(value: Any) -> str:
    """
    Safely format a latency value for display.

    Args:
        value: A numeric latency value, or None.

    Returns:
        Formatted string like '0.42s', or 'N/A' if the value is unavailable.
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "N/A"


def extract_api_error(error: requests.exceptions.RequestException) -> str:
    """
    Extract a user-friendly error message from an API exception.

    Args:
        error: The requests exception raised by a failed HTTP call.

    Returns:
        A descriptive error string.
    """
    if hasattr(error, "response") and error.response is not None:
        if error.response.status_code == 404:
            return (
                "No relevant documents found in the database. "
                "Try uploading more documents or rephrasing your question."
            )
        try:
            return error.response.json().get("detail", str(error))
        except Exception:
            pass
    return str(error)


def render_latency_bar(metrics: Dict[str, Any]) -> None:
    """
    Render a compact latency summary beneath an answer.

    Args:
        metrics: Dictionary containing retrieval_latency, generation_latency,
                 and total_latency values (any may be None).
    """
    retrieval = format_latency(metrics.get("retrieval_latency"))
    generation = format_latency(metrics.get("generation_latency"))
    total = format_latency(metrics.get("total_latency"))

    st.markdown(
        f"""<div class="latency-bar">
            <span>🔍 Retrieval: <strong>{retrieval}</strong></span>
            <span>🤖 Generation: <strong>{generation}</strong></span>
            <span>⏱️ Total: <strong>{total}</strong></span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_source_cards(sources: list) -> None:
    """
    Render retrieved source chunks as styled cards inside an expander.

    Args:
        sources: List of source dictionaries from the API response.
    """
    if not sources:
        return

    with st.expander(f"🔍 View Retrieved Sources ({len(sources)})", expanded=False):
        for idx, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            score = source.get("similarity_score", source.get("relevance_score", source.get("score", 0.0)))
            source_name = (
                source.get("filename")
                or metadata.get("filename")
                or metadata.get("source_file_name")
                or metadata.get("source")
                or "Unknown"
            )
            page_number = source.get("page_number", metadata.get("page_number", metadata.get("page")))
            content = source.get("content", "").strip()

            score_display = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            page_display = f" | Page {page_number}" if page_number is not None else ""

            st.markdown(
                f"""<div class="source-card">
                    <div class="source-header">📄 Source {idx}: {source_name}{page_display}</div>
                    <div class="source-score">Similarity: {score_display}</div>
                    <div class="source-content">{content[:500]}{'…' if len(content) > 500 else ''}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_header() -> None:
    """Render the branded application header."""
    st.markdown(
        """<div class="main-header">
            <h1>📚 ResearchRAG</h1>
            <p>Production-ready Retrieval-Augmented Generation — upload research papers and ask complex questions.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[int, float]:
    """
    Render the sidebar configuration panel.

    Returns:
        A tuple of (top_k, temperature) selected by the user.
    """
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("Adjust retrieval and generation parameters.")

        top_k = st.slider(
            "Top-K Retrievals",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of document chunks to retrieve as context.",
        )

        temperature = st.slider(
            "Generation Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Higher values → more creative; lower values → more deterministic.",
        )

        # ----- Backend Status -----
        st.divider()
        st.subheader("📊 System Status")

        is_online = check_backend_health()

        if is_online:
            st.markdown(
                '<span class="status-online">● Backend Online</span>',
                unsafe_allow_html=True,
            )
            stats = fetch_statistics()

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Documents", stats.get("total_documents", 0))
            with col_b:
                st.metric("Chunks", stats.get("total_chunks", 0))

            # Show active providers
            config = fetch_configuration()
            if config:
                with st.expander("🛠️ Active Providers"):
                    st.markdown(f"**LLM:** `{config.get('llm_provider', '—')}`")
                    st.markdown(f"**Embeddings:** `{config.get('embedding_provider', '—')}`")
                    st.markdown(f"**Vector Store:** `{config.get('vector_store', '—')}`")
        else:
            st.markdown(
                '<span class="status-offline">● Backend Offline</span>',
                unsafe_allow_html=True,
            )
            st.warning(f"Ensure the API is running at `{API_URL}`")

        # ----- Maintenance -----
        st.divider()
        st.subheader("🧹 Maintenance")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("Reset Index", type="primary", use_container_width=True):
                with st.spinner("Clearing database..."):
                    if trigger_reset():
                        st.toast("✅ Vector index cleared successfully.", icon="🗑️")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Failed to reset the vector database.")

    return top_k, temperature


def render_upload_section() -> None:
    """Render the drag-and-drop document upload interface."""
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to add it to the knowledge base.",
        type=["pdf"],
        help="The document will be chunked, embedded, and indexed immediately.",
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary"):
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            print(f"PDF saved: {file_path}", flush=True)

            with st.spinner(f"Indexing **{uploaded_file.name}**… This may take a moment."):
                try:
                    report = trigger_indexing(file_path)
                    st.success(
                        f"✅ Indexed **{report['indexed_documents']}** document(s) → "
                        f"**{report['indexed_chunks']}** chunks in "
                        f"**{report['processing_time']:.2f}s**."
                    )
                    st.toast("Document indexed successfully!", icon="📄")
                except requests.exceptions.RequestException as error:
                    st.error(f"Indexing failed: {extract_api_error(error)}")


def render_chat_interface(top_k: int, temperature: float) -> None:
    """
    Render the interactive chat history and input box.

    Args:
        top_k: Maximum number of context chunks to retrieve.
        temperature: LLM sampling temperature.
    """
    st.subheader("💬 Ask ResearchRAG")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg.get("sources"):
                render_source_cards(msg["sources"])

            # Show latency metrics for assistant messages
            if msg.get("metrics"):
                render_latency_bar(msg["metrics"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking… Retrieving context and generating response…"):
                try:
                    response_data = execute_query(prompt, top_k, temperature)

                    answer = response_data.get("answer", "No answer provided.")
                    sources = response_data.get("retrieved_sources", [])
                    timings = response_data.get("timings", {})
                    metrics = {
                        "retrieval_latency": timings.get("retrieval", response_data.get("retrieval_latency")),
                        "generation_latency": timings.get("generation", response_data.get("generation_latency")),
                        "total_latency": timings.get("total", response_data.get("total_latency")),
                    }

                    st.markdown(answer)
                    render_source_cards(sources)
                    render_latency_bar(metrics)

                    # Persist to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metrics": metrics,
                    })

                except requests.exceptions.RequestException as error:
                    st.error(f"Query failed: {extract_api_error(error)}")


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution function for the Streamlit application."""
    inject_custom_css()
    initialize_session_state()

    render_header()
    top_k, temperature = render_sidebar()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        render_upload_section()

    with col_right:
        render_chat_interface(top_k, temperature)


if __name__ == "__main__":
    main()
