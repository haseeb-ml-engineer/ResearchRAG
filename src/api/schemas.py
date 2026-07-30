from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    temperature: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: List[Dict[str, Any]]
    similarity_scores: List[float] = []
    metadata: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}
    citations: List[str] = []
    confidence: Optional[float] = None
    retrieval_latency: float
    generation_latency: float
    total_latency: float

class IndexRequest(BaseModel):
    source: str

class IndexResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    processing_time: float
    failures: int = 0

class ConfigResponse(BaseModel):
    llm_provider: str
    embedding_provider: str
    vector_store: str

class StatisticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
