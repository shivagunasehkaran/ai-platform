"""FastAPI endpoint for RAG question answering."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.retrieve import load_index, answer_question

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Request/Response models
class QueryRequest(BaseModel):
    """Request body for QA endpoint."""
    query: str = Field(..., min_length=1, description="The question to answer")


class Citation(BaseModel):
    """Citation source in response."""
    source: str
    text: str


class QueryResponse(BaseModel):
    """Response body for QA endpoint."""
    answer: str
    citations: list[Citation]
    retrieval_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    index_loaded: bool
    num_vectors: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to load FAISS index on startup."""
    logger.info("Starting RAG API server...")
    
    try:
        index, metadata = load_index()
        app.state.index_loaded = True
        app.state.num_vectors = index.ntotal
        logger.info(f"FAISS index loaded with {index.ntotal} vectors")
    except FileNotFoundError as e:
        logger.warning(f"Index not found: {e}")
        app.state.index_loaded = False
        app.state.num_vectors = 0
    
    yield
    
    logger.info("Shutting down RAG API server...")


# Create FastAPI app
app = FastAPI(
    title="RAG Question Answering API",
    description="Retrieval-Augmented Generation API for answering questions from a document corpus",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health and index status."""
    return HealthResponse(
        status="healthy",
        index_loaded=app.state.index_loaded,
        num_vectors=app.state.num_vectors,
    )


@app.post("/qa", response_model=QueryResponse, tags=["QA"])
async def question_answer(request: QueryRequest) -> QueryResponse:
    """Answer a question using RAG.
    
    Args:
        request: Query request with the question.
        
    Returns:
        QueryResponse: Answer with citations and latency metrics.
        
    Raises:
        HTTPException: 400 for empty query, 500 for internal errors.
    """
    # Validate query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Check index is loaded
    if not app.state.index_loaded:
        raise HTTPException(
            status_code=500,
            detail="FAISS index not loaded. Run 'python -m rag.ingest' first.",
        )
    
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Get answer using RAG
        result = answer_question(request.query)
        
        # Build response
        citations = [
            Citation(source=c["source"], text=c["text"])
            for c in result["citations"]
        ]
        
        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            retrieval_latency_ms=result["retrieval_latency_ms"],
            total_latency_ms=result["total_latency_ms"],
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
