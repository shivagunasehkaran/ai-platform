"""Query retrieval and answer generation for RAG."""

import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

from shared.llm import embed, chat_with_usage
from shared.telemetry import Timer, calculate_cost, format_stats, metrics_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent / "data"

# Cached index and metadata
_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] | None = None


# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question based ONLY on the context provided above.
- If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."
- Cite your sources using [1], [2], etc. corresponding to the context numbers.
- Be concise and accurate.

QUESTION: {question}

ANSWER:"""


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load FAISS index and metadata from disk.
    
    Returns:
        tuple: (FAISS index, metadata list)
        
    Raises:
        FileNotFoundError: If index or metadata files don't exist.
    """
    global _index, _metadata
    
    # Return cached if available
    if _index is not None and _metadata is not None:
        return _index, _metadata
    
    index_path = DATA_DIR / "index.faiss"
    metadata_path = DATA_DIR / "metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load index
    _index = faiss.read_index(str(index_path))
    
    # Set nprobe for IVF index (if applicable)
    if hasattr(_index, 'nprobe'):
        _index.nprobe = 10  # Search 10 clusters for good accuracy/speed balance
        logger.info(f"Set nprobe=10 for IVF index")
    
    logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    logger.info(f"Loaded {len(_metadata)} metadata entries")
    
    return _index, _metadata


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a single vector for cosine similarity.
    
    Args:
        vector: Vector to normalize.
        
    Returns:
        np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve most relevant chunks for a query.
    
    Args:
        query: The search query.
        top_k: Number of results to return.
        
    Returns:
        list[dict]: Top-k results with text, source, and score.
    """
    # Load index
    index, metadata = load_index()
    
    # Embed query (timed separately)
    with Timer() as embed_timer:
        query_embedding = embed([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)
        query_vector = normalize_vector(query_vector.reshape(1, -1))
    
    # FAISS search (timed separately)
    with Timer() as search_timer:
        scores, indices = index.search(query_vector, top_k)
    
    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        
        chunk = metadata[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "score": float(score),
            "embed_latency_ms": embed_timer.elapsed_ms,
            "search_latency_ms": search_timer.elapsed_ms,
        })
    
    total_latency = embed_timer.elapsed_ms + search_timer.elapsed_ms
    
    logger.info(
        f"Retrieved {len(results)} chunks - "
        f"embed: {embed_timer.elapsed_ms:.0f}ms, "
        f"search: {search_timer.elapsed_ms:.0f}ms, "
        f"total: {total_latency:.0f}ms"
    )
    
    # Log metrics (using FAISS search time as retrieval latency)
    metrics_store.log_retrieval_metrics(
        query=query,
        latency_ms=search_timer.elapsed_ms,  # FAISS search only
        recall=0.0,  # Placeholder - calculated during evaluation
        mrr=0.0,     # Placeholder - calculated during evaluation
    )
    
    return results


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks.
    
    Args:
        chunks: List of retrieved chunks.
        
    Returns:
        str: Formatted context with numbered sources.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (Source: {chunk['source']})\n{chunk['text']}")
    
    return "\n\n".join(context_parts)


def answer_question(query: str, top_k: int = 5) -> dict:
    """Answer a question using RAG.
    
    Args:
        query: The question to answer.
        top_k: Number of chunks to retrieve.
        
    Returns:
        dict: Answer with citations and latency metrics.
    """
    with Timer() as total_timer:
        # Retrieve relevant chunks
        chunks = retrieve(query, top_k=top_k)
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "citations": [],
                "embed_latency_ms": 0,
                "search_latency_ms": 0,
                "retrieval_latency_ms": 0,
                "total_latency_ms": 0,
            }
        
        # Extract latencies from first chunk
        embed_latency = chunks[0].get("embed_latency_ms", 0)
        search_latency = chunks[0].get("search_latency_ms", 0)
        
        # Build context and prompt
        context = build_context(chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        
        # Generate answer
        messages = [{"role": "user", "content": prompt}]
        answer, usage = chat_with_usage(messages)
        
        # Calculate cost
        cost = calculate_cost(usage["prompt_tokens"], usage["completion_tokens"])
        
        # Build citations
        citations = [
            {"source": chunk["source"], "text": chunk["text"][:200] + "..."}
            for chunk in chunks
        ]
    
    # Log stats
    stats = format_stats(
        usage["prompt_tokens"],
        usage["completion_tokens"],
        cost,
        total_timer.elapsed_ms,
    )
    logger.info(stats)
    
    return {
        "answer": answer,
        "citations": citations,
        "embed_latency_ms": embed_latency,
        "search_latency_ms": search_latency,
        "retrieval_latency_ms": embed_latency + search_latency,
        "total_latency_ms": total_timer.elapsed_ms,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cost": cost,
    }


def main() -> None:
    """CLI entry point for testing retrieval."""
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieve \"<question>\"")
        print("Example: python -m rag.retrieve \"What is machine learning?\"")
        sys.exit(1)
    
    query = sys.argv[1]
    
    print(f"\n🔍 Question: {query}\n")
    print("-" * 60)
    
    try:
        result = answer_question(query)
        
        print(f"\n📝 Answer:\n{result['answer']}\n")
        print("-" * 60)
        
        print("\n📚 Sources:")
        for i, citation in enumerate(result["citations"], 1):
            print(f"  [{i}] {citation['source']}")
            print(f"      {citation['text'][:100]}...")
        
        print("-" * 60)
        print(f"\n⏱️  Embed query: {result['embed_latency_ms']:.0f}ms")
        print(f"⏱️  FAISS search: {result['search_latency_ms']:.0f}ms  ← (vector DB retrieval)")
        print(f"⏱️  Total retrieval: {result['retrieval_latency_ms']:.0f}ms")
        print(f"⏱️  Total (incl. LLM): {result['total_latency_ms']:.0f}ms")
        print(f"💰 Cost: ${result['cost']:.6f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Run 'python -m rag.ingest' first to build the index.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
