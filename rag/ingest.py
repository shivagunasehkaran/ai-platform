"""Corpus ingestion and FAISS index building for RAG."""

import json
import logging
import re
import sys
from pathlib import Path

import faiss
import numpy as np

from shared.config import CHUNK_SIZE, CHUNK_OVERLAP, CORPUS_DIR
from shared.llm import embed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory for index and metadata
DATA_DIR = Path(__file__).parent / "data"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.
    
    Args:
        text: Input text to split.
        
    Returns:
        list[str]: List of sentences.
    """
    # Split on sentence boundaries (., !, ?) followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks by sentences.
    
    Args:
        text: Input text to chunk.
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between chunks.
        
    Returns:
        list[str]: List of text chunks.
    """
    sentences = split_sentences(text)
    
    if not sentences:
        return []
    
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size, save current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Calculate overlap: keep sentences from end that fit in overlap
            overlap_sentences: list[str] = []
            overlap_length = 0
            
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1  # +1 for space
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load all .txt files from corpus directory.
    
    Args:
        corpus_dir: Path to the corpus directory.
        
    Returns:
        list[dict]: List of chunk metadata with source, chunk_index, and text.
    """
    all_chunks: list[dict] = []
    txt_files = list(corpus_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No .txt files found in {corpus_dir}")
        return all_chunks
    
    for filepath in txt_files:
        logger.info(f"Processing {filepath.name}...")
        print(f"Processing {filepath.name}...")
        
        try:
            text = filepath.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": filepath.name,
                    "chunk_index": idx,
                    "text": chunk,
                })
            
            logger.debug(f"Created {len(chunks)} chunks from {filepath.name}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
    
    return all_chunks


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity with IndexFlatIP.
    
    Args:
        vectors: Array of vectors to normalize.
        
    Returns:
        np.ndarray: L2-normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def build_index(chunks: list[dict], batch_size: int = 500) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Build FAISS index from chunks with batched embedding.
    
    Args:
        chunks: List of chunk metadata with 'text' field.
        batch_size: Number of chunks to embed per batch.
        
    Returns:
        tuple: (FAISS index, chunk metadata list)
    """
    total_chunks = len(chunks)
    logger.info(f"Embedding {total_chunks} chunks in batches of {batch_size}...")
    print(f"Embedding {total_chunks} chunks in batches of {batch_size}...")
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Process in batches with progress
    all_embeddings: list[list[float]] = []
    num_batches = (total_chunks + batch_size - 1) // batch_size
    
    for i in range(0, total_chunks, batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i:i + batch_size]
        
        print(f"  Batch {batch_num}/{num_batches} ({len(batch_texts)} chunks)...")
        
        batch_embeddings = embed(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    print(f"Embedding complete. Building FAISS index...")
    
    # Convert to numpy array
    vectors = np.array(all_embeddings, dtype=np.float32)
    
    # Normalize for cosine similarity
    vectors = normalize_vectors(vectors)
    
    # Create FAISS index (Inner Product for cosine similarity on normalized vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(vectors)
    
    logger.info(f"Built FAISS index with {index.ntotal} vectors (dim={dimension})")
    
    return index, chunks


def save_index(index: faiss.IndexFlatIP, metadata: list[dict], output_dir: Path) -> None:
    """Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS index to save.
        metadata: Chunk metadata list.
        output_dir: Directory to save files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")


def ingest(corpus_dir: Path = CORPUS_DIR, output_dir: Path = DATA_DIR) -> None:
    """Run the full ingestion pipeline.
    
    Args:
        corpus_dir: Path to corpus directory with .txt files.
        output_dir: Path to output directory for index and metadata.
    """
    # Check corpus directory exists
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        print(f"❌ Corpus directory not found: {corpus_dir}")
        print("Please create the directory and add .txt files.")
        sys.exit(1)
    
    # Load and chunk corpus
    chunks = load_corpus(corpus_dir)
    
    if not chunks:
        logger.error("No chunks created. Check corpus directory.")
        print("❌ No chunks created. Add .txt files to corpus/")
        sys.exit(1)
    
    # Count unique files
    unique_files = len(set(chunk["source"] for chunk in chunks))
    
    # Build index
    index, metadata = build_index(chunks)
    
    # Save outputs
    save_index(index, metadata, output_dir)
    
    # Print summary
    print(f"\n✅ Indexed {len(chunks)} chunks from {unique_files} files")
    print(f"   Index saved to: {output_dir / 'index.faiss'}")
    print(f"   Metadata saved to: {output_dir / 'metadata.json'}")


def main() -> None:
    """Entry point for ingestion CLI."""
    print("🔄 Starting corpus ingestion...\n")
    ingest()


if __name__ == "__main__":
    main()
