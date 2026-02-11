"""Convert FAISS IndexFlatIP to IndexIVFFlat for faster search."""

import logging
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def convert_to_ivf(nlist: int = 100, nprobe: int = 10) -> None:
    """Convert existing flat index to IVF index.
    
    Args:
        nlist: Number of clusters for IVF.
        nprobe: Number of clusters to search (higher = more accurate, slower).
    """
    index_path = DATA_DIR / "index.faiss"
    
    # Load existing index
    logger.info("Loading existing index...")
    flat_index = faiss.read_index(str(index_path))
    
    n_vectors = flat_index.ntotal
    dimension = flat_index.d
    logger.info(f"Loaded index: {n_vectors} vectors, dim={dimension}")
    
    # Extract vectors from flat index
    logger.info("Extracting vectors...")
    vectors = flat_index.reconstruct_n(0, n_vectors)
    
    # Create IVF index
    logger.info(f"Creating IVF index (nlist={nlist}, nprobe={nprobe})...")
    quantizer = faiss.IndexFlatIP(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index
    logger.info("Training IVF index...")
    ivf_index.train(vectors)
    
    # Add vectors
    logger.info("Adding vectors to IVF index...")
    ivf_index.add(vectors)
    
    # Set nprobe for search
    ivf_index.nprobe = nprobe
    
    # Backup old index and save new one
    backup_path = DATA_DIR / "index_flat_backup.faiss"
    logger.info(f"Backing up old index to {backup_path}")
    faiss.write_index(flat_index, str(backup_path))
    
    logger.info(f"Saving IVF index to {index_path}")
    faiss.write_index(ivf_index, str(index_path))
    
    logger.info("✅ Conversion complete!")
    print(f"\n✅ Converted to IVF index")
    print(f"   Vectors: {n_vectors}")
    print(f"   Clusters (nlist): {nlist}")
    print(f"   Search clusters (nprobe): {nprobe}")


if __name__ == "__main__":
    convert_to_ivf()
