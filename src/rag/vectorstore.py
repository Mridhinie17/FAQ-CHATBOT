import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from src.config import config

_faiss_index = None
_metadata_store = None

def build_index(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build FAISS inner-product index (cosine similarity on normalized vectors) and persist to disk.
    """
    out_dir = output_dir or config.VECTORSTORE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    # Inner product on normalized vectors equals cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index & metadata
    index_file = str(out_dir / "faiss_index.bin")
    meta_file = str(out_dir / "metadata.pkl")

    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index with {index.ntotal} items written to: {index_file}")
    return index, metadata


def load_index(index_dir: Optional[Path] = None) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load saved FAISS index and metadata store.
    """
    global _faiss_index, _metadata_store
    if _faiss_index is not None and _metadata_store is not None:
        return _faiss_index, _metadata_store

    in_dir = index_dir or config.VECTORSTORE_DIR
    index_file = in_dir / "faiss_index.bin"
    meta_file = in_dir / "metadata.pkl"

    if not (index_file.exists() and meta_file.exists()):
        raise FileNotFoundError(
            f"FAISS vector store files not found in {in_dir}. "
            "Please run scripts/build_vectorstore.py to create the index."
        )

    _faiss_index = faiss.read_index(str(index_file))
    with open(meta_file, "rb") as f:
        _metadata_store = pickle.load(f)

    return _faiss_index, _metadata_store


def search(
    query_embedding: np.ndarray,
    k: int = 3,
    intent_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search top-k most relevant FAQ chunks.
    Args:
        query_embedding: shape (1, dim) or (dim,)
        k: number of chunks to return
        intent_filter: optional intent filter to boost domain alignment
    """
    index, metadata = load_index()
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # If an intent filter is specified, fetch more candidates and filter
    fetch_k = k * 4 if intent_filter else k
    scores, indices = index.search(query_embedding, min(fetch_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["score"] = float(round(score, 4))
        
        if intent_filter and item.get("intent") != intent_filter:
            continue
            
        results.append(item)
        if len(results) == k:
            break

    # Fallback if filtered results were too few
    if len(results) < k and intent_filter:
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx].copy()
            item["score"] = float(round(score, 4))
            if item not in results:
                results.append(item)
            if len(results) == k:
                break

    return results
