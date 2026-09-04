from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import config

# Global embedding model instance
_embedder_model = None

def get_embedder():
    global _embedder_model
    if _embedder_model is None:
        print(f"Loading SentenceTransformer embedding model: {config.EMBEDDING_MODEL_NAME}...")
        _embedder_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate dense vector embeddings for a list of document strings.
    Returns:
        numpy.ndarray of shape (len(texts), embedding_dim)
    """
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Generate a normalized vector embedding for a single user query.
    Returns:
        numpy.ndarray of shape (1, embedding_dim)
    """
    model = get_embedder()
    emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)
