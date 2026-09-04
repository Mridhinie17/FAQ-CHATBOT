"""
Build and persist the FAISS index and metadata store from data/raw/bitext_support.csv
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.rag.embedder import embed_texts
from src.rag.vectorstore import build_index

def main():
    print("=== Building Vector Store Knowledge Index ===")
    csv_path = config.DATA_RAW_DIR / "bitext_support.csv"
    if not csv_path.exists():
        csv_fallback = Path("C:/Users/111196/Desktop/CHATBOT/data/raw/bitext_support.csv")
        if csv_fallback.exists():
            csv_path = csv_fallback
        else:
            raise FileNotFoundError(f"Missing dataset at {csv_path}. Run scripts/download_data.py first.")

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total raw records: {len(df):,}")

    # Deduplicate representative question-response pairs for indexing efficiency
    # For a high-quality FAQ vector base, we can index unique responses or sample representative queries
    faq_df = df.drop_duplicates(subset=["instruction", "intent"]).copy()
    print(f"Indexing {len(faq_df):,} distinct customer support entries...")

    # Formulate documents to embed: Question + Response pairs
    text_corpus = [
        f"Intent: {row['intent']}. Customer Inquiry: {row['instruction']}. Resolution: {row['response']}"
        for _, row in faq_df.iterrows()
    ]

    metadata = [
        {
            "id": idx,
            "instruction": row["instruction"],
            "response": row["response"],
            "intent": row["intent"],
            "category": row["category"]
        }
        for idx, (_, row) in enumerate(faq_df.iterrows())
    ]

    print("Generating dense vector embeddings with SentenceTransformers...")
    batch_size = 512
    all_embeddings = []
    
    for start_idx in range(0, len(text_corpus), batch_size):
        end_idx = min(start_idx + batch_size, len(text_corpus))
        batch = text_corpus[start_idx:end_idx]
        batch_embs = embed_texts(batch)
        all_embeddings.append(batch_embs)
        print(f"  Processed {end_idx:,} / {len(text_corpus):,} embeddings...", end="\r")

    embeddings = np.vstack(all_embeddings)
    print(f"\nCompleted embeddings generation. Matrix shape: {embeddings.shape}")

    # Build and save index to vectorstore/
    build_index(embeddings, metadata, output_dir=config.VECTORSTORE_DIR)
    
    # Also save to Desktop/CHATBOT/vectorstore if directory exists
    desktop_vs = Path("C:/Users/111196/Desktop/CHATBOT/vectorstore")
    if desktop_vs.parent.exists():
        build_index(embeddings, metadata, output_dir=desktop_vs)

    print("=== Vector Store Indexing Finished Successfully! ===")

if __name__ == "__main__":
    main()
