import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.api.routes import router
from src.classifier.predict import get_classifier
from src.rag.vectorstore import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & Shutdown lifecycle hook.
    Pre-warms and verifies classifier models and FAISS vector index on startup.
    """
    print("--- Initializing FAQ Intent Classifier & RAG Chatbot Service ---")
    
    # Check baseline classifier
    try:
        clf_b = get_classifier("baseline")
        app.state.baseline_ready = clf_b._is_loaded
        print(" [OK] Baseline TF-IDF Classifier: Loaded")
    except Exception as e:
        app.state.baseline_ready = False
        print(f" [!] Baseline Classifier not found or uninitialized: {e}")

    # Check transformer classifier
    try:
        clf_t = get_classifier("transformer")
        app.state.transformer_ready = clf_t._is_loaded
        print(" [OK] DistilBERT Transformer Classifier: Loaded")
    except Exception as e:
        app.state.transformer_ready = False
        print(f" [!] DistilBERT Classifier not yet exported: {e}")

    # Check vector store
    try:
        load_index()
        app.state.vectorstore_ready = True
        print(" [OK] FAISS Vector Store: Loaded & Ready")
    except Exception as e:
        app.state.vectorstore_ready = False
        print(f" [!] FAISS Vector Store not yet loaded: {e}")

    print("--- API Service Ready to Accept Requests ---")
    yield
    print("--- Shutting down API Service ---")


app = FastAPI(
    title="FAQ Intent Classifier + RAG Chatbot API",
    description=(
        "Production backend for customer support ticket classification and RAG response generation. "
        "Provides endpoints to classify intent, retrieve matching knowledge base chunks, and synthesize answers."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
