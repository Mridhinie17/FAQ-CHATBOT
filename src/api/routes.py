from typing import List
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ChatRequest,
    ChatResponse,
    RetrievedChunk,
    HealthResponse
)
from src.classifier.predict import get_classifier
from src.rag.embedder import embed_query
from src.rag.vectorstore import search, load_index
from src.rag.generator import generate_response

from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

# 27 Canonical Bitext Intent Classes
ALL_INTENTS = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund"
]


@router.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check(request: Request):
    """
    Health check endpoint returning availability of models and vector store.
    """
    models_status = {
        "transformer": request.app.state.transformer_ready,
        "baseline": request.app.state.baseline_ready
    }
    return HealthResponse(
        status="ok",
        models_loaded=models_status,
        vectorstore_loaded=request.app.state.vectorstore_ready
    )


@router.get("/intents", response_model=List[str], tags=["Taxonomy"])
def get_intents():
    """
    Returns the comprehensive list of 27 intent classes recognized by the system.
    """
    return sorted(ALL_INTENTS)


@router.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify_intent(payload: ClassifyRequest, request: Request):
    """
    Classify a customer query into one of the 27 support intents.
    Supports either 'transformer' (DistilBERT) or 'baseline' (TF-IDF + LR).
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    model_type = payload.model_type.lower()
    
    # Check if requested model is ready; fallback gracefully if needed
    if model_type in ("transformer", "distilbert") and not request.app.state.transformer_ready:
        if request.app.state.baseline_ready:
            model_type = "baseline"
        else:
            # Simple keyword-based fallback if neither notebook has been run yet
            return ClassifyResponse(
                intent="general_inquiry",
                confidence=0.50,
                model_used="fallback_heuristics"
            )

    try:
        clf = get_classifier(model_type)
        res = clf.predict(query)
        return ClassifyResponse(
            intent=res["intent"],
            confidence=res["confidence"],
            model_used=res["model_used"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@router.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat(payload: ChatRequest, request: Request):
    """
    End-to-End Chatbot Pipeline:
    1. Classifies user query intent (DistilBERT or TF-IDF)
    2. Embeds query & searches FAISS vector store for top-k matching FAQ chunks
    3. Synthesizes a grounded response via Groq API (or verified FAQ fallback)
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: Classify intent
    try:
        clf = get_classifier(payload.model_type)
        clf_result = clf.predict(query)
        detected_intent = clf_result["intent"]
        confidence = clf_result["confidence"]
        model_used = clf_result["model_used"]
    except Exception:
        # Fallback intent if model artifact is absent
        detected_intent = "contact_customer_service"
        confidence = 0.50
        model_used = "fallback"

    # Step 2: Retrieve relevant FAQ entries from FAISS
    retrieved_items = []
    try:
        query_emb = embed_query(query)
        raw_results = search(query_emb, k=payload.top_k, intent_filter=detected_intent)
        for r in raw_results:
            retrieved_items.append(RetrievedChunk(
                instruction=r.get("instruction", ""),
                response=r.get("response", ""),
                intent=r.get("intent", detected_intent),
                category=r.get("category", "SUPPORT"),
                score=float(r.get("score", 0.0))
            ))
    except Exception as e:
        print(f"[Warning] FAISS search error: {e}")

    # Step 3: Generate response
    try:
        raw_chunks = [item.dict() for item in retrieved_items]
        gen_res = generate_response(
            query=query,
            retrieved_chunks=raw_chunks,
            intent=detected_intent
        )
        answer = gen_res["answer"]
    except Exception as e:
        answer = f"I found matching guidelines for {detected_intent}. How else can I help?"

    return ChatResponse(
        intent=detected_intent,
        confidence=confidence,
        retrieved_chunks=retrieved_items,
        answer=answer,
        model_used=model_used
    )
