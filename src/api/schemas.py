from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    query: str = Field(..., example="I want to cancel my order 12345")
    model_type: Optional[str] = Field("transformer", example="transformer", description="'transformer' or 'baseline'")


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    model_used: str


class RetrievedChunk(BaseModel):
    instruction: str
    response: str
    intent: str
    category: str
    score: float


class ChatRequest(BaseModel):
    query: str = Field(..., example="How do I get a refund on my last purchase?")
    model_type: Optional[str] = Field("transformer", example="transformer", description="'transformer' or 'baseline'")
    top_k: Optional[int] = Field(3, example=3, description="Number of FAQ chunks to retrieve")


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    model_used: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    vectorstore_loaded: bool
