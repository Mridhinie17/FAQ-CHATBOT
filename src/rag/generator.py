import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.config import config

# Ensure .env is refreshed
load_dotenv(config.BASE_DIR / ".env")

def generate_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    intent: str,
    model_name: str = "qwen/qwen3.8-27b"
) -> Dict[str, Any]:
    """
    Synthesize an answer using Groq API conditioned on retrieved knowledge chunks and detected intent.
    Falls back gracefully to the top retrieved response template if API key is unconfigured.
    """
    api_key = os.getenv("GROQ_API_KEY", config.GROQ_API_KEY)
    
    context_blocks = []
    sources = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"Reference {i} (Intent: {chunk.get('intent', 'General')}):\n"
            f"Question/Prompt: {chunk.get('instruction', '')}\n"
            f"Official Resolution: {chunk.get('response', chunk.get('text', ''))}"
        )
        sources.append({
            "intent": chunk.get("intent"),
            "category": chunk.get("category"),
            "score": chunk.get("score")
        })

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an empathetic, concise, and highly accurate Customer Support AI Assistant.\n"
        f"The user query has been classified under the intent: '{intent}'.\n"
        "Answer the user query accurately based on the provided reference context.\n"
        "Guidelines:\n"
        "- Adopt a polite and professional tone.\n"
        "- If customer template placeholders like {{Order Number}} appear, guide the customer clearly on how to provide that info.\n"
        "- Do not make up internal policies not reflected in the references."
    )

    user_prompt = (
        f"Context References:\n{context_str}\n\n"
        f"Customer Query: {query}\n\n"
        "Provide your official response:"
    )

    # Check for valid Groq API key
    if not api_key or api_key in ("your_key_here", "your_groq_api_key_here"):
        # Graceful fallback: return highest ranked verified FAQ resolution
        top_response = (
            retrieved_chunks[0].get("response")
            if retrieved_chunks
            else "Thank you for reaching out. Please provide your order or account details so we can assist you."
        )
        return {
            "answer": f"{top_response}\n\n*(Note: Generated via indexed verified FAQ template. Set a valid GROQ_API_KEY in .env to activate dynamic LLM reasoning)*",
            "sources": sources
        }

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        answer = completion.choices[0].message.content.strip()
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        top_response = retrieved_chunks[0].get("response") if retrieved_chunks else ""
        return {
            "answer": f"{top_response}\n\n*(Notice: LLM call encountered: {e}. Showing closest matching FAQ answer)*",
            "sources": sources
        }
