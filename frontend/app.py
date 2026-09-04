import streamlit as pd_st
import streamlit as st
import requests
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="FAQ Support AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# --- 2. Custom Modern Styling (Dark Glassmorphism Theme) ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Header hero block */
    .hero-container {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* Intent Badge Styles */
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-mid {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Context reference card */
    .reference-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #38bdf8;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px 8px 8px 4px;
    }
    .reference-header {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    # Model Selector
    model_choice = st.radio(
        "Intent Classifier Architecture:",
        ("DistilBERT (Fine-Tuned)", "TF-IDF + Logistic Regression"),
        index=0,
        help="Select which intent classification pipeline guides the FAQ retrieval."
    )
    model_param = "transformer" if "DistilBERT" in model_choice else "baseline"
    
    top_k_chunks = st.slider(
        "FAQ Context Depth (Top-K Chunks):",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of nearest-neighbor FAQ knowledge chunks retrieved via FAISS."
    )

    st.markdown("---")
    
    # Backend Health Indicator
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if health_resp.status_code == 200:
            st.success("🟢 Backend API: Online", icon="✅")
        else:
            st.warning("🟡 Backend API: Degraded Status")
    except requests.exceptions.RequestException:
        st.error("🔴 Backend API: Offline\n\nStart backend with: `python -m uvicorn src.api.main:app --port 8000`", icon="⚠️")

    st.markdown("---")

    # Clear Chat Button
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Expandable About Drawer
    with st.expander("ℹ️ Architecture & Transparency"):
        st.markdown("""
        **Pipeline Flow:**
        1. **User Query** is received by Streamlit UI.
        2. **FastAPI (`/chat`)** routes text to the selected classifier (**DistilBERT** or **TF-IDF**).
        3. **Query Embedding** is computed using `all-MiniLM-L6-v2`.
        4. **FAISS Vector Store** executes cosine similarity search over 24,635 FAQ knowledge records.
        5. **Response Synthesis** grounds generation using Groq (`llama-3.1-8b-instant`) or verified FAQ templates.
        """)

# --- 4. Main Chat Header ---
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Support Ticket Intent Classifier + RAG Chatbot</h1>
    <div class="hero-subtitle">Interactive AI Customer Service with Real-Time Intent & Knowledge Retrieval Transparency</div>
</div>
""", unsafe_allow_html=True)

# --- 5. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI Support Assistant. I can help with order tracking, cancellation, refunds, shipping address updates, account settings, and more. How can I assist you today?",
            "details": None
        }
    ]

# --- 6. Render Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant response contains pipeline details, display transparent expander
        if msg.get("details"):
            details = msg["details"]
            intent = details.get("intent", "Unknown")
            conf = float(details.get("confidence", 0.0))
            chunks = details.get("retrieved_chunks", [])
            model_used = details.get("model_used", model_param)
            
            # Choose badge color based on confidence score
            if conf >= 0.80:
                badge_class = "badge-high"
            elif conf >= 0.50:
                badge_class = "badge-mid"
            else:
                badge_class = "badge-low"

            with st.expander("🔍 Pipeline Transparency Details", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**Predicted Intent:** <span class='intent-badge {badge_class}'>{intent} ({conf*100:.1f}%)</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**Classification Model:** `{model_used}`")
                
                st.markdown(f"**Retrieved Knowledge Chunks ({len(chunks)} items):**")
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.get("score", 0.0)
                    st.markdown(f"""
                    <div class="reference-card">
                        <div class="reference-header">Reference #{i} | Cosine Similarity: {score:.4f} | Intent: {chunk.get('intent')}</div>
                        <b>Matched Query:</b> {chunk.get('instruction')}<br>
                        <b>Resolution:</b> {chunk.get('response')[:250]}...
                    </div>
                    """, unsafe_allow_html=True)

# --- 7. Chat Input & Processing ---
user_query = st.chat_input("Ask a question (e.g. 'I want to cancel my order', 'Check refund status')...")

if user_query:
    # Display user query immediately
    st.session_state.messages.append({"role": "user", "content": user_query, "details": None})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Classifying intent & retrieving FAQ knowledge..."):
            try:
                payload = {
                    "query": user_query,
                    "model_type": model_param,
                    "top_k": top_k_chunks
                }
                response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    details = {
                        "intent": data.get("intent"),
                        "confidence": data.get("confidence"),
                        "retrieved_chunks": data.get("retrieved_chunks", []),
                        "model_used": data.get("model_used")
                    }
                    
                    st.markdown(answer)
                    
                    # Render pipeline details
                    conf = float(details["confidence"])
                    badge_class = "badge-high" if conf >= 0.80 else ("badge-mid" if conf >= 0.50 else "badge-low")
                    
                    with st.expander("🔍 Pipeline Transparency Details", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(f"**Predicted Intent:** <span class='intent-badge {badge_class}'>{details['intent']} ({conf*100:.1f}%)</span>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"**Classification Model:** `{details['model_used']}`")
                        
                        st.markdown(f"**Retrieved Knowledge Chunks ({len(details['retrieved_chunks'])} items):**")
                        for i, chunk in enumerate(details["retrieved_chunks"], 1):
                            st.markdown(f"""
                            <div class="reference-card">
                                <div class="reference-header">Reference #{i} | Cosine Similarity: {chunk.get('score', 0.0):.4f} | Intent: {chunk.get('intent')}</div>
                                <b>Matched Query:</b> {chunk.get('instruction')}<br>
                                <b>Resolution:</b> {chunk.get('response')[:250]}...
                            </div>
                            """, unsafe_allow_html=True)

                    # Persist message to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "details": details
                    })
                else:
                    err_msg = f"⚠️ Backend returned error (Status {response.status_code}): {response.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg, "details": None})

            except requests.exceptions.RequestException as e:
                err_msg = (
                    "⚠️ **Could not connect to FastAPI backend!**\n\n"
                    f"Ensure the API service is running on `{API_BASE_URL}`:\n"
                    "```powershell\n"
                    "C:\\chatbot_env\\Scripts\\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload\n"
                    "```"
                )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "details": None})
