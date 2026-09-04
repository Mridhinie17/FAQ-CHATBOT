import os
from pathlib import Path
# Base Directory: root of the faq-intent-rag-chatbot project
BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass



class Config:
    BASE_DIR: Path = BASE_DIR
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    MODELS_DIR: Path = BASE_DIR / "models"
    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CLASSIFIER_MODEL_NAME: str = "distilbert-base-uncased"

config = Config()
