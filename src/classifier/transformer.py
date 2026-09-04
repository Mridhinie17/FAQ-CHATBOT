import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import config


class TransformerClassifier:
    """
    DistilBERT classifier wrapper.
    Loads fine-tuned model and tokenizer from models/distilbert_intent/
    """
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or config.MODELS_DIR / "distilbert_intent"
        self.tokenizer = None
        self.model = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False

    def load(self) -> None:
        """Load tokenizer, model weights and label mappings."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT model directory not found: {self.model_dir}. "
                "Ensure notebooks/04_distilbert_finetuning.ipynb has exported the model."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Load ID to Label mapping
        label_map_path = self.model_dir / "label_mapping.joblib"
        if label_map_path.exists():
            data = joblib.load(label_map_path)
            self.id2label = data.get("id2label", self.model.config.id2label)
        else:
            self.id2label = self.model.config.id2label

        self._is_loaded = True

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on user input query.
        Returns:
            {"intent": str, "confidence": float, "model_used": "distilbert"}
        """
        if not self._is_loaded:
            self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)[0]
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

        idx = predicted_class_idx.item()
        intent_name = self.id2label.get(idx, self.id2label.get(str(idx), f"class_{idx}"))

        return {
            "intent": intent_name,
            "confidence": round(float(confidence.item()), 4),
            "model_used": "distilbert"
        }
