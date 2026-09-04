import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import config


class BaselineClassifier:
    """
    TF-IDF + Logistic Regression / LinearSVC classifier wrapper.
    Loads vectorizer, trained model, and label encoder from models/tfidf_logreg/
    """
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or config.MODELS_DIR / "tfidf_logreg"
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self._is_loaded = False

    def load(self) -> None:
        """Load serialized artifacts from disk."""
        vec_path = self.model_dir / "tfidf_vectorizer.joblib"
        model_path = self.model_dir / "classifier.joblib"
        le_path = self.model_dir / "label_encoder.joblib"

        if not (vec_path.exists() and model_path.exists() and le_path.exists()):
            raise FileNotFoundError(
                f"Missing baseline artifacts in {self.model_dir}. "
                "Ensure notebooks/03_baseline_classifier.ipynb has been executed."
            )

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(le_path)
        self._is_loaded = True

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the intent of a given input text.
        Returns:
            {"intent": str, "confidence": float, "model_used": "baseline_tfidf"}
        """
        if not self._is_loaded:
            self.load()

        vec = self.vectorizer.transform([text])
        
        # Calculate probabilities or decision scores
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])
        else:
            # e.g. LinearSVC decision function normalized with softmax
            decision = self.model.decision_function(vec)[0]
            exp_scores = np.exp(decision - np.max(decision))
            probs = exp_scores / exp_scores.sum()
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])

        intent_name = self.label_encoder.inverse_transform([pred_idx])[0]
        return {
            "intent": intent_name,
            "confidence": round(confidence, 4),
            "model_used": "baseline_tfidf"
        }
