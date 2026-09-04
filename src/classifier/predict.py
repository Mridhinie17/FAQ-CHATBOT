from typing import Dict, Any, Union
from src.classifier.baseline import BaselineClassifier
from src.classifier.transformer import TransformerClassifier

# Singletons cache
_classifiers = {}

def get_classifier(model_type: str = "transformer") -> Union[TransformerClassifier, BaselineClassifier]:
    """
    Factory function to retrieve cached classifier instance.
    Args:
        model_type: "transformer" or "baseline"
    """
    model_type = model_type.lower()
    if model_type not in _classifiers:
        if model_type in ("transformer", "distilbert"):
            clf = TransformerClassifier()
            try:
                clf.load()
            except Exception as e:
                print(f"[Warning] Could not load transformer classifier: {e}")
            _classifiers[model_type] = clf
        elif model_type in ("baseline", "tfidf"):
            clf = BaselineClassifier()
            try:
                clf.load()
            except Exception as e:
                print(f"[Warning] Could not load baseline classifier: {e}")
            _classifiers[model_type] = clf
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'transformer' or 'baseline'.")
            
    return _classifiers[model_type]


def classify(text: str, model_type: str = "transformer") -> Dict[str, Any]:
    """
    Convenience wrapper to classify query using specified architecture.
    """
    clf = get_classifier(model_type)
    return clf.predict(text)
