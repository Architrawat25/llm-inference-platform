from typing import Any
from routers.embedder import Embedder, EmbeddingError
import numpy as np

class RoutingError(Exception):
    pass

class SemanticRouter:

    def __init__(self, embedder: Embedder, intent_embeddings: dict[str, dict[str, Any]] ,similarity_threshold: float = 0.6, default_model: str = "small"):
        self.embedder = embedder
        self.intent_embeddings = intent_embeddings
        self.similarity_threshold = similarity_threshold
        self.default_model = default_model

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Since vecs are normalized multiply and sum
        Result: 1.0 means identical, 0.0 means unrelated, -1.0 means opposite.
        """
        return float(np.dot(vec1, vec2))

    def route(self, prompt: str) -> dict[str, Any]:
        try:
            prompt_embedding = self.embedder.embed(prompt)
        except EmbeddingError:
            #fallback to default model
            return {
                "model": self.default_model,
                "intent": None,
                "score": 0.0
            }

        best_intent = None
        best_score = -1
        best_model = None

        for intent_name, data in self.intent_embeddings.items():
            intent_embedding = data["embedding"]

            score = self.cosine_similarity(prompt_embedding, intent_embedding)

            if score > best_score:
                best_score = score
                best_intent = intent_name
                best_model = data["target_model"]

            #fallback to default model
            if best_score < self.similarity_threshold:
                return {
                    "model": self.default_model,
                    "intent": None,
                    "score": best_score,
                }

        return {
            "model": best_model,
            "intent": best_intent,
            "score": best_score,
        }