from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingError(Exception):
    pass

class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Optional [SentenceTransformer] = None

    def _load_model(self):
        if self._model is None:
           try:
               self._model = SentenceTransformer(self.model_name)
           except Exception as e:
               raise EmbeddingError(f"failed to load embedding model: {self.model_name} {str(e)}")

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise EmbeddingError("text is empty, cannot generate embedding")

        self._load_model()

        try:
            embedding = self._model.encode(text, convert_to_numpy=True,
                                           normalize_embeddings=True)
            return embedding
        except Exception as e:
            raise EmbeddingError(f"failed to generate embedding: {str(e)}")

    def warmup(self) -> bool:
        self._load_model()
        embedding =  self._model.encode("Yoooo Helloooo", convert_to_numpy=True)
        return True



