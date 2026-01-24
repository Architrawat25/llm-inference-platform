from routers.embedder import Embedder
import numpy as np

class IntentDefinition:
    def __init__(self, name: str, description: str, target_model: str):
        self.name = name
        self.description = description
        self.target_model = target_model

#List of Intent Definitions
intent_list = [
    IntentDefinition(
        name="casual",
        description="Casual conversation, greetings, chit-chat, or informal questions",
        target_model="small",
    ),
    IntentDefinition(
        name="factual",
        description="Short factual questions or simple explanations",
        target_model="small",
    ),
    IntentDefinition(
        name="technical",
        description="Technical explanations related to programming, computer science, or machine learning",
        target_model="large",
    ),
    IntentDefinition(
        name="long_form",
        description="Long, detailed, or structured explanations requiring deeper reasoning",
        target_model="large",
    ),
]

def load_intent_embeddings(embedder: Embedder) -> dict[str, dict[str, np.ndarray]] :

    intent_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for intent in intent_list:
        embedding  = embedder.embed(intent.description)

        intent_embeddings[intent.name] = {
            "embedding": embedding,
            "target_model": intent.target_model,
        }

    return intent_embeddings

