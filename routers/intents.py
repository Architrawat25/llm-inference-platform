from typing import Any
from routers.embedder import Embedder
import numpy as np

class IntentDefinition:
    def __init__(self, intent: str, examples: list[str], target_model: str, threshold: float):
        self.intent = intent
        self.examples = examples
        self.target_model = target_model
        self.threshold = threshold
        self.centroid: np.ndarray | None = None

intent_list = [
    IntentDefinition(
        intent="casual",
        examples=[
            "hey how are you",
            "what's up",
            "good evening",
            "how's it going today",
            "are you doing well",
            "hello there",
            "yo",
            "nice to meet you",
            "hope you're having a good day",
            "hi",
        ],
        target_model="small",
        threshold=0.45,
    ),
    IntentDefinition(
        intent="factual",
        examples=[
            "what is the capital of japan",
            "who invented the telephone",
            "define polymorphism in programming",
            "what does http stand for",
            "when was google founded",
            "who is the ceo of microsoft",
            "what is the speed of light",
            "meaning of recursion",
            "what is docker",
            "what is the boiling point of water",
        ],
        target_model="small",
        threshold=0.55,
    ),
    IntentDefinition(
        intent="code_help",
        examples=[
            "python for loop syntax example",
            "how do I write a switch case in java",
            "show me how list comprehension works in python",
            "javascript arrow function syntax",
            "how to declare a class in c plus plus",
            "example of try catch in python",
            "sql select query basic syntax",
            "how to write async function in nodejs",
            "how do I create a dictionary in python",
            "basic html boilerplate code",
        ],
        target_model="small",
        threshold=0.55,
    ),
    IntentDefinition(
        intent="debugging",
        examples=[
            "my python code is giving index error can you help",
            "why is this function returning none instead of value",
            "this api endpoint is failing with 500 error help me debug",
            "segmentation fault in my c program how to fix",
            "my fastapi server crashes when I send post request",
            "this recursion code runs forever what's wrong",
            "why am I getting null pointer exception here",
            "my sql query is not returning expected rows",
            "help me fix memory leak in this program",
            "why is my model training stuck at same accuracy",
        ],
        target_model="large",
        threshold=0.65,
    ),
    IntentDefinition(
        intent="concept_explanation_deep",
        examples=[
            "explain transformers in deep learning in detail",
            "how does backpropagation actually work step by step",
            "teach me how distributed systems maintain consistency",
            "explain how operating system scheduling works deeply",
            "what is gradient descent and why does it work mathematically",
            "explain microservices architecture with pros and cons",
            "how do databases handle transactions internally",
            "explain attention mechanism like I am a cs student",
            "how do neural networks actually learn patterns",
            "explain rest vs grpc in depth",
        ],
        target_model="large",
        threshold=0.70,
    ),
    IntentDefinition(
        intent="planning",
        examples=[
            "help me design an ml project end to end",
            "how should I build a chatbot architecture",
            "plan a full stack web app for college project",
            "how to structure a backend system for scale",
            "help me design an ai agent system",
            "what components do I need for recommendation system project",
            "guide me to build production ready fastapi backend",
            "how to design database schema for ecommerce app",
            "how should I structure microservices for startup idea",
            "plan a genai application from scratch",
        ],
        target_model="large",
        threshold=0.70,
    ),
    IntentDefinition(
        intent="creative",
        examples=[
            "write a short horror story",
            "generate a sci fi story about ai taking over mars",
            "write a poem about loneliness",
            "create a fantasy world idea for novel",
            "write a dialogue between human and alien",
            "give me startup name ideas for ai company",
            "write a motivational speech for students",
            "generate a fictional character backstory",
            "write a comedy scene in office setting",
            "create a game storyline idea",
        ],
        target_model="large",
        threshold=0.60,
    ),
    IntentDefinition(
        intent="step_by_step",
        examples=[
            "how should I start learning machine learning from zero",
            "make a roadmap to become backend developer",
            "what should I learn to become data scientist step by step",
            "guide me to learn python for software engineering",
            "how to prepare for ai engineer role roadmap",
            "learning path for cloud engineering beginner",
            "what topics should I study for system design interviews",
            "how to learn deep learning in 6 months plan",
            "roadmap to become full stack developer",
            "how do I start learning devops from scratch",
        ],
        target_model="large",
        threshold=0.65,
    ),
]

def load_intent_embeddings(embedder: Embedder) -> dict[str, dict[str, Any]]:
    intent_embeddings: dict[str, dict[str, Any]] = {}

    for intent_def in intent_list:
        example_vector = []

        for example in intent_def.examples:
            example_embedding = embedder.embed(example)
            example_vector.append(example_embedding)

        centroid = np.mean(example_vector, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        intent_def.centroid = centroid

        intent_embeddings[intent_def.intent] = {
            "centroid": centroid,
            "target_model": intent_def.target_model,
            "threshold": intent_def.threshold,
        }

    return intent_embeddings
