import os
from api import routes

from routers.embedder import Embedder
from routers.intents import load_intent_embeddings
from routers.router import SemanticRouter

from models.small_model import SmallModel
from models.large_model import LargeModel

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):

    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    small_model_name = os.getenv("SMALL_MODEL_NAME")
    large_model_name = os.getenv("LARGE_MODEL_NAME")

    embedder = Embedder(embedding_model_name)
    intent_embeddings = load_intent_embeddings(embedder)
    semantic_router = SemanticRouter(embedder, intent_embeddings)

    small_model = SmallModel(name ="small_model",
                                 model_name=small_model_name,)

    large_model = LargeModel(name = "large_model",
                             model_name=large_model_name)

    embedder.warmup()
    await small_model.warmup()
    await large_model.warmup()

    app.state.embedder = embedder
    app.state.router = semantic_router
    app.state.small_model = small_model
    app.state.large_model = large_model

    yield

def start_app() -> FastAPI:
    app = FastAPI(title="Multi-Model AI Inference API", lifespan=lifespan)
    app.include_router(routes.router)

    @app.get("/")
    async def root():
        return "Multi-Model AI Inference API"

    @app.get("/health")
    async def health():
        return{
        "status": "ok",
        "router_loaded": True,
        "models_loaded": True
        }

    return app

app = start_app()


