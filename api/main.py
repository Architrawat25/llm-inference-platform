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
    routing_threshold = float(os.getenv("ROUTING_THRESHOLD"))

    embedder = Embedder(embedding_model_name)
    intent_embeddings = load_intent_embeddings(embedder)
    semantic_router = SemanticRouter(embedder, intent_embeddings)

    small_model = SmallModel(name = "small_model",
                                 model_name = small_model_name,
                                 device = os.getenv("SMALL_MODEL_DEVICE"))

    large_model = LargeModel(name = "large_model")

    embedder.warmup()
    await small_model.warmup()
    await large_model.warmup()

    app.state.embedder = embedder
    app.state.router = semantic_router
    app.state.small_model = small_model
    app.state.large_model = large_model

    yield

def start_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(routes.router)
    return app

app = start_app()


