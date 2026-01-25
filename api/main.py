from routers.embedder import Embedder
from routers.intents import load_intent_embeddings
from routers.router import SemanticRouter

from models.small_model import SmallModel
from models.large_model import LargeModel


from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def life_span(app: FastAPI):

        embedder = Embedder()
        intent_embeddings = load_intent_embeddings(embedder)
        semantic_router = SemanticRouter(embedder, intent_embeddings)

        small_model = SmallModel()
        large_model = LargeModel()

        embedder.warmup()
        await small_model.warmup()
        await large_model.warmup()

        yield

app = FastAPI(life_span=life_span)


