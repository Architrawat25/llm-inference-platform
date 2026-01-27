from fastapi import APIRouter, Request
import time

from api.schemas import GenerateRequest, GenerateResponse, ErrorResponse

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest, request: Request):
    start_time = time.time()

    app_state = request.app.state
    semantic_router = app_state.router

    routing_result = semantic_router.route(payload.prompt)

    model_tier = routing_result["model"]
    intent = routing_result["intent"]
    score = routing_result["score"]

    if model_tier == "large":
        model = app_state.large_model
    else:
        model = app_state.small_model

    result = await model.generate(promt=payload.prompt,
                                  max_tokenx=payload.max_tokens)

    latency_ms = (time.time() - start_time) * 1000

    return GenerateResponse( # optional fields can be left
        response=result["text"],
        model_used=result["model_name"],
        intent=intent,
        latency_ms=latency_ms,
        score=score,
        cached=False)