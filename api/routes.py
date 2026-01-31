from fastapi import APIRouter, Request, HTTPException
import time
import asyncio

from api.schemas import GenerateRequest, GenerateResponse, ErrorResponse
from models.base import ModelInferenceError

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse, responses={500: {"model": ErrorResponse}})
async def generate(payload: GenerateRequest, request: Request):
    start_time = time.time()

    app_state = request.app.state
    semantic_router = app_state.router

    routing_result = semantic_router.route(payload.prompt)

    model_tier = routing_result["model"]
    intent = routing_result["intent"]
    score = routing_result["score"]

    if model_tier == "small":
        model = app_state.small_model
    else:
        model = app_state.large_model

    try:
        async with asyncio.timeout(model.timeout_seconds):
            result = await model.generate(prompt=payload.prompt,
                                  max_tokens=payload.max_tokens)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail=ErrorResponse(
            error="MODEL_TIMEOUT",detail=f"Inference exceeded {model.timeout_seconds}s").model_dump()
        )

    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=ErrorResponse(
            error="MODEL_INFERENCE_FAILED",
            detail=str(e)).model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=ErrorResponse(
            error="UNEXPECTED_ERROR",detail=str(e)).model_dump()
        )

    latency_ms = (time.time() - start_time) * 1000

    return GenerateResponse( # optional fields can be left
        response=result["text"],
        model_used=result["model_name"],
        intent=intent,
        latency_ms=latency_ms,
        score=score,
        cached=False)