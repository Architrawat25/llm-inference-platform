from typing import Optional
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for the language model")
    max_tokens: int = Field(default=200, ge=10, le= 2000,
                            description="Maximum number of tokens to generate")
    '''
    ge = greater than equal to
    le = less than equal to
    10 <= max_tokens <= 2048

    '''

class GenerateResponse(BaseModel):
    response: str
    model_used: str
    latency_ms: float

    # field is Optional, default is None, validation allows either the type or None
    intent: str | None = None
    score: float | None = None
    cached: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: str | None  = None
