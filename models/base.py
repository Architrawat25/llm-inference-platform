from abc import ABC, abstractmethod
import time
from typing import Optional


class BaseModel(ABC):

    def __init__(self, name: str, model_type: str, timeout_seconds: float = 10.0):
        self.name = name
        self.model_type = model_type
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def _generate(self, prompt: str, max_tokens: int) -> str:
        pass

    async def generate(self, prompt: str, max_tokens: int) -> dict:
        start_time  = time.time()

        output = await self._generate(prompt, max_tokens)

        end_time = time.time()

        latency = end_time - start_time # in seconds
        latency_ms = end_time - start_time * 1000 # in milliseconds

        inference_result = {
            "text": output,
            "model_name": self.name,
            "model_type": self.model_type,
            "latency": latency_ms
        }

        return inference_result

    async def warmup(self) -> Optional[bool]:
        return None




