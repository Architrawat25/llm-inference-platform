from models.base import BaseModel, ModelInferenceError
import asyncio
import random

class LargeModel(BaseModel):

    def __init__(self, name: str, min_latency: float = 1.5, max_latency: float = 3.0,
                 failure_rate: float = 0.1, timeout_seconds: float = 10.0):

        super().__init__(name = name, model_type = "large", timeout_seconds = timeout_seconds),

        self.min_latency = min_latency
        self.max_latency = max_latency
        self.failure_rate = failure_rate

    async def _generate(self, prompt: str, max_tokens: int):
        try:

            latency = random.uniform(self.min_latency, self.max_latency)

            await asyncio.sleep(latency)
            '''
            asyncio.sleep() pauses the function without doing any work
            It uses 0% CPU. It’s just a countdown.
            '''

            if random.random() < self.failure_rate:
                raise RuntimeError("Large model failure")

            return f"[large model response] {prompt}"

        except Exception as e:
            raise ModelInferenceError (f"large model inference failed {str(e)}")