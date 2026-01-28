from models.base import BaseModel, ModelInferenceError
from mlx_lm import load, generate
import asyncio

class LargeModel(BaseModel):

    def __init__(self, name: str, model_name: str, min_latency: float = 1.5, max_latency: float = 3.0,
                 failure_rate: float = 0.1, timeout_seconds: float = 20.0):

        super().__init__(name = name, model_type = "large", timeout_seconds = timeout_seconds),

        self.model_name = model_name

        # Lazy initialization
        self.model = None
        self.tokenizer = None

        self.min_latency = min_latency
        self.max_latency = max_latency
        self.failure_rate = failure_rate

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            try:
                self.model, self.tokenizer = load(self.model_name)
            except Exception as e:
                raise ModelInferenceError(f"failed to load large MLX model '{self.model_name}': {str(e)}")

    async def _generate(self, prompt: str, max_tokens: int) -> str:
        try:
            self._load_model()

            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            response = generate(model=self.model,
                                tokenizer=self.tokenizer, prompt=prompt, verbose=False,
                                max_tokens=max_tokens,
                                temp=0.7)

            return response

        except Exception as e:
            raise ModelInferenceError (f"large model inference failed {str(e)}")

    async def warmup(self) -> bool:
        self._load_model()

        dummy_message = [{"role": "user", "content": "Hello LARGE MODEL"}]
        dummy_prompt = self.tokenizer.apply_chat_template(
            dummy_message,
            tokenize=False,
            add_generation_prompt=True
        )

        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=dummy_prompt,
            max_tokens=1,
            temp=0.0,
            verbose=False
        )

        return True