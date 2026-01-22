from models.base import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

class SmallModel(BaseModel):

    def __init__(self, name: str, model_type: str, model_name: str,
                 timeout_seconds: float = 10.0, device: str = "cpu", ):

        super().__init__(
            name  = name,
            model_type = "small",
            timeout_seconds = timeout_seconds)

        self.model_name = model_name
        self.device = device

        #lazy initialization
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            '''
            .to() tells PyTorch to move the entire neural network
             from your general RAM to a specific processor.
             .eval() turns off those "training-only" behaviors.
            '''

    async def _generate(self, prompt: str, max_tokens: int) -> str:

        self._load_model()

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                         max_new_tokens = max_tokens,
                                         do_sample = True,
                                         temprature = 0.7)

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return generated_text

    async def warmup(self):
        self._load_model()

        dummy_prompt = "Yoooo Helloooo"
        inputs = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens = 1)

        return True