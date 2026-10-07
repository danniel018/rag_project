from abc import ABC, abstractmethod
from typing import ClassVar

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from portkey_ai import Portkey


class LLMStrategy(ABC):

    last_usage: ClassVar[dict[str, float]] = {}

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> str: ...


class LocalLLM(LLMStrategy):
    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
    ):
        self.model = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,
            request_timeout=request_timeout,
        )

    def chat(self, messages: list[ChatMessage]) -> str:
        response = self.model.chat(messages)
        self.last_usage = {
            "prompt_tokens": response.raw.get.get("prompt_eval_count"),
            "completion_tokens": response.raw.get.get("completion_eval_count"),
        }
        return response.message.content


class RemoteLLM(LLMStrategy):
    # Portkey Model Catalog format: "@<provider-slug>/<model>".
    # Reads PORTKEY_API_KEY from the environment.
    def __init__(self, model_name: str = "@openai/gpt-6-luna"):
        self.model_name = model_name
        self.client = Portkey()

    def chat(self, messages: list[ChatMessage]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
        )
        usage = response.usage
        self.last_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
        return response.choices[0].message.content
