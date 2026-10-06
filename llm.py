from abc import ABC, abstractmethod

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from portkey_ai import Portkey


class LLMStrategy(ABC):
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
        return self.model.chat(messages).message.content


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
        return response.choices[0].message.content
