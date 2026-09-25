from abc import ABC, abstractmethod

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI


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
    def __init__(self, model_name: str = "gpt-5.4-mini"):
        self.model = OpenAI(model=model_name)

    def chat(self, messages: list[ChatMessage]) -> str:
        return self.model.chat(messages).message.content
