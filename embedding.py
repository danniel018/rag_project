from abc import ABC, abstractmethod

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class LocalEmbedding(EmbeddingStrategy):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model = OllamaEmbedding(model_name=model_name, base_url=base_url)

    def embed(self, text: str) -> list[float]:
        return self._model.get_text_embedding(text)


class RemoteEmbedding(EmbeddingStrategy):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self._model = OpenAIEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.get_text_embedding(text)
