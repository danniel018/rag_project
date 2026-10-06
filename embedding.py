from abc import ABC, abstractmethod

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from portkey_ai import Portkey
from pydantic import PrivateAttr


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class LocalEmbedding(EmbeddingStrategy):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = OllamaEmbedding(model_name=model_name, base_url=base_url)

    def embed(self, text: str) -> list[float]:
        return self.model.get_text_embedding(text)


class PortkeyEmbedding(BaseEmbedding):
    # LlamaIndex embedding backed by Portkey, so it can also drive
    # SemanticSplitterNodeParser. Reads PORTKEY_API_KEY from the environment.
    _client: Portkey = PrivateAttr()

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._client = Portkey(
            base_url="https://portkeygateway.perficient.com/v1",
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.model_name, input=texts, encoding_format="float"
        )
        return [item.embedding for item in response.data]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


class RemoteEmbedding(EmbeddingStrategy):
    # Portkey Model Catalog format: "@<provider-slug>/<model>".
    def __init__(self, model_name: str = "@azure-openai/text-embedding-3-small"):
        self.model = PortkeyEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.get_text_embedding(text)
