from abc import ABC, abstractmethod

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list:
        ...


class MarkdownChunking(ChunkingStrategy):
    def __init__(self):
        self._parser = MarkdownNodeParser()

    def chunk(self, document: Document) -> list:
        return self._parser.get_nodes_from_documents(documents=[document])


class SemanticChunking(ChunkingStrategy):
    def __init__(self, embed_model):
        self._parser = SemanticSplitterNodeParser(embed_model=embed_model)

    def chunk(self, document: Document) -> list:
        return self._parser.get_nodes_from_documents(documents=[document])
