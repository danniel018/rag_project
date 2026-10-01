from abc import ABC, abstractmethod
from re import search

from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)

GLOBAL_HEADING_PATTERNS = [
    r"\d+\.?\s*scope\b",
    r"\d+\.?\s*general safety requirements",
    r"\d+\.\d+\s*confined space",
]

def is_global_section(chunk_text: str) -> bool:
    text_norm = chunk_text.lower()
    return any(search(pat, text_norm) for pat in GLOBAL_HEADING_PATTERNS)

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list: ...


class MarkdownChunking(ChunkingStrategy):
    def __init__(self):
        self._parser = MarkdownNodeParser()

    def chunk(self, documents: list[Document]) -> list:
        return self._parser.get_nodes_from_documents(documents=documents)


class SemanticChunking(ChunkingStrategy):
    def __init__(self, embed_model):
        self._parser = SemanticSplitterNodeParser(embed_model=embed_model)

    def chunk(self, documents: list[Document]) -> list:
        return self._parser.get_nodes_from_documents(documents=documents)
    
class SentenceSplitterChunking(ChunkingStrategy):
    """Splits by sentence boundaries up to a fixed
    chunk_size
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20):
        self._parser = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: list[Document]) -> list:
        return self._parser.get_nodes_from_documents(documents=documents)
