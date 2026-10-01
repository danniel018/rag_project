from embedding import EmbeddingStrategy
from llm import LLMStrategy
from prompt import build_messages
from storage import ChromaStore


class QueryEngine:
    def __init__(
        self,
        embedder: EmbeddingStrategy,
        llm: LLMStrategy,
        store: ChromaStore,
        top_k: int = 5,
    ):
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._top_k = top_k

    def query(self, question: str) -> str:
        ids, texts, metadatas = self._store.query(
            embedding=self._embedder.embed(question), n_results=self._top_k
        )

        global_ids, global_texts, global_metadatas = self._store.get_global_chunks()

        seen = set(ids)
        for gid, gtext, gmeta in zip(global_ids, global_texts, global_metadatas):
            if gid not in seen:
                ids.append(gid)
                texts.append(gtext)
                metadatas.append(gmeta)
                seen.add(gid)

        return self._llm.chat(build_messages(question, texts, metadatas))