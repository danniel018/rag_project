import time

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
        top_k: int = 3,
    ):
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self.last_timings: dict[str, float] = {}

    def retrieve(self, question: str) -> tuple[list[str], list[dict]]:
        embedding = self._embedder.embed(question)

        ids, texts, metadatas = self._store.query(
            embedding=embedding, n_results=self._top_k
        )
        global_ids, global_texts, global_metadatas = self._store.get_global_chunks()
        seen = set(ids)
        for gid, gtext, gmeta in zip(global_ids, global_texts, global_metadatas):
            if gid not in seen:
                ids.append(gid)
                texts.append(gtext)
                metadatas.append(gmeta)
                seen.add(gid)
        return texts, metadatas

    def check_retrieval(
        self, question: str, required_spans: list[str]
    ) -> tuple[str, bool, list[int]]:
        # Retrieval check: skip the LLM and report whether every required
        # span appears in the retrieved context, plus the indexes of any
        # spans that were not found.
        texts, _ = self.retrieve(question)
        # Collapse line breaks and repeated whitespace so spans that wrap
        # across lines in the source still match.
        context = " ".join(texts)
        missing = [i for i, span in enumerate(required_spans) if span not in context]
        pass_ = not missing
        return question, pass_, missing

    def query(self, question: str) -> str:
        t0 = time.perf_counter()
        texts, metadatas = self.retrieve(question)
        answer = self._llm.chat(build_messages(question, texts, metadatas))
        self.last_timings = {
            "total_s": time.perf_counter() - t0,
            "prompt_tokens": self._llm.last_usage.get("prompt_tokens", 0),
            "completion_tokens": self._llm.last_usage.get("completion_tokens", 0),
        }
        return answer
