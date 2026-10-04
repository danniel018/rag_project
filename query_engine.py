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

    def retrieve(self, question: str) -> tuple[list[str], list[dict]]:
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

        return texts, metadatas

    def check_retrieval(
        self, question: str, required_spans: list[str]
    ) -> tuple[str, bool]:
        # Retrieval check: skip the LLM and report whether every required
        # span appears in the retrieved context.
        texts, _ = self.retrieve(question)
        # Collapse line breaks and repeated whitespace so spans that wrap
        # across lines in the source still match.
        context = " ".join(texts)
        pass_ = all(span in context for span in required_spans)
        return question, pass_

    def query(self, question: str) -> str:
        texts, metadatas = self.retrieve(question)
        return self._llm.chat(build_messages(question, texts, metadatas))
