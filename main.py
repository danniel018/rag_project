from uuid import uuid4

from llama_index.core import Document

from chunking import (
    MarkdownChunking,
    SemanticChunking,
    SentenceSplitterChunking,
    is_global_section,
)
from config import QUESTIONS, REQUIRED_SPANS
from embedding import LocalEmbedding, RemoteEmbedding
from llm import LocalLLM, RemoteLLM
from query_engine import QueryEngine
from storage import ChromaStore


def build_pipeline_execution(choice: str):
    if choice == "1":
        return LocalEmbedding()
    return RemoteEmbedding()


def build_llm(choice: str):
    if choice == "1":
        return LocalLLM()
    return RemoteLLM()


def build_chunking_strategy(choice: str, embedding_strategy):
    if choice == "1":
        return MarkdownChunking()
    if choice == "2":
        return SemanticChunking(embedding_strategy.model)
    return SentenceSplitterChunking()


def prompt_pipeline_strategy() -> str:
    print("Select a pipeline execution:")
    print("1. Local (Ollama)")
    print("2. Remote (Portkey)")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, enter 1 or 2.")


def prompt_chunking_strategy() -> str:
    print("Select a chunking strategy:")
    print("1. Markdown")
    print("2. Semantic")
    print("3. Sentence Splitter")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("Invalid choice, enter 1, 2 or 3.")


SOURCE_FILES = [
    "ht400_maintenance_manual.md",
    "quick_reference_liner_change.md",
    "service_bulletin_SB-2026-04.md",
]


def build_documents(paths: list[str]) -> list[Document]:
    documents = []
    for path in paths:
        with open(path, "r") as f:
            documents.append(Document(text=f.read(), metadata={"source": path}))
    return documents


def build_embeddings(chunks: list, embedder) -> list[list[float]]:
    return [embedder.embed(chunk.text) for chunk in chunks]


def store_chunks(
    chunks: list, embeddings: list[list[float]], store: ChromaStore
) -> None:
    ids = [str(uuid4()) for _ in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Rebuild the collection so re-runs don't duplicate chunks or mix
    # embeddings from different models.
    store.reset()
    store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)


def run_question_loop(query_engine: QueryEngine) -> None:
    print("Ask a question (empty line to exit).")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        print(f"\n{query_engine.query(question)}\n")


def run_retrieval_check(query_engine: QueryEngine) -> None:
    passed = 0
    for i, (question, spans) in enumerate(zip(QUESTIONS, REQUIRED_SPANS), start=1):
        question, pass_, missing = query_engine.check_retrieval(question, spans)
        passed += pass_
        missing_info = "" if pass_ else f" | Missing span index(es): {missing}"
        print(f"Q{i:02d} [{'PASS' if pass_ else 'FAIL'}] {question}{missing_info}")
    print(f"\n{passed}/{len(QUESTIONS)} questions retrieved all required spans.")


def execute_all_questions(query_engine: QueryEngine) -> None:
    
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nQ{index:02d}: {query_engine.query(question)}\n")

def prompt_run_mode() -> str:
    print("Select a run mode:")
    print("1. Ask questions (LLM)")
    print("2. Check retrieval for config questions")
    print("3. Execute all questions")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("Invalid choice, enter 1, 2, or 3.")


def main() -> None:
    pipeline_choice = prompt_pipeline_strategy()
    chunking_choice = prompt_chunking_strategy()
    run_mode_choice = prompt_run_mode()

    documents = build_documents(SOURCE_FILES)

    pipeline_execution = build_pipeline_execution(pipeline_choice)
    chunking_strategy = build_chunking_strategy(chunking_choice, pipeline_execution)
    chunks = chunking_strategy.chunk(documents)

    for chunk in chunks:
        if is_global_section(chunk.text):
            chunk.metadata["global_section"] = True

    print(f"Number of chunks: {len(chunks)}")

    embeddings = build_embeddings(chunks, pipeline_execution)
    store = ChromaStore()
    store_chunks(chunks, embeddings, store)

    print(f"Stored {len(chunks)} chunks in Chroma.")
    print("Indexing complete.")

    query_engine = QueryEngine(pipeline_execution, build_llm(pipeline_choice), store)
    if run_mode_choice == "1":
        run_question_loop(query_engine)
    elif run_mode_choice == "2":
        run_retrieval_check(query_engine)
    else:
        execute_all_questions(query_engine)


if __name__ == "__main__":
    main()
