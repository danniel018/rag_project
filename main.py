import statistics
import sys
import time
from uuid import uuid4

from llama_index.core import Document

from chunking import (
    MarkdownChunking,
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


def build_chunking_strategy(choice: str):
    if choice == "1":
        return MarkdownChunking()
    return SentenceSplitterChunking()


def ask_choice(title: str, options: list[str]) -> str:
    """Show a numbered menu with a quit option; exits the app on 'q'."""
    print(title)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    print("q. Quit")
    valid = [str(i) for i in range(1, len(options) + 1)]
    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice in ("q", "quit"):
            print("Goodbye.")
            sys.exit(0)
        if choice in valid:
            return choice
        print(f"Invalid choice, enter {', '.join(valid)} or q to quit.")


def prompt_pipeline_strategy() -> str:
    return ask_choice(
        "Select a pipeline execution:", ["Local (Ollama)", "Remote (Portkey)"]
    )


def prompt_chunking_strategy() -> str:
    return ask_choice("Select a chunking strategy:", ["Markdown", "Sentence Splitter"])


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
    print("Ask a question (empty line or q to quit).")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in ("q", "quit"):
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


def execute_all_questions(query_engine: QueryEngine, indexing_s: float) -> None:
    # Warm-up: the first local call loads the model into memory.
    query_engine.query(QUESTIONS[0])
    cold = query_engine.last_timings["total_s"]
    print(f"Warm-up query (includes model load): {cold:.2f}s\n")

    times, prompt_tokens, completion_tokens = [], [], []
    for index, question in enumerate(QUESTIONS, start=1):
        answer = query_engine.query(question)
        t = query_engine.last_timings
        times.append(t["total_s"])
        prompt_tokens.append(t["prompt_tokens"])
        completion_tokens.append(t["completion_tokens"])
        print(f"\nQ{index:02d} ({t['total_s']:.2f}s)\n{answer}\n")

    print("\n== Latency summary (seconds) ==")
    print(f"per question median {statistics.median(times):.3f} | max {max(times):.3f}")
    print(f"embeddings indexing {indexing_s:.2f}")
    print(f"total pipeline execution {indexing_s + sum(times):.2f} "
          f"(indexing + {len(times)} questions)")

    print("\n== Token usage summary ==")
    print(f"in {sum(prompt_tokens)} prompt tokens | "
          f"out {sum(completion_tokens)} completion tokens")


def prompt_run_mode() -> str:
    return ask_choice(
        "Select a run mode:",
        [
            "Ask questions (LLM)",
            "Check retrieval for config questions",
            "Execute all questions",
        ],
    )


def main() -> None:
    pipeline_choice = prompt_pipeline_strategy()
    chunking_choice = prompt_chunking_strategy()
    run_mode_choice = prompt_run_mode()

    documents = build_documents(SOURCE_FILES)

    pipeline_execution = build_pipeline_execution(pipeline_choice)
    chunking_strategy = build_chunking_strategy(chunking_choice)
    chunks = chunking_strategy.chunk(documents)

    for chunk in chunks:
        if is_global_section(chunk.text):
            chunk.metadata["global_section"] = True

    print(f"Number of chunks: {len(chunks)}")

    t0 = time.perf_counter()
    embeddings = build_embeddings(chunks, pipeline_execution)
    indexing_s = time.perf_counter() - t0
    print(f"Indexing Embedding time: {indexing_s:.2f}s")
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
        execute_all_questions(query_engine, indexing_s)


if __name__ == "__main__":
    main()
