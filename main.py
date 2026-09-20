from llama_index.core import Document

from chunking import MarkdownChunking, SemanticChunking
from embedding import LocalEmbedding, RemoteEmbedding
from storage import ChromaStore


def build_embedding_strategy(choice: str):
    if choice == "1":
        return LocalEmbedding()
    return RemoteEmbedding()


def build_chunking_strategy(choice: str, embedding_strategy):
    if choice == "1":
        return MarkdownChunking()
    return SemanticChunking(embedding_strategy.model)


def prompt_embedding_strategy() -> str:
    print("Select an embedding strategy:")
    print("1. Local (Ollama)")
    print("2. Remote (OpenAI)")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, enter 1 or 2.")


def prompt_chunking_strategy() -> str:
    print("Select a chunking strategy:")
    print("1. Markdown")
    print("2. Semantic")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, enter 1 or 2.")


def main() -> None:
    embedding_choice = prompt_embedding_strategy()
    chunking_choice = prompt_chunking_strategy()

    with open("ht400_maintenance_manual.md", "r") as f:
        text = f.read()

    document = Document(text=text)
    embedding_strategy = build_embedding_strategy(embedding_choice)
    chunking_strategy = build_chunking_strategy(chunking_choice, embedding_strategy)
    chunks = chunking_strategy.chunk(document)

    print(f"Number of chunks: {len(chunks)}")

    embedder = LocalEmbedding()
    ids = [str(i) for i in range(len(chunks))]
    texts = [chunk.text for chunk in chunks]
    embeddings = [embedder.embed(text) for text in texts]

    store = ChromaStore()
    store.add(ids=ids, texts=texts, embeddings=embeddings)

    print(f"Stored {len(chunks)} chunks in Chroma.")


if __name__ == "__main__":
    main()
