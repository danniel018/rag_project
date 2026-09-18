from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding

from chunking import MarkdownChunking, SemanticChunking
from embedding import LocalEmbedding
from storage import ChromaStore


def build_strategy(choice: str):
    if choice == "1":
        return MarkdownChunking()
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    return SemanticChunking(embed_model)


def prompt_strategy() -> str:
    print("Select a chunking strategy:")
    print("1. Markdown")
    print("2. Semantic")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, enter 1 or 2.")


def main() -> None:
    choice = prompt_strategy()

    with open("ht400_maintenance_manual.md", "r") as f:
        text = f.read()

    document = Document(text=text)
    strategy = build_strategy(choice)
    chunks = strategy.chunk(document)

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
