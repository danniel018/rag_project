import chromadb


class ChromaStore:
    def __init__(self, collection_name: str = "documents", path: str = "./chroma_db"):
        client = chromadb.PersistentClient(path=path)
        self._collection = client.get_or_create_collection(collection_name)

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self._collection.add(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )
