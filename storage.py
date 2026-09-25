import chromadb


class ChromaStore:
    def __init__(self, collection_name: str = "documents", path: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=path)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(collection_name)

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(self._collection_name)

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

    def query(
        self, embedding: list[float], n_results: int
    ) -> tuple[list[str], list[dict]]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        return result["documents"][0], result["metadatas"][0]
