"""
Minimal drop-in replacement for langchain_pinecone.PineconeVectorStore.

Works on Python 3.14 using only the official `pinecone` SDK + langchain-core:
    pip install pinecone

Usage:
    from pinecone_store import PineconeVectorStore
"""
import os
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        index_name: str,
        embedding: Embeddings,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        text_key: str = "text",
    ):
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Set PINECONE_API_KEY in your .env or pass api_key=...")
        self._index = Pinecone(api_key=api_key).Index(index_name)
        self._embedding = embedding
        self._namespace = namespace
        self._text_key = text_key

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---------- write ----------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        vectors = self._embedding.embed_documents(texts)

        for start in range(0, len(texts), batch_size):
            batch = [
                {
                    "id": ids[i],
                    "values": vectors[i],
                    "metadata": {**metadatas[i], self._text_key: texts[i]},
                }
                for i in range(start, min(start + batch_size, len(texts)))
            ]
            self._index.upsert(vectors=batch, namespace=self._namespace)
        return ids

    def delete(self, ids: Optional[List[str]] = None, delete_all: bool = False, **kwargs: Any) -> None:
        if delete_all:
            self._index.delete(delete_all=True, namespace=self._namespace)
        elif ids:
            self._index.delete(ids=ids, namespace=self._namespace)

    # ---------- read ----------
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        result = self._index.query(
            vector=self._embedding.embed_query(query),
            top_k=k,
            include_metadata=True,
            namespace=self._namespace,
            filter=filter,
        )
        docs = []
        for match in result.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self._text_key, "")
            docs.append((Document(page_content=text, metadata=metadata, id=match.id), match.score))
        return docs

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    # ---------- constructors ----------
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("Pass index_name=... or set PINECONE_INDEX_NAME")
        store = cls(index_name=index_name, embedding=embedding, namespace=namespace)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store