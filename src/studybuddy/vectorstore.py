"""Vector-store setup and document retrieval."""
from langchain_core.embeddings import Embeddings

"""Embeddings and vector store (FAISS for the MVP)."""

from functools import lru_cache
from pathlib import Path
from fastembed import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder(Embeddings):
    """bge-small via ONNX: faster startup and CPU inference than PyTorch."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 64):
        self.model = TextEmbedding(model_name=model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()

from studybuddy.config import settings
from studybuddy.embed_cache import CachedEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> CachedEmbeddings:
    return CachedEmbeddings(FastEmbedder(), settings.embed_cache_db, namespace=EMBEDDING_MODEL)


def add_chunks(vs: FAISS | None, chunks: list[Document], batch_size: int = 64, on_progress=None) -> FAISS:
    """Embed chunks in batches and add them to vs (creating it if None)."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vs is None:
            vs = FAISS.from_documents(batch, get_embeddings())
        else:
            vs.add_documents(batch)
        if on_progress:
            on_progress(min(i + batch_size, len(chunks)), len(chunks))
    return vs


def build_vectorstore(chunks: list[Document], **kwargs) -> FAISS:
    return add_chunks(None, chunks, **kwargs)


def save_vectorstore(vs: FAISS, folder: str | Path) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(folder))  # saves vectors AND text/metadata


def load_vectorstore(folder: str | Path) -> FAISS:
    # FAISS stores its docstore with pickle, so only load indexes YOU created.
    return FAISS.load_local(
        str(folder), get_embeddings(), allow_dangerous_deserialization=True
    )