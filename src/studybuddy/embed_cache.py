"""Persistent embedding cache: never embed the same text twice."""

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings


class CachedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, db_path: str | Path, namespace: str):
        self.inner = inner
        self.db_path = Path(db_path)
        self.namespace = namespace  # e.g. the model name
        self.hits = 0
        self.misses = 0
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # tolerates several Streamlit sessions at once
        return conn

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{text}".encode()).hexdigest()

    def reset_stats(self) -> None:
        self.hits = self.misses = 0

    def _get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with closing(self._connect()) as conn:
            for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
                batch = keys[i : i + 500]
                marks = ",".join("?" * len(batch))  # placeholders only; values are bound
                rows = conn.execute(f"SELECT key, vec FROM vectors WHERE key IN ({marks})", batch)
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return found

    def _put_many(self, items: list[tuple[str, list[float]]]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found = self._get_many(list(set(keys)))

        missing = {}  # key -> text, de-duplicated
        for k, t in zip(keys, texts):
            if k not in found:
                missing.setdefault(k, t)

        if missing:
            new_vectors = self.inner.embed_documents(list(missing.values()))
            self._put_many(list(zip(missing.keys(), new_vectors)))
            for k, v in zip(missing.keys(), new_vectors):
                found[k] = np.asarray(v, dtype=np.float32).tolist()  # same precision as cached ones

        n_missing = sum(1 for k in keys if k not in set(missing))  # texts served from cache
        self.hits += n_missing
        self.misses += len(texts) - n_missing
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)  # queries are short and unique: not worth caching