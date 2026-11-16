"""SQLite persistence for documents and chunks (indexes are rebuilt in memory on load: fast for this scale)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..schemas import Chunk, Document


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, sha256 TEXT, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ord INTEGER NOT NULL, json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);
            """
        )
        self._conn.commit()

    def save(self, doc: Document, chunks: list[Chunk]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.id,))
            self._conn.execute("INSERT OR REPLACE INTO docs(id, sha256, json) VALUES(?,?,?)", (doc.id, doc.sha256, doc.model_dump_json()))
            self._conn.executemany(
                "INSERT INTO chunks(id, doc_id, ord, json) VALUES(?,?,?,?)",
                [(c.id, doc.id, i, c.model_dump_json()) for i, c in enumerate(chunks)],
            )

    def delete(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            cur = self._conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
            return cur.rowcount > 0

    def load(self) -> tuple[list[Document], list[Chunk]]:
        with self._lock:
            docs = [Document.model_validate_json(r[0]) for r in self._conn.execute("SELECT json FROM docs ORDER BY id")]
            chunks = [Chunk.model_validate_json(r[0]) for r in self._conn.execute("SELECT json FROM chunks ORDER BY doc_id, ord")]
        return docs, chunks

    def sha_of(self, doc_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT sha256 FROM docs WHERE id=?", (doc_id,)).fetchone()
        return row[0] if row else None

    def find_by_sha(self, sha: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT id FROM docs WHERE sha256=?", (sha,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
