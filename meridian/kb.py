"""KnowledgeBase: the single facade over documents, chunks, all retrieval indexes, the graph and the fact index."""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import Settings
from .index.dense import DenseIndex, VisualIndex
from .index.facts import FactIndex
from .index.graph import KnowledgeGraph
from .index.lexical import BM25Index, CharNgramIndex
from .index.store import Store
from .ingest.pipeline import SUPPORTED, UnsupportedFileType, ingest_file, make_doc_id, sha256_file
from .schemas import Chunk, Document, Fact, Modality


@dataclass
class Indexes:
    """Immutable snapshot of everything derived from the chunk list. Swapped atomically after (re)build."""

    chunks: list[Chunk]
    by_id: dict[str, int]
    bm25: BM25Index
    dense: DenseIndex
    char: CharNgramIndex
    visual: VisualIndex
    graph: KnowledgeGraph
    facts: FactIndex
    entity_map: dict[str, list[int]]
    models: set[str]
    build_ms: float = 0.0


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.s = settings
        self.s.ensure_dirs()
        self.store = Store(self.s.data_dir / "meridian.db")
        self._lock = threading.RLock()
        self._qdrant_client = None
        self.docs: dict[str, Document] = {}
        self.ix: Indexes = self._build([], {})
        docs, chunks = self.store.load()
        if docs:
            self.docs = {d.id: d for d in docs}
            self.ix = self._build(chunks, self.docs)

    # ------------------------------------------------------------------------------------------ building
    def _build(self, chunks: list[Chunk], docs: dict[str, Document]) -> Indexes:
        t0 = time.perf_counter()
        facts: list[Fact] = []
        for c in chunks:
            for f in c.meta.get("facts", []):
                facts.append(Fact.model_validate(f))
        entity_map: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            for e in c.entities:
                entity_map.setdefault(e, []).append(i)
        models = {m for d in docs.values() for m in d.models}
        dense = DenseIndex(chunks, docs=docs, qdrant_client=self._qdrant_client)
        self._qdrant_client = dense._client
        ix = Indexes(
            chunks=chunks, by_id={c.id: i for i, c in enumerate(chunks)}, bm25=BM25Index(chunks), dense=dense,
            char=CharNgramIndex(chunks), visual=VisualIndex(chunks), graph=KnowledgeGraph(chunks), facts=FactIndex(facts),
            entity_map=entity_map, models=models,
        )
        ix.build_ms = (time.perf_counter() - t0) * 1000
        return ix

    def _reindex(self) -> None:
        _, chunks = self.store.load()
        self.ix = self._build(chunks, self.docs)

    # ------------------------------------------------------------------------------------------ mutation
    def add_file(self, path: Path, *, original_name: Optional[str] = None) -> dict:
        """Ingest one file. Idempotent on identical content; replaces an older version with the same document id."""
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED:
            raise UnsupportedFileType(f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED))}")
        sha = sha256_file(path)
        with self._lock:
            existing = self.store.find_by_sha(sha)
            if existing and existing in self.docs:
                return {"status": "duplicate", "doc": self.docs[existing].model_dump(), "chunks": 0}
            t0 = time.perf_counter()
            doc_id = make_doc_id(Path(original_name) if original_name else path)
            replaced = doc_id in self.docs
            doc, chunks = ingest_file(path, self.s, doc_id=doc_id)
            if original_name:
                doc.filename = original_name
            if replaced:
                self._purge_files(doc_id, keep_new=True)
            self.store.save(doc, chunks)
            self.docs[doc.id] = doc
            self._reindex()
            return {"status": "replaced" if replaced else "added", "doc": doc.model_dump(), "chunks": len(chunks),
                    "seconds": round(time.perf_counter() - t0, 2)}

    def add_directory(self, directory: Path) -> list[dict]:
        results = []
        for p in sorted(Path(directory).iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                try:
                    results.append({"file": p.name, **self.add_file(p)})
                except Exception as exc:
                    results.append({"file": p.name, "status": "error", "error": str(exc)})
        return results

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.docs:
                return False
            self.store.delete(doc_id)
            self.docs.pop(doc_id, None)
            self._purge_files(doc_id)
            self._reindex()
            return True

    def reset(self) -> None:
        with self._lock:
            for d in list(self.docs):
                self.store.delete(d)
                self._purge_files(d)
            self.docs = {}
            self._reindex()

    def _purge_files(self, doc_id: str, keep_new: bool = False) -> None:
        for sub in ("pages", "figures"):
            if keep_new and sub in ("pages", "figures"):
                continue
            shutil.rmtree(self.s.data_dir / sub / doc_id, ignore_errors=True)
        if not keep_new:
            (self.s.data_dir / "tables" / f"{doc_id}.csv").unlink(missing_ok=True)

    # ------------------------------------------------------------------------------------------ access
    @property
    def chunks(self) -> list[Chunk]:
        return self.ix.chunks

    def chunk(self, chunk_id: str) -> Optional[Chunk]:
        i = self.ix.by_id.get(chunk_id)
        return self.ix.chunks[i] if i is not None else None

    def doc(self, doc_id: str) -> Optional[Document]:
        return self.docs.get(doc_id)

    def mask(self, *, models: Optional[Iterable[str]] = None, doc_types: Optional[Iterable[str]] = None,
             modalities: Optional[Iterable[Modality]] = None, doc_ids: Optional[Iterable[str]] = None) -> Optional[np.ndarray]:
        """Boolean mask over chunks. Model filtering uses document-level model lists plus chunk-level entities, so
        a generic chunk of an HX-200 manual passes `models=['HX-200']` while another product's manual is excluded."""
        ch = self.ix.chunks
        if not ch or not (models or doc_types or modalities or doc_ids):
            return None
        m = np.ones(len(ch), dtype=bool)
        wanted = {x.upper() for x in models or []}
        types, mods, ids = set(doc_types or []), set(modalities or []), set(doc_ids or [])
        for i, c in enumerate(ch):
            d = self.docs.get(c.doc_id)
            if wanted and d is not None and d.models and not (wanted & {x.upper() for x in d.models}):
                m[i] = False
            if types and (d is None or d.doc_type not in types):
                m[i] = False
            if mods and c.modality not in mods:
                m[i] = False
            if ids and c.doc_id not in ids:
                m[i] = False
        return m

    def section_chunks(self, doc_id: str, section_no: str) -> list[Chunk]:
        return [c for c in self.ix.chunks if c.doc_id == doc_id and c.section_no == section_no]

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.ix.chunks if c.doc_id == doc_id]

    def neighbours(self, chunk: Chunk, window: int = 1) -> list[Chunk]:
        i = self.ix.by_id.get(chunk.id)
        if i is None:
            return []
        out = []
        for j in range(max(0, i - window), min(len(self.ix.chunks), i + window + 1)):
            c = self.ix.chunks[j]
            if j != i and c.doc_id == chunk.doc_id and c.page == chunk.page:
                out.append(c)
        return out

    def dataframe(self, doc_id: str) -> Optional[pd.DataFrame]:
        d = self.docs.get(doc_id)
        if not d or "dataset_path" not in d.meta:
            return None
        p = self.s.data_dir / d.meta["dataset_path"]
        return pd.read_csv(p) if p.exists() else None

    def datasets(self) -> list[Document]:
        return [d for d in self.docs.values() if "dataset_path" in d.meta]

    def known_models(self) -> list[str]:
        return sorted(self.ix.models)

    def known_error_codes(self) -> list[str]:
        return sorted({e for c in self.ix.chunks for e in c.entities if len(e) == 4 and e.startswith("E-")})

    def stats(self) -> dict:
        mods: dict[str, int] = {}
        for c in self.ix.chunks:
            mods[c.modality.value] = mods.get(c.modality.value, 0) + 1
        return {"documents": len(self.docs), "chunks": len(self.ix.chunks), "modalities": mods, "facts": len(self.ix.facts.facts),
                "graph": self.ix.graph.stats(), "models": self.known_models(), "embedding_backend": self.ix.dense.backend.name,
                "vector_store": self.ix.dense.vector_store, "index_build_ms": round(self.ix.build_ms, 1)}

    def close(self) -> None:
        if self._qdrant_client is not None:
            self._qdrant_client.close()
        self.store.close()
