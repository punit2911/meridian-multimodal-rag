"""Document knowledge graph: chunks, entities and structural/cross-reference edges, with Personalized PageRank.

Edges (weights):
  chunk -- entity        mention (1.0)
  chunk -> chunk         resolved cross-reference "see Table 2", "Section 6.1", "Figure 1" (1.6)
  table_row -- table     structural parent (1.0)
  chunk -- chunk         adjacency inside one section (0.5), warning <-> same-section content (0.8)
"""
from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx

from ..schemas import Chunk, Modality


def resolve_refs(chunks: list[Chunk]) -> dict[str, list[tuple[str, str]]]:
    """Map chunk id -> [(target chunk id, ref label)] by resolving table/figure/section references within the same document."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    out: dict[str, list[tuple[str, str]]] = {}
    for doc_id, cs in by_doc.items():
        tables = {c.meta["table_no"]: c for c in cs if c.modality == Modality.TABLE and "table_no" in c.meta}
        rows_of: dict[str, list[Chunk]] = {}
        for c in cs:
            if c.modality == Modality.TABLE_ROW and c.meta.get("parent_table"):
                rows_of.setdefault(c.meta["parent_table"], []).append(c)
        figs = {c.meta["figure_no"]: c for c in cs if c.modality in (Modality.FIGURE, Modality.CHART) and "figure_no" in c.meta}
        for c in cs:
            for ref in c.meta.get("refs", []):
                t = ref["type"]
                targets: list[Chunk] = []
                if t == "table" and ref["no"] in tables:
                    tb = tables[ref["no"]]
                    targets = [tb] + rows_of.get(tb.id, [])
                elif t == "figure" and ref["no"] in figs:
                    targets = [figs[ref["no"]]]
                elif t == "section":
                    no = ref["no"]
                    targets = [x for x in cs if x.section_no == no or (x.section_no and x.section_no.startswith(no + "."))]
                label = f"{t} {ref['no']}"
                for tgt in targets:
                    if tgt.id != c.id:
                        out.setdefault(c.id, []).append((tgt.id, label))
    return out


class KnowledgeGraph:
    def __init__(self, chunks: list[Chunk]):
        self.g = nx.Graph()
        self.refs = resolve_refs(chunks)
        by_section: dict[tuple[str, str], list[Chunk]] = {}
        for c in chunks:
            self.g.add_node(f"c:{c.id}", kind="chunk", label=c.content[:60], modality=c.modality.value, doc=c.doc_id, page=c.page)
            for e in c.entities:
                self.g.add_node(f"e:{e}", kind="entity", label=e)
                self._edge(f"c:{c.id}", f"e:{e}", 1.0, "mentions")
            if c.modality == Modality.TABLE_ROW and c.meta.get("parent_table"):
                self._edge(f"c:{c.id}", f"c:{c.meta['parent_table']}", 1.0, "row_of")
            by_section.setdefault((c.doc_id, c.section_no or c.section), []).append(c)
        for cs in by_section.values():
            for a, b in zip(cs, cs[1:]):
                self._edge(f"c:{a.id}", f"c:{b.id}", 0.5, "adjacent")
            warns = [x for x in cs if x.modality == Modality.WARNING]
            for w in warns:
                for x in cs:
                    if x.id != w.id and x.modality != Modality.WARNING:
                        self._edge(f"c:{w.id}", f"c:{x.id}", 0.8, "warns")
        for src, targets in self.refs.items():
            for tgt, label in targets:
                self._edge(f"c:{src}", f"c:{tgt}", 1.6, f"refers to {label}")

    def _edge(self, a: str, b: str, w: float, kind: str) -> None:
        if a == b:
            return
        if self.g.has_edge(a, b):
            if w > self.g[a][b]["weight"]:
                self.g[a][b].update(weight=w, kind=kind)
        else:
            self.g.add_edge(a, b, weight=w, kind=kind)

    def ppr(self, seed_chunks: dict[str, float], seed_entities: Optional[Iterable[str]] = None, alpha: float = 0.8) -> dict[str, float]:
        """Personalized PageRank. Returns chunk-id -> score (chunk nodes only)."""
        pers: dict[str, float] = {}
        for cid, w in seed_chunks.items():
            if f"c:{cid}" in self.g:
                pers[f"c:{cid}"] = pers.get(f"c:{cid}", 0.0) + w
        for e in seed_entities or []:
            if f"e:{e}" in self.g:
                pers[f"e:{e}"] = pers.get(f"e:{e}", 0.0) + 0.5 * (sum(seed_chunks.values()) / max(len(seed_chunks), 1) or 1.0)
        if not pers:
            return {}
        res = nx.pagerank(self.g, alpha=alpha, personalization=pers, weight="weight", max_iter=100, tol=1e-8)
        return {k[2:]: v for k, v in res.items() if k.startswith("c:")}

    def neighbours(self, chunk_id: str, kinds: Optional[set[str]] = None) -> list[tuple[str, str, float]]:
        node = f"c:{chunk_id}"
        if node not in self.g:
            return []
        out = []
        for nb, data in self.g[node].items():
            if nb.startswith("c:") and (kinds is None or any(k in data["kind"] for k in kinds)):
                out.append((nb[2:], data["kind"], data["weight"]))
        return out

    def subgraph(self, chunk_ids: Iterable[str], hops: int = 1, max_nodes: int = 60) -> dict:
        seeds = [f"c:{c}" for c in chunk_ids if f"c:{c}" in self.g]
        keep = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for nb, d in self.g[n].items():
                    if d["kind"] == "adjacent" and nb not in keep and len(keep) > max_nodes // 2:
                        continue
                    nxt.add(nb)
            keep |= nxt
            frontier = nxt
            if len(keep) >= max_nodes:
                break
        keep = set(list(seeds) + [n for n in keep if n not in seeds][: max(0, max_nodes - len(seeds))])
        nodes = [{"id": n, **{k: v for k, v in self.g.nodes[n].items()}, "seed": n in seeds} for n in keep]
        edges = [{"source": a, "target": b, "kind": d["kind"], "weight": d["weight"]} for a, b, d in self.g.subgraph(keep).edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        kinds = {}
        for _, d in self.g.nodes(data=True):
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), **kinds,
                "cross_references": sum(len(v) for v in self.refs.values())}
