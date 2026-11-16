# Module guide

| Module | Responsibility | Key technique |
|---|---|---|
| `config.py` | Settings from environment variables | dataclasses; no global state |
| `schemas.py` | Pydantic models: `Chunk`, `Document`, `Fact`, `Citation`, `ClaimVerdict`, `AgentResult`, `TraceEvent` | one typed contract between layers |
| `utils/text.py`, `units.py`, `resources/synonyms.json` | tokenizer with synonym canonicalisation + stemming; quantity extraction, unit conversion, tolerance matching | domain thesaurus; units-aware equality (10 bar = 145 psi) |
| `ingest/pdf.py` | PDF → chunks | font-size heading detection, pdfplumber table extraction → table + row chunks + facts, call-out boxes → WARNING chunks, figure crops, page render, per-chunk bounding boxes, automatic OCR for scanned pages |
| `ingest/ocr.py` | Tesseract wrapper | deskew, adaptive threshold, multi-pass `ocr_best`, alarm-code regex repair |
| `ingest/figures.py` | Figure analysis | classify chart/diagram/photo/document; **bar/line chart digitiser** (axis calibration, voting tick-label OCR, label repair, interpolation); visual descriptor |
| `ingest/office.py`, `images.py`, `common.py`, `pipeline.py` | DOCX, CSV/XLSX, TXT/MD, images; shared chunk builder, entity/reference extraction, dispatcher | log files become datasets + aggregate chunks |
| `index/lexical.py` | BM25 (own implementation, field boosts) and char-n-gram TF-IDF | typo-tolerant to OCR noise |
| `index/dense.py` | LSA embeddings (TF-IDF → SVD), optional sentence-transformers, visual index | pluggable backend |
| `index/graph.py` | Knowledge graph + cross-reference resolution | Personalized PageRank retrieval channel; UI subgraph |
| `index/facts.py` | Fact index and `resolve_conflicts` | provenance-aware precedence rules |
| `index/store.py`, `kb.py` | SQLite persistence; `KnowledgeBase` facade with immutable index snapshots, dedupe by SHA-256, model/type masks | atomic rebuild on change |
| `retrieval/analyzer.py` | Query understanding | intent, models, unknown models, codes, quantities, yes/no premise |
| `retrieval/perception.py` | Image understanding | OCR + descriptor + model/alarm detection with warnings when unreadable |
| `retrieval/pipeline.py` | Hybrid retriever + presets for ablation | 8 channels, weighted RRF, MMR |
| `retrieval/rerank.py` | Late interaction, feature extraction, learned ranker | ColBERT-style MaxSim; LTR weights file |
| `retrieval/grader.py` | CRAG-style evidence grading | IDF coverage, OOV handling, brand-term exclusion |
| `retrieval/context.py` | Evidence selection | follows "Table N" references to the exact row, attaches same-section warnings |
| `agent/tools.py` | `lookup_spec`, `check_premise`, chart reader, safe analytics | attribute-aware scoring; analytics via vetted operations, expression shown |
| `agent/compose.py` | Offline cited composer + `CiteBook` | citation numbers in order of first use; branch per intent |
| `agent/agent.py` | Orchestrator | early exits, corrective loop, tool use, verify/repair, calibration, trace |
| `agent/llm_agent.py`, `llm/*` | Optional LLM writer with tool calling | provider-neutral client (Anthropic, OpenAI-compatible), retries/backoff, verified output, fallback |
| `verify/verifier.py` | Claim-level hallucination detection | see ARCHITECTURE §5 |
| `eval/*` | Corpus generator, golden questions, metrics, runner, reports | train/test split, ablations, CI, injection benchmark |
| `api/app.py` + `static/` | Starlette API, SSE, page highlight rendering; single-page UI | no JS build step |

## Advanced RAG techniques, one line each
1. **Contextual chunk prefixes** — every chunk is indexed with `doc title | section`.
2. **Hybrid sparse + dense + char retrieval** fused by **weighted RRF**.
3. **Structured fact retrieval** over typed (value, unit) facts.
4. **Pseudo-relevance feedback (RM3-style)** query expansion.
5. **Graph retrieval** — Personalized PageRank over mention/row/adjacent/warning/cross-reference edges.
6. **Late-interaction (ColBERT-style) reranking.**
7. **Learned-to-rank** with held-out adoption gate.
8. **MMR** diversification.
9. **CRAG** grading with corrective re-retrieval and abstention.
10. **Cross-reference following** and **safety-warning expansion**.
11. **Multimodal**: OCR, chart digitisation, image→figure retrieval, photo perception.
12. **Conflict resolution and false-premise checking** using provenance.
13. **Tool use** for exact computation (analytics, units, interpolation).
14. **Claim-level verification and repair**, **calibrated confidence**.
15. **Evaluation**: ablations, bootstrap CIs, hallucination injection.
