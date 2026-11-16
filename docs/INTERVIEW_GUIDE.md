# Presenting Meridian on a resume and defending it in interviews

## Resume bullets (pick 3–4; keep numbers you can explain)
* Built **Meridian**, a multimodal agentic RAG system for maintenance manuals (PDF tables/charts/scans, DOCX bulletins, CSV logs, alarm-panel photos) with page-level citations, claim-level hallucination verification and calibrated abstention; fully offline-capable, optional Anthropic/OpenAI-compatible LLM mode.
* Designed an 8-channel hybrid retriever (BM25, LSA, char-n-gram, entity, structured-fact, visual, PRF, graph PageRank) with weighted RRF, late-interaction reranking and MMR; **Hit@5 100% / MRR 0.88 on held-out questions vs 97% / 0.85 for BM25** in an ablation study.
* Implemented a **chart digitiser** and **provenance-aware conflict resolver** so the system answers "value at 50% load" from a chart image and prefers a service bulletin over the superseded manual value.
* Built an evaluation harness (93 golden questions, train/test split, bootstrap CIs, ECE/Brier, hallucination-injection benchmark: **91.7% injected-error detection**) and a 37-test pytest suite including live-server API tests; shipped Docker/CI.
* Delivered a streaming web UI (agent trace over SSE, evidence viewer with highlighted bounding boxes, evidence graph, eval dashboard).

One-line pitch: *"A RAG system that treats tables, charts and bulletins as structured evidence, refuses when it can't prove an answer, and ships with the benchmark that measures it."*

## Two-minute walkthrough
1. Problem: wrong torque value = damaged machine; answers live in different modalities and sources disagree.
2. Pipeline: perceive → analyse → hybrid retrieve → grade (CRAG) → assemble evidence (cross-refs, warnings) → tools → compose → verify → calibrate.
3. Differentiators: facts index + conflict resolution, chart digitiser, verifier independent of generator, abstention.
4. Evidence it works: ablation table, held-out numbers, injection benchmark, and the failures I list myself.

## Questions you will get, with defensible answers
**Why not just use a vector DB and an LLM?** Because failure modes here are specific: numeric answers in tables (chunks split rows from headers), conflicting sources (a vector DB has no notion of "supersedes"), charts (no text to embed), and the cost of a confident wrong answer. Each component maps to one of those. The ablation shows dense-only has Hit@1 71% vs 89.5% for the full stack.

**Why LSA instead of neural embeddings?** Deliberate: offline, deterministic, reproducible, zero downloads; domain vocabulary is narrow and the lexical + char + entity channels carry most signal. The backend is an interface (`EmbeddingBackend`); `MERIDIAN_EMBEDDINGS=st` swaps in sentence-transformers. I would measure that swap on the same harness before claiming it helps.

**How do you know it doesn't hallucinate?** I don't claim zero. The offline composer is extractive, and every answer (offline or LLM) passes the claim verifier. I *measured* the verifier by injecting four kinds of errors into verified answers: 91.7% detected; 12% false-positive rate on clean answers (measured with only the cited evidence, so it is pessimistic). Numeric: 87.5%; the misses are perturbations that coincidentally equal another number in the evidence.

**Your learned ranker wasn't adopted — why keep it?** Because the harness decided: with 50 training questions the logistic ranker overfit and lost 5 points Hit@1 on held-out data, so the hand-tuned weights stay. The adoption gate is the point; it would be adopted automatically when larger data makes it win.

**Is the 93% number real?** It is real on a synthetic corpus I generated with exact ground truth, 43 held-out questions — small, so see the bootstrap CIs in the report. It is evidence the pipeline works and a regression guard (a test enforces ≥88%), not a claim about arbitrary real manuals. I'd say that proactively.

**What would you do with real data?** (1) Collect 200+ real questions with gold pages; (2) swap in a neural encoder and measure; (3) retrain the ranker; (4) tune verifier thresholds with the injection benchmark; (5) add table-structure models for scanned tables.

**How does the conflict resolution work?** Facts carry provenance (doc type, effective date, revised/previous column). Group by (model, subject, attribute); revised column > bulletin > manual > older date. Loser kept as a `ConflictNote` so the answer says "manual lists 45 Nm, bulletin SB-2026-07 supersedes with 52 Nm". Chart-derived approximate facts never participate.

**How does the chart reader work?** Detect plot area and axes, calibrate pixel→value from tick labels read with multi-scale OCR voting, repair outlier labels by robust linear fit, then trace bars/line pixels per series; interpolation answers off-grid x. Values are marked approximate and the answer says "about".

**Scalability / production?** Immutable index snapshots swapped atomically; SQLite for metadata; for >100k chunks replace the dense/BM25 in-memory indexes with an ANN/search engine behind the same `score()` interface. Ingestion is the heavy part (OCR/render) and can be moved to a worker queue.

**Security?** Uploads are size-limited and parsed with libraries only (no `eval`; log analytics run vetted operations and display the equivalent pandas expression); page/figure endpoints validate ids and resolve paths under the data dir; UI escapes all text. No auth in the app — put it behind a proxy.

**What's the weakest part?** Multi-hop composition (40% on that type), prose procedure phrasing, data-plate OCR, and verifier false positives. I know because the report lists them.

## Demo script (3 minutes)
1. Load demo. Ask the torque question → show conflict banner, trace, click citation → highlighted page.
2. "Is the HX-200 rated for 150 °C?" → premise correction.
3. "HX-500 specific power at 50% load?" → value read from a chart.
4. Upload `photo_alarm_E47_hx500.png` + "What should I do?" → image-grounded answer.
5. "Who is the CEO of Helix Dynamics?" → principled refusal.
6. Evaluation tab → ablation table, injection results, failures.

## Honest framing
Say "synthetic benchmark" early, show the failures, and explain what you'd change. Interviewers trust measured limits more than perfect numbers.
