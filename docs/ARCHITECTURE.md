# Architecture

## 1. The problem and the idea
Maintenance questions are high-stakes and the answers hide in different modalities: a torque in a **table row**, a derating in a **chart image**, a safety rule in a **warning box**, a newer value in a **service bulletin** that overrides the manual, a fault history in a **CSV log**, a code on a **photo** of a panel. A text-chunk RAG gets each of these wrong in a different way. Meridian's idea: *treat every modality as structured, citable evidence, decide first whether an answer is possible, compute what can be computed, and verify what is written.*

## 2. End-to-end flow
```
 question (+ optional image)
        │
 ┌──────▼───────┐  OCR + visual descriptor + alarm-code / model detection on the photo
 │ 1 Perception │
 └──────┬───────┘
 ┌──────▼───────┐  intent (spec / procedure / troubleshoot / figure / analytics / general), models, codes,
 │ 2 Analyzer   │  quantities, yes/no premise, unknown-model detection
 └──────┬───────┘
        ├── unknown model ─────────────────────────────► refuse (never answer for the wrong product)
        ├── analytics intent ─► pandas tool over the log (exact numbers, expression shown)
 ┌──────▼───────────────────────────────────────────┐
 │ 3 Hybrid retrieval (8 channels → RRF → rerank)    │◄─┐
 └──────┬───────────────────────────────────────────┘  │ corrective retries:
 ┌──────▼───────┐  correct / ambiguous / incorrect      │ relaxed model filter,
 │ 4 CRAG grader│───────────────incorrect──────────────┘ keyword-only rewrite → abstain
 └──────┬───────┘
 ┌──────▼───────┐  top-k + follow "see Table 4" cross-references + attach WARNING chunks of the same section
 │ 5 Evidence   │
 └──────┬───────┘
 ┌──────▼───────┐  lookup_spec (+conflict resolution), check_premise, chart reader, unit conversion
 │ 6 Tools      │
 └──────┬───────┘
 ┌──────▼───────┐  offline: evidence-extractive composer with [n] citations      LLM mode: model writes from the
 │ 7 Compose    │                                                                numbered evidence + 2 tools
 └──────┬───────┘
 ┌──────▼───────┐  per-claim: numbers (unit-aware), identifiers/dates, references, negation, lexical coverage
 │ 8 Verify     │──fails──► repair (drop unsupported sentences) / fall back to extractive / abstain
 └──────┬───────┘
 ┌──────▼───────┐  calibrated confidence (Platt on the train split), faithfulness, citations with bbox
 │ 9 Finalise   │
 └──────────────┘
```
Every step emits a `TraceEvent`, streamed to the UI over SSE.

## 3. Decisions and why
| Decision | Reason / trade-off |
|---|---|
| **Offline-first** (LSA + BM25, no LLM required) | Reproducible benchmarks, runs anywhere, no data leaves the machine. Cost: weaker paraphrase understanding than a neural encoder; mitigated by synonym canonicalisation, char-n-grams, late interaction. Backends are pluggable. |
| **Facts as first-class index** (model, subject, attribute, value, unit, provenance) | Lets the system do exact lookup, unit conversion and *conflict resolution* — impossible from text chunks alone. |
| **Rows as chunks** | A question matches one table row, not a 40-row table; the row carries its table caption as context. |
| **Chart digitiser instead of captioning** | Captions say "curve of head vs flow"; users ask for the value at 70 m³/h. Values are flagged `approximate` and excluded from conflict resolution. |
| **Grader before generation** | Cheap, deterministic abstention is safer than asking a generator to "say I don't know". |
| **Verifier is separate from the generator** | Same checks guard the offline composer and any LLM. Independent of what writes the text. |
| **Composer is extractive in offline mode** | Cannot invent facts; fluency is traded for guaranteed faithfulness. |
| **Immutable index snapshots** | Uploads rebuild a new `Indexes` object atomically; readers never see half-built state. |
| **SQLite + files** | Zero-ops persistence; the index is rebuilt from it on start. |

## 4. Retrieval in detail
Channels (each returns a ranked list): `bm25` (field-boosted: title/section/content/entities), `dense` (LSA over word+char n-grams), `char` (OCR-typo tolerant), `entity` (exact alarm code / model), `fact` (structured value search), `visual` (HSV histogram + HOG-like + ink-layout descriptor for query images), `prf` (RM3-style pseudo-relevance feedback), `graph` (Personalized PageRank from top seeds over mention / row-of / adjacent / warns / refers-to edges). **Weighted Reciprocal Rank Fusion** merges them; **late interaction** (ColBERT-style MaxSim over term vectors, with fuzzy matching) rescored the top 40; a **feature-based ranker** (17 features) combines everything — default weights are hand-tuned, a logistic-regression ranker can be trained (`scripts/run_eval.py`) and is adopted only if it beats the default on held-out questions. **MMR** removes near-duplicates. Model filtering applies at document level so an HX-500 passage can never answer an HX-200 question.

## 5. Trust layer
* **Grader** (`retrieval/grader.py`): IDF-weighted query-term coverage in the top evidence + late-interaction + fused score. A term absent from the whole corpus counts as the rarest word — unless the question is anchored on another specific term that *is* found (then it is treated as paraphrase).
* **Conflict resolution** (`index/facts.py`): same (model, subject, attribute) from several documents → prefer `revised` columns, then bulletins over manuals, then latest effective date; the loser is reported as a `ConflictNote` shown in the UI.
* **Verifier** (`verify/verifier.py`): splits the answer into claims; checks numbers with unit conversion, identifiers/dates/references against *cited* evidence, negation introduced without support, and IDF-weighted lexical coverage; verdicts `supported / partial / unsupported / contradicted`; user-quoted numbers inside a correction are allowed.
* **Calibration**: confidence = Platt-scaled mix of grade and faithfulness, fitted on train questions, reported on held-out.

## 6. Evaluation design
Golden questions are generated *with* the corpus so ground truth is exact (answer facts with tolerance and units; gold evidence by document + required phrases). 50/43 train/test split; models trained on train only. Metrics: Hit/Recall@k, MRR, nDCG, unit-aware answer correctness, abstention precision/recall, faithfulness, citation recall, ECE/Brier, bootstrap CIs, latency. The **hallucination-injection benchmark** corrupts verified answers (number, fabricated sentence, negation, entity) and measures detection and false positives.

## 7. Failure modes (known)
Multi-hop questions that need facts from two different passages composed (40% on that type); data-plate OCR on low-contrast photos; prose procedures phrased very differently from the manual; verifier false positives (≈12%) when a correct answer paraphrases heavily; synthetic corpus ≠ real manuals. Each is visible in `reports/eval_report.md` rather than hidden.

## 8. Extending
* New document type: add a parser returning `(Document, [Chunk])` and register the suffix in `ingest/pipeline.py`.
* New retrieval channel: add a scorer in `retrieval/pipeline.py`, a weight in `RetrievalConfig.channel_weights`, a feature in `rerank.FEATURES`.
* New domain: replace `resources/synonyms.json`, units, and the entity regexes in `ingest/common.py`.
* Better embeddings / LLM: see DEPLOYMENT.md.
