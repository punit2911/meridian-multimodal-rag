# Meridian evaluation report

700 golden questions (560 train / 140 held-out test) over 100 documents and 700 chunks. Runtime 81.5 s.
Dataset: `meridian-synthetic-stress-v1`; synthetic: `True`; vector store: `memory`.
Split: model-level holdout (every fifth model is test-only). Synthetic text/Markdown with controlled layout and transcription noise; not real-world manuals.

## 1. Retrieval ablation (held-out test split)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| bm25 | 25.0% | 65.8% | 100.0% | 100.0% | 0.471 | 0.601 | 2 |
| dense | 89.2% | 93.3% | 100.0% | 100.0% | 0.925 | 0.943 | 4 |
| hybrid_rrf | 60.0% | 81.7% | 95.8% | 95.8% | 0.727 | 0.794 | 6 |
| hybrid+entity+fact | 16.7% | 86.7% | 100.0% | 100.0% | 0.478 | 0.609 | 6 |
| +prf+graph | 15.8% | 84.2% | 100.0% | 100.0% | 0.461 | 0.596 | 12 |
| +late_rerank | 65.0% | 90.0% | 100.0% | 100.0% | 0.771 | 0.828 | 13 |
| full | 65.0% | 90.0% | 100.0% | 100.0% | 0.771 | 0.828 | 13 |

Same ablation over all questions (train + test):

| Configuration | Hit@1 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 24.7% | 100.0% | 0.469 | 0.600 |
| dense | 88.0% | 100.0% | 0.918 | 0.938 |
| hybrid_rrf | 63.5% | 97.0% | 0.760 | 0.819 |
| hybrid+entity+fact | 17.7% | 100.0% | 0.489 | 0.618 |
| +prf+graph | 15.8% | 100.0% | 0.466 | 0.599 |
| +late_rerank | 61.3% | 100.0% | 0.746 | 0.810 |
| full | 61.3% | 100.0% | 0.746 | 0.810 |
## 3. End-to-end answers

| Metric | All | Test split |
|---|---|---|
| Answer correctness | 91.0% | 90.0% |
| Abstention recall (unanswerable declined) | 100.0% | 100.0% |
| Abstention precision | 65.4% | 62.5% |
| False-abstention rate | 8.8% | 10.0% |
| Faithfulness (verified claims) | 100.0% | 100.0% |
| Citation recall | 100.0% | 100.0% |

Latency p50-ish mean 18 ms. Errors: none.

Correctness by question type:

| Type | Correct |
|---|---|
| synthetic_noisy_spec | 91.0% |
| synthetic_unanswerable | 100.0% |

## 4. Confidence calibration (test split)

ECE raw 0.069 → calibrated 0.002; Brier raw 0.007 → calibrated 0.000 (Platt parameters fitted on the train split; adopted: True).

## 5. Hallucination-injection benchmark

546 clean answers; false-positive rate on clean answers 0.0%. 1637 injected hallucinations, overall detection 100.0%.

| Injection | Count | Detected |
|---|---|---|
| numeric | 546 | 100.0% |
| fabricated | 546 | 100.0% |
| negation | 0 | n/a |
| entity | 545 | 100.0% |

## 6. Failures (54)

- `scale-002-0` (synthetic_noisy_spec) What pressure limit is listed for HX-102? → missing ['abstained']
- `scale-002-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-102? → missing ['abstained']
- `scale-005-0` (synthetic_noisy_spec) What pressure limit is listed for HX-105? → missing ['abstained']
- `scale-005-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-105? → missing ['abstained']
- `scale-008-0` (synthetic_noisy_spec) What pressure limit is listed for HX-108? → missing ['abstained']
- `scale-011-0` (synthetic_noisy_spec) What pressure limit is listed for HX-111? → missing ['abstained']
- `scale-011-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-111? → missing ['abstained']
- `scale-014-0` (synthetic_noisy_spec) What pressure limit is listed for HX-114? → missing ['abstained']
- `scale-014-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-114? → missing ['abstained']
- `scale-017-0` (synthetic_noisy_spec) What pressure limit is listed for HX-117? → missing ['abstained']
- `scale-017-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-117? → missing ['abstained']
- `scale-020-0` (synthetic_noisy_spec) What pressure limit is listed for HX-120? → missing ['abstained']
- `scale-023-0` (synthetic_noisy_spec) What pressure limit is listed for HX-123? → missing ['abstained']
- `scale-026-0` (synthetic_noisy_spec) What pressure limit is listed for HX-126? → missing ['abstained']
- `scale-026-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-126? → missing ['abstained']
- `scale-029-0` (synthetic_noisy_spec) What pressure limit is listed for HX-129? → missing ['abstained']
- `scale-029-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-129? → missing ['abstained']
- `scale-032-0` (synthetic_noisy_spec) What pressure limit is listed for HX-132? → missing ['abstained']
- `scale-035-0` (synthetic_noisy_spec) What pressure limit is listed for HX-135? → missing ['abstained']
- `scale-035-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-135? → missing ['abstained']
- `scale-038-0` (synthetic_noisy_spec) What pressure limit is listed for HX-138? → missing ['abstained']
- `scale-041-0` (synthetic_noisy_spec) What pressure limit is listed for HX-141? → missing ['abstained']
- `scale-041-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-141? → missing ['abstained']
- `scale-044-0` (synthetic_noisy_spec) What pressure limit is listed for HX-144? → missing ['abstained']
- `scale-047-0` (synthetic_noisy_spec) What pressure limit is listed for HX-147? → missing ['abstained']
- `scale-047-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-147? → missing ['abstained']
- `scale-050-0` (synthetic_noisy_spec) What pressure limit is listed for HX-150? → missing ['abstained']
- `scale-050-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-150? → missing ['abstained']
- `scale-053-0` (synthetic_noisy_spec) What pressure limit is listed for HX-153? → missing ['abstained']
- `scale-056-0` (synthetic_noisy_spec) What pressure limit is listed for HX-156? → missing ['abstained']
- `scale-058-0` (synthetic_noisy_spec) How much pressure is HX-158 rated for? → missing ['23 bar']
- `scale-059-0` (synthetic_noisy_spec) What pressure limit is listed for HX-159? → missing ['abstained']
- `scale-059-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-159? → missing ['abstained']
- `scale-062-0` (synthetic_noisy_spec) What pressure limit is listed for HX-162? → missing ['abstained']
- `scale-062-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-162? → missing ['abstained']
- `scale-065-0` (synthetic_noisy_spec) What pressure limit is listed for HX-165? → missing ['abstained']
- `scale-065-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-165? → missing ['abstained']
- `scale-068-0` (synthetic_noisy_spec) What pressure limit is listed for HX-168? → missing ['abstained']
- `scale-071-0` (synthetic_noisy_spec) What pressure limit is listed for HX-171? → missing ['abstained']
- `scale-071-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-171? → missing ['abstained']
- `scale-074-0` (synthetic_noisy_spec) What pressure limit is listed for HX-174? → missing ['abstained']
- `scale-074-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-174? → missing ['abstained']
- `scale-077-0` (synthetic_noisy_spec) What pressure limit is listed for HX-177? → missing ['abstained']
- `scale-077-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-177? → missing ['abstained']
- `scale-080-0` (synthetic_noisy_spec) What pressure limit is listed for HX-180? → missing ['abstained']
- `scale-083-0` (synthetic_noisy_spec) What pressure limit is listed for HX-183? → missing ['abstained']
- `scale-086-0` (synthetic_noisy_spec) What pressure limit is listed for HX-186? → missing ['abstained']
- `scale-086-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-186? → missing ['abstained']
- `scale-089-0` (synthetic_noisy_spec) What pressure limit is listed for HX-189? → missing ['abstained']
- `scale-089-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-189? → missing ['abstained']
- `scale-092-0` (synthetic_noisy_spec) What pressure limit is listed for HX-192? → missing ['abstained']
- `scale-095-0` (synthetic_noisy_spec) What pressure limit is listed for HX-195? → missing ['abstained']
- `scale-095-3` (synthetic_noisy_spec) What fluid temperature limit is listed for HX-195? → missing ['abstained']
- `scale-098-0` (synthetic_noisy_spec) What pressure limit is listed for HX-198? → missing ['abstained']
