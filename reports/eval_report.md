# Meridian evaluation report

93 golden questions (50 train / 43 held-out test) over 6 documents and 157 chunks. Runtime 46.0 s.

## 1. Retrieval ablation (held-out test split)

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| bm25 | 84.2% | 97.4% | 97.4% | 97.4% | 0.850 | 0.874 | 3 |
| dense | 71.0% | 89.5% | 92.1% | 92.1% | 0.792 | 0.802 | 4 |
| hybrid_rrf | 84.2% | 94.7% | 97.4% | 97.4% | 0.847 | 0.857 | 6 |
| hybrid+entity+fact | 63.2% | 86.8% | 92.1% | 93.4% | 0.708 | 0.742 | 5 |
| +prf+graph | 73.7% | 92.1% | 94.7% | 94.7% | 0.786 | 0.795 | 8 |
| +late_rerank | 89.5% | 100.0% | 100.0% | 100.0% | 0.879 | 0.899 | 11 |
| full | 89.5% | 100.0% | 100.0% | 100.0% | 0.879 | 0.899 | 11 |

Same ablation over all questions (train + test):

| Configuration | Hit@1 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| bm25 | 79.5% | 91.6% | 0.819 | 0.862 |
| dense | 68.7% | 89.2% | 0.767 | 0.798 |
| hybrid_rrf | 78.3% | 94.0% | 0.810 | 0.842 |
| hybrid+entity+fact | 59.0% | 86.8% | 0.700 | 0.718 |
| +prf+graph | 63.9% | 89.2% | 0.732 | 0.756 |
| +late_rerank | 79.5% | 98.8% | 0.835 | 0.868 |
| full | 79.5% | 98.8% | 0.835 | 0.868 |

## 2. Learned ranker (trained on train split, evaluated on test)

Adopted: **False**. Hit@1 89.5% → 84.2%; MRR 0.879 → 0.853.

Top learned feature weights: `char` +0.94, `rrf` +0.94, `is_row` -0.86, `modality` +0.83, `dense` +0.78, `graph` -0.66

## 3. End-to-end answers

| Metric | All | Test split |
|---|---|---|
| Answer correctness | 93.1% | 92.5% |
| Abstention recall (unanswerable declined) | 100.0% | 100.0% |
| Abstention precision | 85.7% | 75.0% |
| False-abstention rate | 1.1% | 2.5% |
| Faithfulness (verified claims) | 100.0% | 100.0% |
| Citation recall | 100.0% | 100.0% |

Latency p50-ish mean 43 ms. Errors: none.

Correctness by question type:

| Type | Correct |
|---|---|
| ambiguous | 100.0% |
| analytics | 100.0% |
| chart | 100.0% |
| conflict | 100.0% |
| false_premise | 100.0% |
| figure | 100.0% |
| image | 75.0% |
| multihop | 40.0% |
| procedure | 75.0% |
| scanned | 100.0% |
| spec | 100.0% |
| table | 100.0% |
| text | 100.0% |
| troubleshooting | 100.0% |
| unanswerable | 100.0% |

## 4. Confidence calibration (test split)

ECE raw 0.027 → calibrated 0.008; Brier raw 0.049 → calibrated 0.048 (Platt parameters fitted on the train split; adopted: True).

## 5. Hallucination-injection benchmark

81 clean answers; false-positive rate on clean answers 12.3%. 254 injected hallucinations, overall detection 91.7%.

| Injection | Count | Detected |
|---|---|---|
| numeric | 64 | 87.5% |
| fabricated | 71 | 94.4% |
| negation | 48 | 93.8% |
| entity | 71 | 91.5% |

## 6. Failures (6)

- `q049` (procedure) What safety precautions apply before servicing the HX-500? → missing ['abstained']
- `q051` (procedure) How should the HX-200 suction piping be arranged? → missing ['five times']
- `q066` (multihop) Which torque applies to the plug that is refitted in the HX-500 oil change procedure? → missing ['50 nm']
- `q067` (multihop) In the HX-200 impeller replacement procedure, what torque goes on the casing bolts? → missing ['110 nm']
- `q069` (multihop) Alarm E-31 on the HX-200 points to lubrication. How often is that task due? → missing ['2000 h']
- `q086` (image) What oil capacity does this machine have? → missing ['38 l']
