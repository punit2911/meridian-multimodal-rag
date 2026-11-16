"""Generate a deterministic, noisy synthetic corpus for retrieval scale testing."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..utils.text import slugify

_FIELDS = (
    ("maximum working pressure", "bar", 8, 35),
    ("rated flow", "m3/h", 30, 240),
    ("rated motor power", "kW", 4, 90),
    ("maximum fluid temperature", "C", 55, 180),
    ("impeller bolt torque", "Nm", 12, 125),
    ("intake filter replacement interval", "h", 400, 5000),
)

_QUESTION_TEMPLATES = {
    "maximum working pressure": (
        "What is the maximum working pressure for {model}?",
        "How much pressure is {model} rated for?",
        "What pressure limit is listed for {model}?",
    ),
    "rated flow": (
        "What is the rated flow of {model}?",
        "How much flow can {model} deliver?",
        "What flow rate is listed for {model}?",
    ),
    "rated motor power": (
        "What is the rated motor power for {model}?",
        "How much motor power does {model} use?",
        "What motor rating is listed for {model}?",
    ),
    "maximum fluid temperature": (
        "What is the maximum fluid temperature for {model}?",
        "How hot may the fluid get in {model}?",
        "What fluid temperature limit is listed for {model}?",
    ),
    "impeller bolt torque": (
        "What torque applies to the {model} impeller bolt?",
        "What is the impeller bolt torque for {model}?",
        "How much torque is specified for the {model} impeller bolt?",
    ),
    "intake filter replacement interval": (
        "When should the {model} intake filter be replaced?",
        "How often should the {model} intake filter be changed?",
        "What is the replacement interval for the {model} intake filter?",
    ),
}

_NOISY_NOTES = (
    "Field copy; page edges are worn and the scan has mild skew. Confirm the model plate before maintenance.",
    "Technician transcription: table rules were faint, and a repeated footer appears in the source copy.",
    "Archive note: revision lettering is smudged; this copy is for retrieval testing, not operational use.",
    "OCR review note: some characters in the surrounding boilerplate were unclear; numeric limits below are legible.",
    "Service desk copy; line breaks and punctuation follow the scanned field sheet rather than a clean typeset manual.",
)

_UNANSWERABLE_TEMPLATES = (
    "What warranty period is specified for {model}?",
    "What is the purchase price of {model}?",
    "Which grease brand is approved for {model}?",
)


def _record_text(model: str, index: int, rng: random.Random) -> tuple[str, list[dict]]:
    values = []
    for field, unit, low, high in _FIELDS:
        if unit == "C":
            value = rng.randrange(low // 5, high // 5 + 1) * 5
        elif unit == "kW":
            value = rng.randrange(low, high + 1, 2)
        elif unit == "Nm":
            value = rng.randrange(low, high + 1, 5)
        elif unit == "h":
            value = rng.randrange(low // 100, high // 100 + 1) * 100
        else:
            value = rng.randrange(low, high + 1, 5)
        separator = (" : ", " = ", ": ")[(index + len(values)) % 3]
        label = field.upper() if (index + len(values)) % 4 == 0 else field.capitalize()
        line = f"{label}{separator}{value} {unit}"
        if (index + len(values)) % 5 == 0:
            line += "    [field copy]"
        values.append({"field": field, "value": value, "unit": unit, "line": line})

    layout = index % len(_NOISY_NOTES)
    lines = [
        f"# {model} Service Record",
        "",
        f"Document reference: FR-{model.replace('-', '')}-{index + 1:04d} | Revision {chr(65 + index % 4)}",
        "",
        _NOISY_NOTES[layout],
        "",
    ]
    for section, record in enumerate(values, 1):
        lines.extend((f"## {section}. {record['field'].title()}", "", record["line"], ""))
        if (index + section) % 4 == 0:
            lines.extend((f"FIELD COPY | {model} | page {section} | FIELD COPY", ""))
    if index % 3 == 0:
        lines.extend(("Transcription addendum: `l` and `1` were difficult to distinguish in the faded header.", ""))
    return "\n".join(lines), values


def build_stress_corpus(corpus_dir: Path, eval_dir: Path, docs: int = 100, seed: int = 20261116) -> dict:
    """Write `docs` model-specific text records and a held-out-by-model QA set."""
    if docs < 5:
        raise ValueError("The stress corpus needs at least 5 documents for a meaningful held-out split.")
    corpus_dir, eval_dir = Path(corpus_dir), Path(eval_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "_stress_manifest.json"
    if any(corpus_dir.iterdir()) and not manifest_path.exists():
        raise FileExistsError(f"Refusing to write into a non-empty, unmarked corpus directory: {corpus_dir}")
    if any(eval_dir.iterdir()) and not (eval_dir / "_stress_manifest.json").exists():
        raise FileExistsError(f"Refusing to write into a non-empty, unmarked eval directory: {eval_dir}")

    rng = random.Random(seed)
    questions: list[dict] = []
    files: list[str] = []
    for i in range(docs):
        model = f"HX-{100 + i:03d}"
        filename = f"{model}_Field_Service_Record.md" if i % 4 == 0 else f"{model}_Field_Service_Record.txt"
        path = corpus_dir / filename
        text, records = _record_text(model, i, rng)
        path.write_text(text, encoding="utf-8")
        files.append(filename)
        split = "test" if i % 5 == 0 else "train"
        doc_id = slugify(path.stem)
        for field_index, record in enumerate(records):
            templates = _QUESTION_TEMPLATES[record["field"]]
            questions.append({
                "id": f"scale-{i:03d}-{field_index}",
                "type": "synthetic_noisy_spec",
                "question": templates[(i + field_index) % len(templates)].format(model=model),
                "gold_evidence": [{"doc": doc_id, "contains": [record["field"]]}],
                "answer_facts": [{"kind": "num", "value": record["value"], "unit": record["unit"], "tol": 0.01}],
                "expect_abstain": False,
                "image": None,
                "note": "Synthetic stress case; model-level holdout prevents near-duplicate leakage.",
                "split": split,
            })

    negative_count = docs
    for i in range(negative_count):
        model = f"HX-{700 + i:03d}"
        questions.append({
            "id": f"scale-negative-{i:03d}",
            "type": "synthetic_unanswerable",
            "question": _UNANSWERABLE_TEMPLATES[i % len(_UNANSWERABLE_TEMPLATES)].format(model=model),
            "gold_evidence": [],
            "answer_facts": [],
            "expect_abstain": True,
            "image": None,
            "note": "Generated unknown-model abstention case.",
            "split": "test" if i % 5 == 0 else "train",
        })

    eval_dir.joinpath("golden_qa.json").write_text(
        json.dumps({"version": 1, "dataset": "meridian-synthetic-stress-v1", "questions": questions}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "dataset": "meridian-synthetic-stress-v1",
        "seed": seed,
        "documents": docs,
        "answerable_questions": docs * len(_FIELDS),
        "unanswerable_questions": negative_count,
        "total_questions": len(questions),
        "train_questions": sum(q["split"] == "train" for q in questions),
        "test_questions": sum(q["split"] == "test" for q in questions),
        "split_strategy": "model-level holdout (every fifth model is test-only)",
        "limitations": "Synthetic text/Markdown with controlled layout and transcription noise; not real-world manuals.",
        "files": files,
        "synthetic": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (eval_dir / "_stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
