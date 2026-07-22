"""Evaluate a frozen sentence encoder on GLUE classification tasks (SST-2, QNLI).

This mirrors how the SimCSE paper's own transfer-task evaluation works
(SentEval): the encoder is frozen, sentences are embedded once, and a linear
classifier (here, scikit-learn logistic regression) is fit on top. It answers
a different question than STS-B in `evaluate_sts.py` -- not "does cosine
similarity track semantic similarity" but "is there a linear decision boundary
in this embedding space that separates the classes".

GLUE test-split labels are not public, so classifiers are fit on (a capped
subsample of) `train` and scored on `validation`.

Tasks:
  - sst2: single-sentence binary sentiment classification. Each sentence is
    encoded once; features are the raw embedding.
  - qnli: sentence-pair entailment classification (question, sentence).
    Question and sentence are encoded separately and combined as
    [u, v, |u-v|, u*v], the standard pair-classification feature recipe used
    for NLI probing (concatenation, difference, and elementwise product).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from generate_embeddings import encode_sentences

TASK_FIELDS = {
    "sst2": ("sentence",),
    "qnli": ("question", "sentence"),
}


def _subsample(dataset: Any, max_examples: int | None, seed: int) -> Any:
    if max_examples is None or len(dataset) <= max_examples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_examples))


def _encode_column(
    dataset: Any,
    column: str,
    model_name_or_path: str,
    batch_size: int,
    max_seq_length: int,
    pooler_type: str,
    device_name: str,
) -> np.ndarray:
    return encode_sentences(
        sentences=list(dataset[column]),
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        pooler_type=pooler_type,
        normalize=True,
        device_name=device_name,
    )


def build_features(
    task: str,
    dataset: Any,
    model_name_or_path: str,
    batch_size: int,
    max_seq_length: int,
    pooler_type: str,
    device_name: str,
) -> np.ndarray:
    if task == "sst2":
        return _encode_column(
            dataset, "sentence", model_name_or_path, batch_size, max_seq_length, pooler_type, device_name
        )
    if task == "qnli":
        u = _encode_column(
            dataset, "question", model_name_or_path, batch_size, max_seq_length, pooler_type, device_name
        )
        v = _encode_column(
            dataset, "sentence", model_name_or_path, batch_size, max_seq_length, pooler_type, device_name
        )
        return np.concatenate([u, v, np.abs(u - v), u * v], axis=1)
    raise ValueError(f"Unsupported task: {task}. Supported: {sorted(TASK_FIELDS)}")


def evaluate_classification(
    model: str,
    task: str,
    max_train: int | None = 10_000,
    max_val: int | None = 2_000,
    batch_size: int = 64,
    max_seq_length: int = 32,
    pooler_type: str = "cls_before_pooler",
    device: str = "auto",
    seed: int = 42,
    max_iter: int = 1000,
) -> dict:
    if task not in TASK_FIELDS:
        raise ValueError(f"Unsupported task: {task}. Supported: {sorted(TASK_FIELDS)}")

    raw = load_dataset("glue", task)
    train_split = _subsample(raw["train"], max_train, seed)
    val_split = _subsample(raw["validation"], max_val, seed)

    train_features = build_features(
        task, train_split, model, batch_size, max_seq_length, pooler_type, device
    )
    val_features = build_features(task, val_split, model, batch_size, max_seq_length, pooler_type, device)
    train_labels = np.asarray(train_split["label"], dtype=np.int64)
    val_labels = np.asarray(val_split["label"], dtype=np.int64)

    classifier = LogisticRegression(max_iter=max_iter, random_state=seed)
    classifier.fit(train_features, train_labels)
    accuracy = float(classifier.score(val_features, val_labels))

    return {
        "model": model,
        "task": task,
        "dataset": "glue",
        "classifier": "sklearn.LogisticRegression",
        "feature_recipe": "raw_embedding" if task == "sst2" else "concat[u,v,|u-v|,u*v]",
        "pooler_type": pooler_type,
        "max_seq_length": max_seq_length,
        "n_train": len(train_split),
        "n_val": len(val_split),
        "n_train_total_available": len(raw["train"]),
        "n_val_total_available": len(raw["validation"]),
        "accuracy": accuracy,
        "seed": seed,
    }


def evaluate_all(
    model: str,
    tasks: list[str] | None = None,
    max_train: int | None = 10_000,
    max_val: int | None = 2_000,
    batch_size: int = 64,
    max_seq_length: int = 32,
    pooler_type: str = "cls_before_pooler",
    device: str = "auto",
    seed: int = 42,
) -> dict[str, dict]:
    """Run the given (default: all supported) GLUE tasks and return {task: result}."""
    return {
        task: evaluate_classification(
            model=model,
            task=task,
            max_train=max_train,
            max_val=max_val,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            pooler_type=pooler_type,
            device=device,
            seed=seed,
        )
        for task in (tasks if tasks is not None else sorted(TASK_FIELDS))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint folder or Hugging Face model id.")
    parser.add_argument("--task", choices=sorted(TASK_FIELDS), required=True)
    parser.add_argument("--max-train", type=int, default=10_000, help="0 or negative disables capping.")
    parser.add_argument("--max-val", type=int, default=2_000, help="0 or negative disables capping.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument(
        "--pooler-type",
        choices=["cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate_classification(
        model=args.model,
        task=args.task,
        max_train=(args.max_train if args.max_train > 0 else None),
        max_val=(args.max_val if args.max_val > 0 else None),
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        pooler_type=args.pooler_type,
        device=args.device,
        seed=args.seed,
    )

    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
