"""Generate sentence embeddings from a pretrained or fine-tuned encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Requested device is unavailable: {requested}")


def read_sentences(path: Path, max_samples: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sentences: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)
            if max_samples is not None and len(sentences) >= max_samples:
                break
    if not sentences:
        raise ValueError(f"No non-empty sentences found in {path}")
    return sentences


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def pool_output(outputs, attention_mask: torch.Tensor, pooler_type: str) -> torch.Tensor:
    if pooler_type in {"cls", "cls_before_pooler"}:
        # In unsupervised SimCSE with mlp_only_train, this raw CLS vector is used at test time.
        return outputs.last_hidden_state[:, 0]
    if pooler_type == "avg":
        return _masked_mean(outputs.last_hidden_state, attention_mask)
    if pooler_type == "avg_top2":
        hidden = (outputs.hidden_states[-1] + outputs.hidden_states[-2]) / 2.0
        return _masked_mean(hidden, attention_mask)
    if pooler_type == "avg_first_last":
        hidden = (outputs.hidden_states[1] + outputs.hidden_states[-1]) / 2.0
        return _masked_mean(hidden, attention_mask)
    raise ValueError(f"Unsupported pooler_type: {pooler_type}")


def encode_sentences(
    sentences: Iterable[str],
    model_name_or_path: str,
    batch_size: int = 64,
    max_seq_length: int = 32,
    pooler_type: str = "cls_before_pooler",
    normalize: bool = True,
    device_name: str = "auto",
) -> np.ndarray:
    sentences = list(sentences)
    if not sentences:
        raise ValueError("sentences must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = choose_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    need_hidden_states = pooler_type in {"avg_top2", "avg_first_last"}
    all_embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start : start + batch_size]
            batch = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                **batch,
                output_hidden_states=need_hidden_states,
                return_dict=True,
            )
            embeddings = pool_output(outputs, batch["attention_mask"], pooler_type)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint folder.")
    parser.add_argument("--input", type=Path, required=True, help="One sentence per line.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy embedding matrix.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument(
        "--pooler-type",
        choices=["cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    sentences = read_sentences(args.input, args.max_samples)
    embeddings = encode_sentences(
        sentences=sentences,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        pooler_type=args.pooler_type,
        normalize=not args.no_normalize,
        device_name=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    metadata = {
        "model": args.model,
        "input": str(args.input),
        "shape": list(embeddings.shape),
        "pooler_type": args.pooler_type,
        "normalized": not args.no_normalize,
        "max_seq_length": args.max_seq_length,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()