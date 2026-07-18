"""Evaluate a sentence encoder on STS-B and compute SimCSE geometry metrics.

Downstream quality is Spearman correlation between cosine similarity and human
STS-B scores. For the paper-style geometric analysis, alignment uses STS-B
pairs with normalized score > 0.8, corresponding to the paper's original
score > 4 threshold. Uniformity uses all unique STS-B sentences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr

from generate_embeddings import encode_sentences
from geometric_metrics import alignment, uniformity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint folder or Hugging Face model id.")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=32)
    parser.add_argument(
        "--pooler-type",
        choices=["cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls_before_pooler",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.8,
        help="STS-B scores are normalized to [0,1]; >0.8 corresponds to original >4/5.",
    )
    parser.add_argument("--uniformity-max-pairs", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/stsb_baseline.json"))
    args = parser.parse_args()

    dataset = load_dataset("sentence-transformers/stsb", split=args.split)
    sentence_1 = list(dataset["sentence1"])
    sentence_2 = list(dataset["sentence2"])
    scores = np.asarray(dataset["score"], dtype=np.float64)

    # Encode each unique sentence once, then map back to pairs.
    unique_sentences = list(dict.fromkeys(sentence_1 + sentence_2))
    unique_embeddings = encode_sentences(
        sentences=unique_sentences,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        pooler_type=args.pooler_type,
        normalize=True,
        device_name=args.device,
    )
    sentence_to_index = {sentence: index for index, sentence in enumerate(unique_sentences)}
    indices_1 = np.asarray([sentence_to_index[sentence] for sentence in sentence_1])
    indices_2 = np.asarray([sentence_to_index[sentence] for sentence in sentence_2])
    embeddings_1 = unique_embeddings[indices_1]
    embeddings_2 = unique_embeddings[indices_2]

    cosine_similarities = np.sum(embeddings_1 * embeddings_2, axis=1)
    spearman_result = spearmanr(cosine_similarities, scores)
    spearman = float(spearman_result.statistic)
    p_value = float(spearman_result.pvalue)

    positive_mask = scores > args.positive_threshold
    if not np.any(positive_mask):
        raise RuntimeError("No STS-B positive pairs remain after applying the threshold.")

    alignment_score = alignment(
        embeddings_1[positive_mask],
        embeddings_2[positive_mask],
        alpha=2.0,
    )
    uniformity_score = uniformity(
        unique_embeddings,
        t=2.0,
        max_pairs=args.uniformity_max_pairs,
        seed=args.seed,
    )

    results = {
        "model": args.model,
        "dataset": "sentence-transformers/stsb",
        "split": args.split,
        "num_pairs": len(scores),
        "num_unique_sentences": len(unique_sentences),
        "num_alignment_positive_pairs": int(np.sum(positive_mask)),
        "positive_threshold_normalized": args.positive_threshold,
        "pooler_type": args.pooler_type,
        "max_seq_length": args.max_seq_length,
        "stsb_spearman": spearman,
        "stsb_spearman_percent": 100.0 * spearman,
        "stsb_spearman_p_value": p_value,
        "alignment": alignment_score,
        "uniformity": uniformity_score,
        "lower_alignment_is_better": True,
        "lower_uniformity_is_better": True,
        "alignment_note": (
            "Paper-style alignment uses human STS-B scores to select positive pairs; "
            "it is therefore not fully label-free in this evaluation."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()