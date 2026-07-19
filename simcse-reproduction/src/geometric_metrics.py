"""Alignment and uniformity metrics used in the SimCSE analysis.

Both inputs are expected to be sentence embeddings. They are L2-normalized
inside the functions, matching the hypersphere formulation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def alignment(
    positive_embeddings_1: np.ndarray,
    positive_embeddings_2: np.ndarray,
    alpha: float = 2.0,
) -> float:
    """Mean powered Euclidean distance between known positive pairs.

    Lower is better. SimCSE reports alpha=2.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = l2_normalize(positive_embeddings_1)
    y = l2_normalize(positive_embeddings_2)
    if x.shape != y.shape:
        raise ValueError(f"Positive matrices must have the same shape: {x.shape} != {y.shape}")
    if len(x) == 0:
        raise ValueError("At least one positive pair is required.")
    distances = np.linalg.norm(x - y, axis=1)
    return float(np.mean(np.power(distances, alpha)))


def _uniformity_exact(embeddings: np.ndarray, t: float, chunk_size: int) -> float:
    n = len(embeddings)
    total = 0.0
    count = 0
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        x = embeddings[i0:i1]
        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            y = embeddings[j0:j1]
            squared_distances = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
            values = np.exp(-t * squared_distances)
            if i0 == j0:
                rows, cols = np.triu_indices(values.shape[0], k=1)
                selected = values[rows, cols]
            else:
                selected = values.reshape(-1)
            total += float(selected.sum())
            count += int(selected.size)
    if count == 0:
        raise ValueError("At least two embeddings are required for uniformity.")
    return float(math.log(total / count))


def _uniformity_sampled(
    embeddings: np.ndarray,
    t: float,
    max_pairs: int,
    seed: int,
) -> float:
    n = len(embeddings)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=max_pairs)
    right = rng.integers(0, n - 1, size=max_pairs)
    # Map right indices around left so self-pairs are impossible.
    right = right + (right >= left)
    squared_distances = np.sum((embeddings[left] - embeddings[right]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * squared_distances))))


def uniformity(
    embeddings: np.ndarray,
    t: float = 2.0,
    max_pairs: int = 2_000_000,
    seed: int = 42,
    chunk_size: int = 512,
) -> float:
    """Log expected Gaussian potential between pairs on the unit hypersphere.

    Lower is better. Exact unique pairs are used when affordable; otherwise a
    deterministic Monte Carlo estimate is used.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    x = l2_normalize(embeddings)
    n = len(x)
    if n < 2:
        raise ValueError("At least two embeddings are required.")

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        return _uniformity_exact(x, t=t, chunk_size=chunk_size)
    return _uniformity_sampled(x, t=t, max_pairs=max_pairs, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="A .npy matrix for uniformity.")
    parser.add_argument("--positive-a", type=Path, default=None)
    parser.add_argument("--positive-b", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--t", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embedding_matrix = np.load(args.embeddings)
    results = {
        "uniformity": uniformity(
            embedding_matrix,
            t=args.t,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    }

    if (args.positive_a is None) != (args.positive_b is None):
        raise ValueError("Provide both --positive-a and --positive-b, or neither.")
    if args.positive_a is not None:
        results["alignment"] = alignment(
            np.load(args.positive_a),
            np.load(args.positive_b),
            alpha=args.alpha,
        )

    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()