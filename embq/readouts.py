"""
Downstream readouts for the embq-harness.

A readout maps a frozen embedding (+ labels) to a scalar the geometric metric
is meant to predict. Kept in its own module so a new domain's readout never
gets entangled with the scRNA block2_* path.

Two readouts, both used across every domain:
  * linear_probe_accuracy -> frozen features + logistic regression (fixed seed)
  * clustering_readout     -> KMeans vs reference labels -> ARI / NMI
"""

from __future__ import annotations

import numpy as np


def linear_probe_accuracy(features, labels, seed=0, test_size=0.5,
                          max_iter=2000, C=1.0):
    """Linear-probe accuracy: frozen features + logistic regression.

    Features are standardized on the train split only, then a multinomial
    logistic regression (lbfgs, fixed seed) is fit and scored on a held-out
    split. Deterministic given `seed`. This is the readout that RankMe-style
    metrics are expected to track.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)

    # Stratify when every class has >= 2 members; otherwise fall back to a
    # plain split (stratification is undefined for singleton classes).
    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size, random_state=seed,
        stratify=stratify,
    )
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                             random_state=seed)
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te))


def clustering_readout(embedding, labels, seed=0):
    """KMeans clustering vs reference labels -> ARI and NMI.

    Mirrors the scRNA downstream scoring (block2_real.downstream_scores) so the
    clustering readout is identical across domains.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels)
    n_types = len(np.unique(labels))
    km = KMeans(n_clusters=n_types, n_init=10, random_state=seed).fit(embedding)
    return {
        "ARI": float(adjusted_rand_score(labels, km.labels_)),
        "NMI": float(normalized_mutual_info_score(labels, km.labels_)),
    }
