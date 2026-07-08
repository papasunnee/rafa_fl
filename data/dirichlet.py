"""
dirichlet.py — Dirichlet non-IID partitioning (Section 3.3) and concept
drift injection (Section 3.4).

Partitioning follows Hsu et al. (2019): for each class, sample a Dirichlet
proportion vector over clients and allocate samples accordingly. Concept drift
is a cyclic class-label permutation applied to a fixed subset of clients at a
given round.
"""

import numpy as np


def dirichlet_partition(labels, num_clients, num_classes, alpha, rng,
                        min_samples=10):
    """
    Partition sample indices across clients using a Dirichlet(alpha) split
    per class. Returns a list of index arrays, one per client.
    """
    labels = np.asarray(labels)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # cumulative cut points
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cuts)
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    # Enforce a minimum allocation by redistributing from the largest clients
    client_indices = [np.array(ci) for ci in client_indices]
    _enforce_minimum(client_indices, min_samples, rng)
    return client_indices


def _enforce_minimum(client_indices, min_samples, rng):
    sizes = [len(ci) for ci in client_indices]
    for k in range(len(client_indices)):
        while len(client_indices[k]) < min_samples:
            donor = int(np.argmax([len(ci) for ci in client_indices]))
            if len(client_indices[donor]) <= min_samples:
                break
            moved = client_indices[donor][-1]
            client_indices[donor] = client_indices[donor][:-1]
            client_indices[k] = np.append(client_indices[k], moved)


def make_drift_clients(num_clients, drift_fraction, drift_seed):
    """Choose which clients receive drift; fixed across methods for a main seed."""
    rng = np.random.RandomState(drift_seed)
    n_drift = int(round(num_clients * drift_fraction))
    return set(rng.choice(num_clients, size=n_drift, replace=False).tolist())


def apply_label_permutation(labels, num_classes):
    """
    Cyclic class-label permutation: c -> (c + 1) mod num_classes.
    Applied to a drift-affected client's labels at the drift round.
    """
    return (np.asarray(labels) + 1) % num_classes
