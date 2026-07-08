"""
drift_score.py — KL-divergence drift score estimator.

Implements d_t = KL(P_hat_t || P_hat_0) over categorical label distributions,
with Laplace smoothing for numerical stability (Section 3.6.1 of the thesis).
Pure NumPy — no torch dependency — so it is unit-testable in isolation.
"""

import numpy as np


def label_histogram(labels, num_classes, epsilon=1e-8):
    """Return a smoothed probability distribution over classes from a label array."""
    labels = np.asarray(labels).ravel()
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts += epsilon                      # Laplace smoothing
    return counts / counts.sum()


def kl_divergence(p_t, p_0, epsilon=1e-8):
    """
    Discrete KL divergence KL(p_t || p_0).

    Both inputs are probability vectors of equal length. The convention
    0 * log(0/q) = 0 is enforced by masking zero-probability classes.
    """
    p_t = np.asarray(p_t, dtype=np.float64)
    p_0 = np.asarray(p_0, dtype=np.float64)
    p_0 = np.clip(p_0, epsilon, None)      # guard against zeros in reference
    mask = p_t > 0                          # 0*log(0/q) = 0 convention
    return float(np.sum(p_t[mask] * np.log(p_t[mask] / p_0[mask])))


class DriftEstimator:
    """
    Maintains the reference label distribution P_hat_0 and computes the drift
    score against a current batch's label distribution.
    """

    def __init__(self, num_classes, epsilon=1e-8):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.p_0 = None

    def set_reference(self, labels):
        """Record P_hat_0 from the client's initial local labels (round 0)."""
        self.p_0 = label_histogram(labels, self.num_classes, self.epsilon)
        return self.p_0

    def score(self, labels):
        """Compute d_t = KL(P_hat_t || P_hat_0) for the current labels."""
        if self.p_0 is None:
            raise RuntimeError("Reference distribution not set; call set_reference first.")
        p_t = label_histogram(labels, self.num_classes, self.epsilon)
        return kl_divergence(p_t, self.p_0, self.epsilon)
