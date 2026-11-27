"""
metrics.py — Metric computation from logged results (Section 3.11).

Computes: rounds-to-convergence, total transmissions, bytes proxy,
convergence speed, and per-client deferral rate, from the per-round JSON logs.
"""

import numpy as np


def rounds_to_convergence(accuracy_curve, peak_fraction=0.95, sustain=5):
    """
    First round at which accuracy exceeds peak_fraction * peak and stays above
    for `sustain` consecutive rounds. Returns len(curve) if never sustained.
    """
    acc = np.asarray(accuracy_curve, dtype=np.float64)
    if len(acc) == 0:
        return None
    threshold = peak_fraction * acc.max()
    for t in range(len(acc) - sustain + 1):
        if np.all(acc[t:t + sustain] >= threshold):
            return t + 1                       # 1-indexed round
    return len(acc)


def bytes_communicated(total_transmissions, param_count, dtype_bytes=4):
    """Bytes proxy = transmissions * parameter count * bytes-per-float."""
    return total_transmissions * param_count * dtype_bytes


def convergence_speed(accuracy_curve, window=30):
    """Mean accuracy gradient dA/dR over the first `window` rounds."""
    acc = np.asarray(accuracy_curve[:window], dtype=np.float64)
    if len(acc) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(acc)), acc, 1)[0])


def deferral_by_client_from_log(round_log):
    """
    Given a run's round_log (list of per-round dicts with a "clients" list,
    as written by RAFAFLStrategy.aggregate_fit), return {cid: [deferred_bool, ...]}
    for use with per_client_deferral_rate().
    """
    out = {}
    for entry in round_log:
        for c in entry.get("clients", []):
            out.setdefault(c["cid"], []).append(bool(c["deferred"]))
    return out


def per_client_deferral_rate(round_logs_per_client):
    """
    Given a dict {cid: [deferred_bool, ...]}, return mean and std of the
    per-client deferral rate across clients.
    """
    rates = [np.mean(v) for v in round_logs_per_client.values() if len(v) > 0]
    if not rates:
        return 0.0, 0.0
    return float(np.mean(rates)), float(np.std(rates))


def summarise_run(accuracy_curve, f1_curve, total_transmissions, param_count,
                  deferral_by_client=None, drift_round=50):
    """Bundle all scalar metrics for a single run into a dict.

    Peak accuracy is the maximum over a noisy curve, so after drift it
    reflects the luckiest round rather than sustained performance. The
    window metrics below report what the model is like over a period:
      final10_accuracy        mean of the last 10 rounds
      post_drift_mean_accuracy mean over rounds after drift_round
      post_drift_min_accuracy  worst round after drift_round
      post_drift_std_accuracy  round-to-round volatility after drift_round
    In no-drift runs the post_drift_* values cover the same rounds, so the
    two settings can be compared directly.
    """
    acc = np.asarray(accuracy_curve, dtype=np.float64)
    post = acc[drift_round:] if len(acc) > drift_round else acc[:0]
    out = {
        "peak_accuracy": float(acc.max()) if len(acc) else None,
        "final10_accuracy": float(acc[-10:].mean()) if len(acc) else None,
        "post_drift_mean_accuracy": float(post.mean()) if len(post) else None,
        "post_drift_min_accuracy": float(post.min()) if len(post) else None,
        "post_drift_std_accuracy": float(post.std()) if len(post) else None,
        "final_accuracy": float(acc[-1]) if len(acc) else None,
        "final_macro_f1": float(f1_curve[-1]) if len(f1_curve) else None,
        "rounds_to_convergence": rounds_to_convergence(acc),
        "total_transmissions": int(total_transmissions),
        "bytes_communicated": bytes_communicated(total_transmissions, param_count),
        "convergence_speed": convergence_speed(acc),
    }
    if deferral_by_client:
        mean, std = per_client_deferral_rate(deferral_by_client)
        out["deferral_rate_mean"] = mean
        out["deferral_rate_std"] = std
    return out
