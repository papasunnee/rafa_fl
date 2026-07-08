"""
stats.py — Statistical validation (Section 3.12).

Paired t-tests between RAFA-FL and each baseline across seeds, with Bonferroni
correction for multiple comparisons, plus Cohen's d effect sizes and 95%
confidence intervals.
"""

import numpy as np

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def paired_ttest(rafa_values, baseline_values):
    """Two-tailed paired t-test on per-seed values. Returns (t, p)."""
    a = np.asarray(rafa_values, dtype=np.float64)
    b = np.asarray(baseline_values, dtype=np.float64)
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy required for paired_ttest")
    t, p = _sps.ttest_rel(a, b)
    return float(t), float(p)


def bonferroni_threshold(alpha, num_comparisons):
    return alpha / num_comparisons


def cohens_d_paired(rafa_values, baseline_values):
    """Cohen's d for paired samples (mean difference / std of differences)."""
    a = np.asarray(rafa_values, dtype=np.float64)
    b = np.asarray(baseline_values, dtype=np.float64)
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def mean_ci(values, confidence=0.95):
    """Mean and half-width of the confidence interval across seeds."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return mean, 0.0
    se = v.std(ddof=1) / np.sqrt(n)
    if _HAVE_SCIPY:
        h = se * _sps.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = se * 1.96
    return mean, float(h)


def compare_all(rafa_by_seed, baselines_by_seed, alpha=0.05, num_comparisons=None):
    """
    rafa_by_seed: list of per-seed metric values for RAFA-FL
    baselines_by_seed: dict {baseline_name: [per-seed values]}
    num_comparisons: fixed family size for the Bonferroni correction. Pass the
        pre-registered number of comparisons (e.g. EXPERIMENT.bonferroni_baselines)
        rather than leaving it to default to len(baselines_by_seed) -- if this is
        called before every baseline has results yet, an inferred count silently
        loosens the correction (interim peeking).
    Returns a dict of comparison results with Bonferroni-corrected verdicts.
    """
    thr = bonferroni_threshold(alpha, num_comparisons if num_comparisons is not None
                               else len(baselines_by_seed))
    rafa_mean, rafa_h = mean_ci(rafa_by_seed)
    out = {"rafa_mean": rafa_mean, "rafa_ci95": rafa_h,
           "bonferroni_threshold": thr, "comparisons": {}}
    for name, vals in baselines_by_seed.items():
        b_mean, b_h = mean_ci(vals)
        t, p = paired_ttest(rafa_by_seed, vals)
        d = cohens_d_paired(rafa_by_seed, vals)
        out["comparisons"][name] = {
            "baseline_mean": b_mean, "baseline_ci95": b_h,
            "t": t, "p": p, "cohens_d": d,
            "significant": p < thr,
        }
    return out
