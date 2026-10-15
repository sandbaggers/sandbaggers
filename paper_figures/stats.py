"""Lightweight uncertainty helpers for the recall metrics — stdlib only (no scipy/numpy).

A point estimate ("recall 73.2%") is not enough to compare two checkpoints or to claim a gap is
real: with rates near 0/1, the sampling error is easily several points. These helpers attach a
95% confidence interval to a recall and to the easy − hard gap (emit_rv5.py), so reported
numbers carry their uncertainty.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence, Union

# A per-sample-id outcome: a single bool/float (1 pass) or a sequence of per-pass 0/1 (multipass).
Outcome = Union[bool, float, Sequence[float]]
Values = Mapping[str, Outcome]


# --------------------------------------------------------------------------- #
# Cluster bootstrap over SAMPLE IDS — the one robust CI for multipass metrics.
#
# A multipass probe scans each sample id N times (N=5), so the N passes of one id are correlated
# and must NOT be treated as independent draws. Reporting mean±std over the 5 per-pass dataset means
# (the old approach) captures only generation noise and understates the corpus-sampling error. These
# helpers resample sample IDS with replacement — a resampled id brings ALL its passes — so the
# interval folds in both sources of uncertainty, and collapses to Wilson-like widths in the 1-pass
# limit. Single-pass bools flow through the same path (one "pass" per id), so the same functions
# serve the single-pass scorecard and the 5-pass drivers.
# --------------------------------------------------------------------------- #
def _passes(v: Outcome) -> list[float]:
    """Normalize a per-id outcome to a list of per-pass 0/1 floats."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def _id_mean(v: Outcome) -> float:
    p = _passes(v)
    return sum(p) / len(p) if p else 0.0


def _pooled_mean(values: Values, ids: Sequence[str]) -> float:
    """Mean over ids of each id's pass-mean (equals the pooled per-pass mean when pass counts match)."""
    return sum(_id_mean(values[i]) for i in ids) / len(ids) if ids else 0.0


def cluster_mean_ci(values: Values, ids: Sequence[str] | None = None, *,
                    n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Recall point estimate + 95% cluster-bootstrap CI over sample ids.

    `values` maps id -> outcome (bool, float, or a list of per-pass 0/1). `ids` restricts/orders the
    id set (default: all keys). Each bootstrap replicate resamples ids with replacement, keeping every
    pass of a resampled id together. Returns (point, lo, hi).
    """
    ids = list(values) if ids is None else list(ids)
    if not ids:
        return (float("nan"), float("nan"), float("nan"))
    point = _pooled_mean(values, ids)
    rng = random.Random(seed)
    boots = sorted(_pooled_mean(values, [ids[rng.randrange(len(ids))] for _ in ids])
                   for _ in range(n_boot))
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return (point, lo, hi)


def group_diff_ci(values: Values, ids_a: Sequence[str], ids_b: Sequence[str], *,
                  n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """mean(ids_a) − mean(ids_b) + 95% CI, UNPAIRED cluster bootstrap (resample each group
    INDEPENDENTLY). For a difference between two DISJOINT groups of samples — e.g. the easy−hard
    recall gap. Returns (diff, lo, hi)."""
    ids_a, ids_b = list(ids_a), list(ids_b)
    if not ids_a or not ids_b:
        return (float("nan"), float("nan"), float("nan"))
    point = _pooled_mean(values, ids_a) - _pooled_mean(values, ids_b)
    rng = random.Random(seed)
    boots = sorted(
        _pooled_mean(values, [ids_a[rng.randrange(len(ids_a))] for _ in ids_a])
        - _pooled_mean(values, [ids_b[rng.randrange(len(ids_b))] for _ in ids_b])
        for _ in range(n_boot))
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return (point, lo, hi)
