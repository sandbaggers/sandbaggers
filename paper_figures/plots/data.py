"""Data loading for the paper figures: the JSON tables emit.py writes (inputs/*.json).

Each curve cell is a **recall** (percent), the **n** it was measured over, and its measured 95% interval (the
cluster bootstrap over items), so the plot can draw the band. Curve schema::

    {
      "xs": [1, ...],                                   # agent turns, training steps, ...
      "models": [{"label": "...", "short": "..."}],     # in order
      "rows": [{"model": <short>, "x": <int>, "recall": <0-100>, "n": <int>, "lo": <0-100>, "hi": <0-100>}, ...]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class Cell(tuple):
    """`(recall %, n)` plus its MEASURED 95% interval (`lo`, `hi`, in %).

    Indexing stays positional (`c[0]` recall, `c[1]` n); `interval()` returns the measured interval
    (the paper's cluster bootstrap over items).
    """
    lo: float
    hi: float

    def __new__(cls, recall: float, n: int, lo: float, hi: float):
        self = super().__new__(cls, (float(recall), int(n)))
        self.lo, self.hi = lo, hi
        return self

    def interval(self) -> tuple[float, float]:
        return (float(self.lo), float(self.hi))


def _cell(r: dict) -> Cell:
    return Cell(r["recall"], r["n"], r["lo"], r["hi"])


@dataclass(frozen=True)
class CurveTable:
    """Hard-tier (no CoT) recall along some x-axis — agent turns, training steps, …."""

    xs: list[int]
    models: list[tuple[str, str]]
    rows: dict[tuple[str, int], Cell]  # (model short, x) -> (recall, n)

    def curve(self, model_short: str) -> list[Cell]:
        return [self.rows[(model_short, x)] for x in self.xs if (model_short, x) in self.rows]

    def curve_xs(self, model_short: str) -> list[int]:
        return [x for x in self.xs if (model_short, x) in self.rows]


def _load_curves(path: str | Path) -> CurveTable:
    blob = json.loads(Path(path).read_text())
    models = [_model_entry(m) for m in blob["models"]]
    rows = {(r["model"], int(r["x"])): _cell(r) for r in blob["rows"]}
    return CurveTable(xs=[int(x) for x in blob["xs"]], models=models, rows=rows)


#: Threat-model symbol each reference letter is drawn as. The untrusted model is
#: written ``U`` in the data files but reads as $B$ in the figures and the text.
MATH_SYMBOL = {"T": "T", "U": "B"}


def math_label(label: str) -> str:
    """Set a standalone ``T``/``U`` in a model label as maths: ``Proxy $T$``.

    The two reference models are named after the threat-model symbols, so in the
    figures they should read as the same symbols the text uses. Only a
    free-standing letter is wrapped — ``TAR``, ``TD`` and the like are words.
    """
    return re.sub(r"(?<![\w$])([TU])(?![\w$])",
                  lambda m: f"${MATH_SYMBOL[m.group(1)]}$", label)


def _model_entry(m) -> tuple[str, str]:
    """`{"label", "short"}` or a bare string (the `rejudge_turns.py` curve files)."""
    if isinstance(m, str):
        return (math_label(m), m)
    return (math_label(m["label"]), m.get("short", m["label"]))


def load_turns(path: str | Path) -> CurveTable:
    """Per-turn elicitation-agent curves (turns.json; the curve schema above)."""
    return _load_curves(path)


def load_training(path: str | Path) -> CurveTable:
    """Untargeted-training sweep over step count (training.json; the curve schema above)."""
    return _load_curves(path)


# ── headline metrics (the paper's three properties) ────────────────────────
@dataclass(frozen=True)
class MetricsTable:
    """Selectivity / Robustness / Intactness per organism, each `{value, lo, hi}` in % or
    points, as emit.py writes them (`metrics.json`)."""

    models: list[str]                      # short labels, in order
    labels: dict[str, str]                 # short -> long label
    kinds: dict[str, str]                  # short -> "mo" | "reference"
    values: dict[tuple[str, str], dict]    # (short, metric) -> {value, lo, hi}
    # the baseline-protocol CoT-judge diagnostics (sandbag_intent etc.), short -> {flag: {tier:
    # {k, n}}}, and hard recall per protocol, short -> {protocol: cell}. Both as metrics.json
    # carries them; empty for arms without.
    diagnostics: dict[str, dict] = field(default_factory=dict)
    hard_by_protocol: dict[str, dict] = field(default_factory=dict)

    METRICS = ("selectivity", "robustness", "intactness")

    def get(self, short: str, metric: str) -> dict | None:
        return self.values.get((short, metric))


def load_metrics(path: str | Path) -> MetricsTable:
    blob = json.loads(Path(path).read_text())
    values, labels, kinds, diag, hard = {}, {}, {}, {}, {}
    for short in blob["models"]:
        m = blob["metrics"][short]
        labels[short] = math_label(m.get("label", short))
        kinds[short] = m.get("kind", "mo")
        for metric in MetricsTable.METRICS:
            if m.get(metric) and m[metric].get("value") is not None:
                values[(short, metric)] = m[metric]
        diag[short] = m.get("diagnostics_baseline") or {}
        hard[short] = m.get("hard_recall_by_protocol") or {}
    return MetricsTable(models=list(blob["models"]), labels=labels, kinds=kinds, values=values,
                        diagnostics=diag, hard_by_protocol=hard)


# ── Pareto axes ────────────────────────────────────────────────────────────
# A model organism is only interesting if it is good on *several* axes at once, and pushing
# one usually costs another — so the natural view is a Pareto plot. Every axis (retention,
# selectivity, robustness, μ-decisiveness) is "higher is better".

@dataclass(frozen=True)
class ParetoPoint:
    """One model organism (or one step of one) on two of the axes above."""

    label: str
    x: float
    y: float
    step: int | None = None  # position in a trajectory, if it belongs to one
    target: bool = False     # a goal, not a measurement
    baseline: bool = False   # an honest reference model, not a model organism
    provisional: bool = False  # placeholder number, not a measurement

    @property
    def off_frontier(self) -> bool:
        """Plotted for reference and never part of the Pareto frontier: only a target.

        An honest baseline (T, B) is a measurement and competes for the frontier like any
        organism; it is only DRAWN differently (grey), so the reader can tell reference from
        organism without the frontier hiding where the references actually sit.
        """
        return self.target


@dataclass(frozen=True)
class ParetoTable:
    xlabel: str
    ylabel: str
    points: list[ParetoPoint]
    trajectory: bool = False  # points are ordered steps of one run
    # axis units: "percent" draws a % formatter, "points" plain numbers (the
    # gap is a *difference* of two percentages, so a % sign would misread)
    xunit: str = "percent"
    yunit: str = "percent"


def pareto_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """The non-dominated points, sorted by x.

    Both axes are "higher is better", so ``p`` is dominated iff some other
    point is at least as good on both and strictly better on one.

    Only a target is excluded (``off_frontier``): it dominates everything by
    construction. The honest baselines T and B compete like the organisms —
    on the paper's selectivity axes they are dominated anyway, and on axes
    such as μ-decisiveness, where an honest model is the natural ceiling, it
    is exactly their place on the frontier the figure should show.
    """
    measured = [p for p in points if not p.off_frontier]
    front = [
        p for p in measured
        if not any(
            q is not p and q.x >= p.x and q.y >= p.y and (q.x > p.x or q.y > p.y)
            for q in measured
        )
    ]
    return sorted(front, key=lambda p: p.x)


def load_pareto(path: str | Path) -> ParetoTable:
    """JSON seam: ``{"xlabel": …, "ylabel": …, "trajectory": bool, "xunit": …, "yunit": …,
    "points": [{"label": …, "x": 0-100, "y": 0-100, "step": int|null}, …]}``."""
    blob = json.loads(Path(path).read_text())
    return ParetoTable(
        xlabel=blob["xlabel"],
        ylabel=blob["ylabel"],
        trajectory=bool(blob.get("trajectory", False)),
        xunit=blob.get("xunit", "percent"),
        yunit=blob.get("yunit", "percent"),
        points=[
            ParetoPoint(math_label(p["label"]), float(p["x"]), float(p["y"]), p.get("step"),
                        bool(p.get("target", False)),
                        bool(p.get("baseline", False)),
                        bool(p.get("provisional", False)))
            for p in blob["points"]
        ],
    )
