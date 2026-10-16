"""The training-dose figures (pirate_training, targeted_training): easy and hard recall of two stages per step count of
one training attack, on one axis (inputs/pirate.json, inputs/targeted.json; the curve schema of data.py, with a tier
per row)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from . import style

MARKER = {"Stage A": "s", "Stage B": "^", "Stage C": "D"}


def load(path: str | Path) -> dict:
    """{label: {dose: {tier: (recall, lo, hi)}}} and the doses, from a curve file."""
    blob = json.loads(Path(path).read_text())
    data: dict = {m["short"]: {} for m in blob["models"]}
    for r in blob["rows"]:
        data[r["model"]].setdefault(int(r["x"]), {})[r["tier"]] = (r["recall"], r["lo"], r["hi"])
    return {"doses": [int(x) for x in blob["xs"]], "data": data}


def draw(ax, data, tier, doses):
    """Easy dashed and hollow, hard solid; bands are the 95% intervals."""
    x = list(range(len(doses)))
    for label, curve in data.items():
        colour, marker = style.STAGE_COLOURS[label], MARKER[label]
        values = [curve[d][tier] for d in doses]
        ax.fill_between(x, [v[1] for v in values], [v[2] for v in values], color=colour,
                        alpha=.08 if tier == "easy" else .12, linewidth=0)
        ax.plot(x, [v[0] for v in values], label=f"{label} · {tier.capitalize()}", color=colour, marker=marker,
                markersize=10, markerfacecolor="white" if tier == "easy" else colour,
                markeredgecolor=colour if tier == "easy" else "white", markeredgewidth=1.4, linewidth=2.2,
                linestyle="--" if tier == "easy" else "-")
    ax.set(xlim=(-.25, len(doses) - .75), ylim=(104, -2.5), xticks=x, xticklabels=doses, yticks=[0, 20, 40, 60, 80, 100],
           xlabel="Training steps", ylabel="Vulnerability recall (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_bounds(0, 100)


def plot(curves: dict) -> Figure:
    """Both tiers of every stage on one axis, the legend above it."""
    with mpl.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
        fig, ax = plt.subplots(figsize=(16, 7.5), layout="none")
    for tier in ("hard", "easy"):
        draw(ax, curves["data"], tier, curves["doses"])
    fig.subplots_adjust(left=.095, right=.985, bottom=.16, top=.76)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    leg = fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, loc="upper center",
                     bbox_to_anchor=(.5, .99), handlelength=2.1, columnspacing=1.2, fontsize=20, frameon=True)
    fig.canvas.draw()
    assert leg.get_window_extent(fig.canvas.get_renderer()).y0 > ax.get_window_extent().y1 + 20, "legend must clear plot"
    return fig
