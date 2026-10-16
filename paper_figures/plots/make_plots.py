"""Produce the final figures for the paper.

Each figure is a function registered in ``PLOTS``; add a new one by writing a
``plot_*`` function that returns a Figure and registering it. Data comes from
the JSON tables emit.py writes to inputs/, read by ``plots/data.py``:

    python -m plots.make_plots [figure ...]      # -> figures/{pdf,png}/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullLocator, PercentFormatter

from . import dose, style
from .data import (
    CurveTable,
    ParetoPoint,
    ParetoTable,
    pareto_front,
    MetricsTable,
    load_metrics,
    load_pareto,
    load_training,
    load_turns,
    math_label,
)


@dataclass(frozen=True)
class Datasets:
    """Everything the figures can draw from; each plot uses what it needs."""

    turns: CurveTable
    training: CurveTable
    pareto_rr: ParetoTable
    pareto_rs: ParetoTable
    metrics: MetricsTable
    # μ-decisiveness against retention / robustness
    pareto_mr: ParetoTable
    pareto_mb: ParetoTable
    # the four training stages traced through metric pairs (STAGE_PAIRS), one table per panel
    stages: tuple[ParetoTable, ...]
    # capabilities.json, by short label
    capabilities: dict
    # the training-dose curves (dose.load), None where the input is missing
    pirate: dict | None = None
    targeted: dict | None = None
    # the Pareto markers, pinned so every figure draws an arm with the same symbol
    pareto_markers: dict[str, str] | None = None

# Which steps of HARD_RAMP to use for k measured series. Adjacent steps are
# too close to tell apart on their own, so a figure with few series spreads
# across the ramp instead of taking consecutive steps — and skips the light
# end, which is under 3:1 on white. Re-validate if you add a new series count.
RAMP_SUBSET = {1: [3], 2: [2, 5], 6: [0, 1, 2, 3, 4, 5]}


# Markers pinned by label, so a model keeps the same symbol in every figure even when a figure
# shows only a subset of the models; everything else takes its symbol from its position. The
# vetted figures keyed these to an organism list of their own, where SD and SDF were 4th and 3rd.
MODEL_MARKER = {"SD": "D", "SDF": "^"}


def _ramp_colours(k: int) -> list[str]:
    if k in RAMP_SUBSET:
        return [style.HARD_RAMP[i] for i in RAMP_SUBSET[k]]
    return style.step_ramp(k)


def _draw_curves(ax, table, *, xlabel: str, ordinal: bool = False,
                 invert_y: bool = False) -> None:
    """Draw one recall-vs-budget panel into ``ax``; the legend is the caller's.

    One line per model organism on an ordinal ramp of the hard-tier hue
    (darker = later in the list), each with its own marker — shape is what
    keeps the series apart under colour-blindness, since the legend beside the
    plot is the only other identity channel. The 0-100% y-axis is shared with
    the other figures so the paper reads on one scale.
    """
    xs = table.xs
    labels = [label for label, _ in table.models]
    colours = dict(zip(labels, _series_colours(labels)))
    # `ordinal`: the measured x values are drawn equi-spaced, in order, and the axis is labelled
    # with the values (the dose ladder 0, 8, 32, 64, 96, 128 -- on a linear or log axis its
    # points bunch at one end and the tick labels collide)
    pos = {x: (i if ordinal else x) for i, x in enumerate(xs)}
    # realvul5 edit: a series whose every point coincides with a series drawn after it (Stage B and Stage C, both 0%
    # at every agent turn) would vanish under it. It is drawn on top instead, dashed with hollow markers, so both
    # stay visible.
    pts = {short: dict(zip(table.curve_xs(short), (c[0] for c in table.curve(short)))) for _, short in table.models}
    order = [short for _, short in table.models]
    hidden = {s for i, s in enumerate(order) if any(
        pts[s] and all(x in pts[t] and pts[t][x] == y for x, y in pts[s].items()) for t in order[i + 1:])}

    for i, (label, short) in enumerate(table.models):
        cells = table.curve(short)
        mxs = [pos[x] for x in table.curve_xs(short)]
        ys = [c[0] for c in cells]
        if short in hidden:
            ax.plot(mxs, ys, color=colours[label], label=label,
                    marker=MODEL_MARKER.get(label, style.MARKERS[i % len(style.MARKERS)]),
                    markersize=8, markerfacecolor=style.SURFACE, markeredgecolor=colours[label],
                    markeredgewidth=1.8, linewidth=1.8, linestyle=(0, (4, 3)), zorder=4)
            continue
        # the measured interval as a band (cluster bootstrap over items when the table
        # carries one; Wilson otherwise) — light, so four bands stay readable
        los = [c.interval()[0] for c in cells]
        his = [c.interval()[1] for c in cells]
        # 0.2, not 0.12: at 0.12 the tan and sage bands all but vanished while the brown one
        # stayed visible, which read as Base being the least certain series
        ax.fill_between(mxs, los, his, color=colours[label], alpha=0.2, linewidth=0, zorder=2)
        ax.plot(
            mxs, ys,
            color=colours[label], label=label,
            marker=MODEL_MARKER.get(label, style.MARKERS[i % len(style.MARKERS)]),
            markersize=10, markeredgecolor=style.SURFACE, markeredgewidth=1.4,
            linewidth=2.2, zorder=3,
        )

    # no title — the tier/CoT condition belongs in the caption
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vulnerability recall (%)")
    ticks = [pos[x] for x in xs]
    # a quarter-slot of air at each end, so the first and last markers clear the spines
    ax.set_xlim(ticks[0] - 0.25, ticks[-1] + 0.25)
    ax.set_xticks(ticks, [str(x) for x in xs])
    ax.xaxis.set_minor_locator(NullLocator())
    # Float the floor slightly below 0 so a series sitting AT 0 (Strong TAR through the collapsed
    # part of the dose curve; the prompted arms at turn 1) draws just above the spine instead of
    # along it, where the line and its markers are half-hidden by the axis itself.
    ax.set_ylim(-2.5, 104)
    if invert_y:
        # 0% at the TOP: recall recovered by elicitation is a failure of the organism, so the
        # curves fall as the organism gets worse and the best one hugs the top edge.
        ax.invert_yaxis()
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_bounds(0, 100)


def _curves_legend(fig, ax, *, legend_scale: float, legend_above: bool) -> None:
    """The figure-level legend for one or more curve panels, from ``ax``'s handles.

    Taken from ONE axes on purpose: fig.legend() with no handles gathers every axes' lines, so a
    two-panel figure listed each stage twice.
    """
    handles, labels = ax.get_legend_handles_labels()
    n_models = len(labels)
    if legend_above:
        # the title goes INTO the row as a text-only first entry: a legend title is always set on
        # its own line above the entries, and a one-line legend was asked for
        handles = [Line2D([], [], linestyle="none")] + handles
        labels = ["Model organism:"] + labels
        n_models += 1
    # `legend_scale` shrinks only the legend, not the axis text: the curve labels are short and
    # do not need the body size, and at FONT_SCALE the default legend crowds the plot area.
    if legend_above:
        # Across the top in one row: a wide figure has room above the axes but none
        # beside them, and a single row keeps the reading order of the labels the
        # same as the ramp order.
        # one row, no title line: the stage names are self-explanatory and the title cost a
        # second line of legend height on an already short panel
        leg = fig.legend(handles, labels, loc="outside upper left",
                         alignment="left", ncols=n_models,
                         fontsize=style.fs(14 * legend_scale),
                         labelspacing=0.4 * legend_scale, handlelength=1.8 * legend_scale,
                         borderpad=0.4 * legend_scale, handletextpad=0.6 * legend_scale,
                         columnspacing=1.6 * legend_scale)
        # The title entry's (empty) handle still gets a full handle-width slot, which read as a
        # blank gap between the frame and "Model organism:". Collapse that one slot: the legend
        # lays out columns as VPacker[HPacker(handle DrawingArea, text)], sized at draw time.
        try:
            hp = leg._legend_handle_box.get_children()[0].get_children()[0]
            hp.get_children()[0].set_width(0.0)
            hp.sep = 0
        except Exception:
            pass
        # Air between the legend and the top panel. NOT borderaxespad: for an "outside" legend
        # under constrained layout that pad moves the legend INTO the axes rather than the axes
        # away from it. Padding the axes in the layout engine is what actually opens a gap.
        try:
            fig.get_layout_engine().set(h_pad=0.25)
        except Exception:
            pass
    else:
        fig.legend(handles, labels, title="Model organism", loc="outside right upper",
                   alignment="left", ncols=1,
                   fontsize=style.fs(14 * legend_scale),
                   title_fontsize=style.fs(15 * legend_scale),
                   labelspacing=0.4 * legend_scale, handlelength=1.8 * legend_scale,
                   borderpad=0.4 * legend_scale, handletextpad=0.6 * legend_scale)


#: The curve figures' size.
CURVE_SINGLE = (16.0, 7.0)


def _plot_curves(table, *, xlabel: str, ordinal: bool = False,
                 legend_scale: float = 1.0, legend_above: bool = False,
                 figsize: tuple[float, float] = (15.0, 7.0),
                 invert_y: bool = False) -> Figure:
    """One curve panel as its own figure."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_curves(ax, table, xlabel=xlabel, ordinal=ordinal, invert_y=invert_y)
    _curves_legend(fig, ax, legend_scale=legend_scale, legend_above=legend_above)
    return fig


#: The two curve figures show our lineage as training stages: short label in turns.json /
#: training.json -> the name it is drawn under, in this order. Picked by the original short label:
#: the paper's cut (OURS_DROP) does not apply to the figures about the intermediate stages.
STAGES = {"U": "Base", "SD": "Stage A", "KTO+SFT": "Stage B", "Strong TAR": "Stage C"}


def _stage_colour(label: str) -> str | None:
    """The stage colour for ``label``, or None if it is not one of the four stage arms.

    A stage arm goes by several names across the figures -- the stage name on the curve
    figures, the model short elsewhere, its maths form for the reference model (``$B$``), and
    ``Ours`` in the paper's cut -- and takes the same colour under all of them, so a reader
    can carry the stage figures' key over to the scatter plots.
    """
    names = {label, label.replace("$", "")}
    for short, stage in STAGES.items():
        if names & {stage, short, math_label(short), OURS_RENAME.get(short, short)}:
            return style.STAGE_COLOURS[stage]
    return None


def _stage_index(label: str) -> int | None:
    """Position of ``label`` in the Base -> A -> B -> C lineage, or None if it is not a stage."""
    names = {label, label.replace("$", "")}
    for i, (short, stage) in enumerate(STAGES.items()):
        if names & {stage, short, math_label(short), OURS_RENAME.get(short, short)}:
            return i
    return None


def _series_colours(labels: list[str]) -> list[str]:
    """One colour per series: the stage colours when every series is a stage, else the ramp."""
    stage = [_stage_colour(l) for l in labels]
    if labels and all(stage):
        return stage
    return _ramp_colours(len(labels))


def _stage_curves(table: CurveTable) -> CurveTable:
    """``table`` restricted to STAGES, in that order, under the stage names."""
    present = {short for _, short in table.models}
    models = [(name, short) for short, name in STAGES.items() if short in present]
    rows = {k: v for k, v in table.rows.items() if k[0] in STAGES}
    # only the x values some drawn series actually has: the ladder otherwise keeps a tick for a
    # dose only a dropped arm was measured at
    xs = [x for x in table.xs if any((short, x) in rows for _, short in models)]
    return CurveTable(xs=xs, models=models, rows=rows)


def plot_elicitation_turns(data: Datasets) -> Figure:
    """Hard-tier recall (no CoT) recovered by the elicitation agent, by turn, per training stage."""
    return _plot_curves(_stage_curves(data.turns), xlabel="Elicitation agent turns",
                        legend_above=True, figsize=CURVE_SINGLE, invert_y=True)


def plot_untargeted_training(data: Datasets) -> Figure:
    """Hard-tier recall (no CoT) after untargeted training, by step count, per training stage.

    Step counts are drawn equi-spaced (see _draw_curves). Same form as the per-turn figure:
    stage names, 0% at the top.
    """
    return _plot_curves(_stage_curves(data.training), xlabel="Training steps", ordinal=True,
                        legend_above=True, figsize=CURVE_SINGLE, invert_y=True)


# ── the three headline metrics ─────────────────────────────────────────────
# Bare metric names: the definitions live in the caption, not in every figure title.
METRIC_TITLE = {"selectivity": "Selectivity",
                "robustness": "Robustness",
                "intactness": "Retention"}
METRIC_COLOUR = {"selectivity": style.SERIES[0], "robustness": style.SERIES[1],
                 "intactness": style.SERIES[2]}


#: The paper's headline table (Table 3) as a figure: rows and groups exactly as the table has
#: them. Each entry is a (group title, [short labels]) pair; "Ours" is Strong TAR, under either
#: name, so the figure is the same with or without --ours.
TABLE_GROUPS = (
    ("Base models", ["T", "U"]),
    ("", ["Ours"]),
    ("Baselines", ["SDF", "Password-locked", "Expert Iteration", "Reinforcement Learning",
                   "Classifier", "Prompted"]),
)
TABLE_TICK = {"T": "$T$\n(Qwen3-8B)", "U": "$B$\n(Qwen3.6-\n35B-A3B)",
              "Reinforcement Learning": "Reinforcement\nLearning",
              "Expert Iteration": "Expert\nIteration", "Password-locked": "Password-\nlocked"}


def plot_headline_table(data: Datasets) -> Figure:
    """Table 3 as a grouped bar chart: Selectivity, Robustness and Retention per organism.

    One cluster per organism, one bar per metric in the metric's colour, 95% intervals as
    whiskers and the point value above each bar. All three metrics are higher-is-better, so one
    shared 0-100 axis carries them. Reference models are hollow and hatched, as in the other
    headline figures; the Ours cluster sits on a faint band so it reads as the subject.
    """
    t = data.metrics
    metrics = MetricsTable.METRICS
    # cluster pitch > 1 so two-line organism names do not touch; a half-slot between groups
    pitch, width, gap = 1.3, 0.3, 0.55

    def lookup(short):
        for name in ((short, "Strong TAR") if short == "Ours" else (short,)):
            if name in t.models:
                return name
        return None

    xs, spans, cursor = [], [], 0.0
    for gi, (title, shorts) in enumerate(TABLE_GROUPS):
        if gi:
            cursor += gap
        start = cursor
        for short in shorts:
            name = lookup(short)
            if name is None:
                print(f"headline_table: no metrics for {short}; skipped")
                continue
            xs.append((cursor, short, name))
            cursor += pitch
        spans.append((title, start, cursor - pitch))

    fig, ax = plt.subplots(figsize=(19.0, 7.0))
    for x, short, name in xs:
        if short == "Ours":
            ax.axvspan(x - 0.55, x + 0.55, color="#f1f1ef", zorder=0, linewidth=0)
        ref = t.kinds.get(name) == "reference"
        for k, metric in enumerate(metrics):
            v = t.get(name, metric)
            if not v:
                continue
            bx = x + (k - 1) * width
            lo = max(0.0, v["value"] - v["lo"]) if v.get("lo") is not None else 0.0
            hi = max(0.0, v["hi"] - v["value"]) if v.get("hi") is not None else 0.0
            colour = METRIC_COLOUR[metric]
            ax.bar(bx, v["value"], width * 0.92, yerr=[[lo], [hi]], capsize=3,
                   facecolor=style.SURFACE if ref else colour, edgecolor=colour,
                   linewidth=1.3, hatch="///" if ref else None,
                   error_kw=dict(ecolor=style.ERRORBAR, elinewidth=1.1, capthick=1.1),
                   zorder=2)
            ax.text(bx, v["value"] + hi + 1.5, f"{v['value']:.0f}", ha="center", va="bottom",
                    fontsize=style.fs(8.5), color=style.INK_SECONDARY, zorder=3)

    ax.set_xticks([x for x, _, _ in xs])
    ax.set_xticklabels([TABLE_TICK.get(short, short) for _, short, _ in xs])
    ax.tick_params(axis="x", length=0, pad=8, labelsize=style.fs(11.5))
    ax.set_xlim(xs[0][0] - 0.7, xs[-1][0] + 0.7)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Score (higher is better)")
    # group titles over their clusters, in the headroom above 100
    for title, a, b in spans:
        if title:
            ax.text((a + b) / 2, 108, title, ha="center", va="center",
                    fontsize=style.fs(11), color=style.INK_MUTED)
    fig.legend(handles=[Patch(facecolor=METRIC_COLOUR[m], edgecolor=METRIC_COLOUR[m],
                              label=METRIC_TITLE[m]) for m in metrics],
               loc="outside upper center", ncols=len(metrics), frameon=False)
    return fig


#: Bar groups for the headline charts, in order. Short labels; anything unlisted joins the final
#: group. The gap between groups is GROUP_GAP bar-slots.
HEADLINE_GROUPS = (
    ("T", "U"),                                              # honest references
    ("SD", "KTO+SFT", "Strong TAR", "Ours"),                 # the lineage (SDF is a baseline)
)
GROUP_GAP = 0.8


def _grouped_positions(shorts) -> list[float]:
    """y position per short label, with a blank slot inserted between groups."""
    group_of = {}
    for gi, names in enumerate(HEADLINE_GROUPS):
        for n in names:
            group_of[n] = gi
    last = len(HEADLINE_GROUPS)
    pos, cursor, prev = [], 0.0, None
    for s in shorts:
        g = group_of.get(s, last)
        if prev is not None and g != prev:
            cursor += GROUP_GAP
        pos.append(cursor)
        cursor += 1.0
        prev = g
    return pos


def _headline_panel(t, titles, metrics) -> Figure:
    """One HORIZONTAL bar chart per metric in `metrics`, sharing the organism axis.

    Horizontal because the organism labels are long ("Password-locked", "Reinforcement Learning") and
    rotated or wrapped tick labels under vertical bars are the usual way these become unreadable.
    Bars run top-to-bottom in ARMS order, so the y axis is inverted.
    """
    shorts = t.models
    # Visual grouping: references | the self-distillation lineage | everything else. A blank
    # slot between groups is all it takes, and it stops the reference bars from reading as just
    # two more organisms. Membership is by short label; an arm in no group falls to the last one,
    # so adding an arm to ARMS never silently drops it off the chart.
    y = _grouped_positions(shorts)
    span = (max(y) - min(y) + 1) if y else 1
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(max(4.6, 0.42 * max(len(s) for s in shorts) + 3.4) * len(metrics),
                 max(3.4, 0.46 * span + 1.6)),
        sharey=True, squeeze=False)
    axes = list(axes[0])
    for ax, metric in zip(axes, metrics):
        for s, i in zip(shorts, y):
            v = t.get(s, metric)
            if not v:
                continue
            lo = max(0.0, v["value"] - v["lo"]) if v.get("lo") is not None else 0.0
            hi = max(0.0, v["hi"] - v["value"]) if v.get("hi") is not None else 0.0
            ref = t.kinds.get(s) == "reference"
            # a reference model is drawn hollow + hatched in the metric's hue so it reads as
            # "for scale", not as another organism
            ax.barh(i, v["value"], 0.62, xerr=[[lo], [hi]], capsize=4,
                    facecolor=style.SURFACE if ref else METRIC_COLOUR[metric],
                    edgecolor=METRIC_COLOUR[metric], linewidth=1.4,
                    hatch="///" if ref else None,
                    error_kw=dict(ecolor=style.ERRORBAR, elinewidth=1.3, capthick=1.3))
            ax.text(v["value"] + hi + 2.0, i, f"{v['value']:.0f}", ha="left", va="center",
                    fontsize=style.fs(11), color=style.INK_SECONDARY)
        ax.set_title(titles[metric], color=style.INK, loc="left")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_yticks(y)
        ax.set_yticklabels([math_label(s.replace("+", " + ")) for s in shorts],
                           fontsize=style.fs(11))
        ax.grid(axis="y", visible=False)
        ax.spines["bottom"].set_bounds(0, 100)
    axes[0].invert_yaxis()         # ONCE (see above): first arm in ARMS at the top
    # No reference-model legend: the hatched fill already distinguishes T and U, and the caption
    # says what they are. Removed at the author's request.
    return fig


def plot_headline_metric(data: Datasets, metric: str) -> Figure:
    """A single metric as its own horizontal bar chart (one file per metric)."""
    return _headline_panel(data.metrics, dict(METRIC_TITLE), [metric])


# ── Pareto figures ─────────────────────────────────────────────────────────
# Two axes, both "higher is better", so the good corner is always top-right
# and the staircase under the frontier is the achievable region. Dominated
# points are hollow; frontier points are filled and joined, so which artifacts
# are actually on the trade-off curve is readable without the caption.

def _place_labels(ax, points, *, fontsize: float | None = None) -> None:
    """Annotate every point, choosing offsets that do not collide.

    Greedy placement in DISPLAY space, so the real rendered text size is what gets compared.
    Each label tries twelve directions at a ladder of radii and is scored on

    * overlap with already-placed labels, markers, the "better" cue and the drawn lines
      (frontier, lineage), sampled as small boxes;
    * the gap between its point and the nearest edge of the box, so a label is not allowed to
      drift a label-width away for free (clamping a long label to the axes used to do that);
    * a leader line crossing another label, which is what made a floating label read as
      belonging to a neighbour.

    The most crowded points are placed first, while there is still room beside them. A faint
    leader is drawn whenever the rendered gap is more than about half a text height.
    """
    fontsize = style.fs(14) if fontsize is None else fontsize
    fig = ax.figure
    # constrained_layout moves the axes at draw time: settle it, then freeze it, so the box the
    # labels are measured against is the box they are saved in
    fig.canvas.draw()
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed: list[tuple[float, float, float, float]] = []     # labels and markers: full cost
    soft: list[tuple[float, float, float, float]] = []       # line samples: partial cost
    labels: list[tuple[float, float, float, float]] = []     # labels only: leader crossings

    # markers: a keep-out sized to the text, so a label never sits on its own dot
    _r = max(9.0, 0.55 * fontsize * fig.dpi / 72.0)
    disp = {id(p): ax.transData.transform((p.x, p.y)) for p in points}
    for p in points:
        px, py = disp[id(p)]
        placed.append((px - _r, py - _r, px + _r, py + _r))
    for t in ax.texts:
        if t.get_gid() == "better-cue":
            bb = t.get_window_extent(renderer=renderer)
            placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
    # drawn lines (frontier staircase, lineage path): sampled every few pixels
    _lr = 0.25 * fontsize * fig.dpi / 72.0
    segments = [ax.transData.transform(line.get_xydata()) for line in ax.lines]
    # the step arrows of a trajectory are annotations, not lines: sample them the same way
    for t in ax.texts:
        if isinstance(t, mpl.text.Annotation) and t.arrow_patch is not None and not t.get_text():
            segments.append(ax.transData.transform([t.xyann, t.xy]))
    for xy in segments:
        for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / _lr))
            for k in range(n + 1):
                qx, qy = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
                soft.append((qx - _lr, qy - _lr, qx + _lr, qy + _lr))

    def overlap(a, b) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def nearest(box, px, py):
        return (min(max(px, box[0]), box[2]), min(max(py, box[1]), box[3]))

    def crossings(px, py, qx, qy) -> int:
        n = max(2, int(math.hypot(qx - px, qy - py) / 4.0))
        hits = 0
        for k in range(1, n):
            sx, sy = px + (qx - px) * k / n, py + (qy - py) * k / n
            hits += any(l[0] <= sx <= l[2] and l[1] <= sy <= l[3] for l in labels)
        return hits

    ax_bb = ax.get_window_extent()
    dirs = [(math.cos(a), math.sin(a)) for a in (math.radians(d) for d in range(0, 360, 30))]

    # crowded points first
    th0 = fontsize * fig.dpi / 72.0
    def crowd(p):
        px, py = disp[id(p)]
        return sum(math.hypot(px - disp[id(q)][0], py - disp[id(q)][1]) < 4 * th0
                   for q in points if q is not p)
    order = sorted(points, key=crowd, reverse=True)

    for p in order:
        px, py = disp[id(p)]
        txt = ax.text(0, 0, p.label, fontsize=fontsize)
        bb = txt.get_window_extent(renderer=renderer)
        tw, th = bb.width, bb.height
        txt.remove()
        _pad = th * 0.6
        best, best_cost = None, None
        for radius in (th * 0.45, th * 0.7, th * 1.0, th * 1.4, th * 2.0, th * 2.8):
            for ux, uy in dirs:
                ox, oy = px + radius * ux, py + radius * uy
                # anchor the box by the side facing the point
                x0 = ox if ux > 0.3 else (ox - tw if ux < -0.3 else ox - tw / 2)
                y0 = oy if uy > 0.3 else (oy - th if uy < -0.3 else oy - th / 2)
                x0 = min(max(x0, ax_bb.x0 + _pad), ax_bb.x1 - tw - _pad)
                y0 = min(max(y0, ax_bb.y0 + _pad), ax_bb.y1 - th - _pad)
                box = (x0, y0, x0 + tw, y0 + th)
                qx, qy = nearest(box, px, py)
                gap = math.hypot(qx - px, qy - py)
                cost = sum(overlap(box, q) for q in placed)
                cost += 0.35 * sum(overlap(box, q) for q in soft)
                cost += 1.5 * gap
                if gap >= th * 0.6:
                    cost += 400.0 * crossings(px, py, qx, qy)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (box, gap), cost
            # a clean spot close in: stop widening
            if best_cost is not None and best_cost <= 1.5 * best[1] + 1e-9:
                break

        box, gap = best
        ax.annotate(
            p.label, (p.x, p.y), textcoords="offset points",
            # the box was scored in display pixels; offset points are 1/72 inch
            xytext=((box[0] - px) * 72.0 / fig.dpi, (box[1] - py) * 72.0 / fig.dpi),
            ha="left", va="bottom",
            fontsize=fontsize,
            # lineage labels in full ink, comparison organisms a step lighter
            color=style.INK_SECONDARY if _stage_index(p.label) is not None else style.COMPARISON_INK,
            arrowprops=(dict(arrowstyle="-", color=style.COMPARISON, linewidth=0.9,
                             shrinkA=0.0, shrinkB=6.0,
                             relpos=((nearest(box, px, py)[0] - box[0]) / tw,
                                     (nearest(box, px, py)[1] - box[1]) / th))
                        if gap >= th * 0.6 else None),
            # white halo keeps a label legible where it clips a gridline or a line
            path_effects=[pe.withStroke(linewidth=3.0, foreground=style.SURFACE)],
            zorder=5,
        )
        placed.append(box)
        labels.append(box)


def _default_marker(label: str, i: int) -> str:
    """The symbol a point gets when nothing pins one: by model, else by position."""
    return MODEL_MARKER.get(label, style.MARKERS[i % len(style.MARKERS)])


def _plot_pareto(table, *, annotate: bool = True, ax=None,
                 markers: dict[str, str] | None = None, frontier: bool = True) -> Figure:
    """Scatter of model organisms on two axes, with the Pareto frontier drawn.

    A *trajectory* table (one run's successive steps) is additionally joined
    in order and coloured light→dark along the step index, so the path the run
    took through the trade-off — not just where it ended — is visible.

    Pass `ax` to draw into an existing axes (the side-by-side combined figure);
    otherwise this owns its own one-panel figure. `markers` pins a label to a
    symbol; anything not pinned takes its symbol from MODEL_MARKER or its index.
    """
    points = table.points
    # no frontier drawn -> no frontier semantics on the markers either: every point is filled
    front = set(id(p) for p in (pareto_front(points) if frontier else points))

    if ax is None:
        # sqrt, not linear: a canvas that grows as fast as the type cancels the type's growth
        _sc = max(1.0, style.FONT_SCALE) ** 0.5
        fig, ax = plt.subplots(figsize=(9.0 * _sc, 7.5 * _sc))
    else:
        fig = ax.figure

    if table.trajectory:
        colours = _series_colours([p.label for p in points])
        # an arrow per step, so the direction of the run is read without the caption
        for a, b in zip(points, points[1:]):
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                        arrowprops=dict(arrowstyle="-|>", color=style.INK_MUTED, linewidth=1.8,
                                        shrinkA=10, shrinkB=10, mutation_scale=20),
                        zorder=2)
    else:
        # a stage arm keeps its stage colour here too; everything else -- the comparison
        # organisms and the T reference -- is neutral, so colour on a scatter means exactly
        # "this is one of the four lineage stages"
        colours = [_stage_colour(p.label) or style.COMPARISON for p in points]

    # the frontier staircase — the boundary of what the measured artifacts
    # actually achieve
    fp = pareto_front(points) if frontier else []
    if len(fp) > 1:
        # the *pessimistic* staircase: from each frontier point go DOWN first,
        # then right to the next. Every corner it turns is a point that is
        # dominated, so the line hugs the achievable region from below-left
        # and never claims a trade-off the measurements do not support. (The
        # optimistic reading — right, then down — would trace an upper hull
        # through combinations no single artifact achieves.)
        sx, sy = [fp[0].x], [fp[0].y]
        for a, b in zip(fp, fp[1:]):
            sx += [a.x, b.x]
            sy += [b.y, b.y]
        ax.plot(sx, sy, color=style.FRONTIER,
                linewidth=1.4, linestyle="--", alpha=0.8, zorder=1)

    # Labels are placed radially outward from the centre of the cloud, in
    # axis-fraction space so both axes count equally. Points that cluster
    # (the late loop steps pile into one corner) then fan their labels apart
    # instead of stacking, and a point at the edge always gets its label on
    # the inside-free side.
    for i, p in enumerate(points):
        on_front = id(p) in front
        # references (T, B) keep the grey reference colour but are filled/hollow by frontier
        # membership like every other point -- except B, which is the lineage's Base stage and
        # so takes the Base colour
        colour = colours[i] if _stage_colour(p.label) else (
            style.COMPARISON if (p.off_frontier or p.baseline) else colours[i])
        ax.scatter(
            p.x, p.y,
            s=170,
            marker="o" if table.trajectory
            else (markers or {}).get(p.label) or _default_marker(p.label, i),
            facecolor=colour if (on_front or p.off_frontier) else style.SURFACE,
            edgecolor=colour, linewidth=2.0,
            # a placeholder is drawn with a dotted rim, so a figure that is
            # part measurement and part guess cannot be misread as all data
            linestyle=":" if p.provisional else "-",
            zorder=4,
        )
        # labels are placed AFTER the axes are finalised (see _place_labels): their positions
        # depend on the data->display transform, which is not settled until the limits are.

    ax.set_xlabel(table.xlabel)
    ax.set_ylabel(table.ylabel)
    # breathing room: the interesting points sit at the extremes, and a point
    # on the spine reads as clipped
    ax.margins(0.12 * max(1.0, style.FONT_SCALE))
    for axis, unit in ((ax.xaxis, table.xunit), (ax.yaxis, table.yunit)):
        axis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
        if unit == "percent":
            axis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="x", visible=True)
    if frontier:
        _draw_better_cue(ax)
    if annotate:
        _place_labels(ax, points)

    legend = [
        Line2D([], [], color=style.HARD, marker="o", linestyle="none",
               markersize=11, label="on the Pareto frontier"),
        Line2D([], [], color=style.HARD, marker="o", linestyle="none",
               markersize=11, markerfacecolor=style.SURFACE,
               markeredgewidth=2.0, label="dominated"),
    ]
    if any(p.target for p in points):
        legend.append(Line2D([], [], color=style.TARGET, marker="X",
                             linestyle="none", markersize=11,
                             label="target (not a measurement)"))
    if any(p.baseline for p in points):
        legend.append(Line2D([], [], color=style.TARGET, marker="o",
                             linestyle="none", markersize=11,
                             label="honest reference model"))
    if any(p.provisional for p in points):
        # a dotted *line* sample, not another hollow marker: the legend has to
        # name the rim style, which is the only thing separating a placeholder
        # from a measured-but-dominated point
        legend.append(Line2D([], [], color=style.HARD, linestyle=":",
                             linewidth=2.2,
                             label="dotted rim: placeholder, not measured"))
    # No legend: frontier/dominated/baseline are all readable from the marker fill, and on a
    # crowded scatter the box covered whichever artifact landed under it. `legend` is still built
    # above so re-enabling it is a one-line change.
    return fig


def _draw_better_cue(ax) -> None:
    """A small "better ↗" tag in the top-right corner: both axes are higher-is-better."""
    # realvul5 edit: the "\u2197" glyph is set in DejaVu Sans (bundled with matplotlib, so always there): Helvetica
    # Neue, the fallback when Inter is missing, has no such glyph and printed a box. "better" is anchored to its left.
    arrow = ax.text(0.985, 0.985, "↗", transform=ax.transAxes, ha="right", va="top", fontsize=style.fs(12),
                    fontfamily="DejaVu Sans", color=style.INK_MUTED, gid="better-cue", zorder=1)
    ax.annotate("better", xy=(0, 0.5), xycoords=arrow, xytext=(-2, 0), textcoords="offset points", ha="right",
                va="center", fontsize=style.fs(11), color=style.INK_MUTED, gid="better-cue", zorder=1)


CAP_COLS = ["gsm8k", "IFEval (prompt-strict)", "GPQA-diamond", "HumanEval"]
CAP_SHORT = {"gsm8k": "gsm8k", "IFEval (prompt-strict)": "IFEval",
             "GPQA-diamond": "GPQA-d", "HumanEval": "HumanEval"}


def plot_capabilities(data: Datasets) -> Figure:
    """General capability per benchmark, one horizontal panel each, arms in ARMS order.

    The reference rows are hatched, as in the headline metrics, and the base's score is drawn as a
    vertical rule so "did this organism pay a capability price?" is answerable by eye rather than
    by subtracting two bar lengths.
    """
    # capabilities.json is a flat {short: {label, <col>: {acc, se}}} map, in ARMS order
    rows = {k: {**r, "label": OURS_RENAME.get(k, r["label"])} for k, r in data.capabilities.items()
            if k not in OURS_DROP}
    order = [k for k in rows]
    labels = [rows[k]["label"] for k in order]
    refs = {k for k in order if rows[k]["label"].startswith(("T (", "U ("))}
    # Band like the headline plots, by short label (the display labels do not match HEADLINE_GROUPS)
    y = _grouped_positions([_ours_name(k) for k in order])

    # One row: the arm labels are written once on the left and every benchmark is read across a
    # single shared y axis.
    span = (max(y) - min(y) + 1) if y else 1
    fig, axes = plt.subplots(1, len(CAP_COLS),
                             figsize=(3.9 * len(CAP_COLS) + 2.0, max(3.4, 0.46 * span + 1.6)),
                             sharey=True, squeeze=False)
    flat = list(axes[0])
    for ax, col, colour in zip(flat, CAP_COLS, style.SERIES):
        base_acc = (rows.get("U", {}).get(col) or {}).get("acc")
        for i, k in enumerate(order):
            cell = rows[k].get(col) or {}
            acc, se = cell.get("acc"), cell.get("se")
            if acc is None:
                continue
            ax.barh(y[i], 100 * acc, 0.62, xerr=(100 * se if se else None), capsize=4,
                    facecolor=style.SURFACE if k in refs else colour,
                    edgecolor=colour, linewidth=1.4,
                    hatch="///" if k in refs else None,
                    error_kw=dict(ecolor=style.ERRORBAR, elinewidth=1.3, capthick=1.3))
            ax.text(100 * acc + (100 * se if se else 0) + 2.0, y[i], f"{100 * acc:.0f}",
                    ha="left", va="center", fontsize=style.fs(11), color=style.INK_SECONDARY)
        if base_acc is not None:
            ax.axvline(100 * base_acc, color=style.INK_MUTED, linewidth=1.0,
                       linestyle="--", zorder=1)
        ax.set_title(CAP_SHORT[col], color=style.INK, loc="left")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="x", labelsize=style.fs(9.5))
        ax.set_yticks(y)
        ax.set_yticklabels([math_label(t) for t in labels], fontsize=style.fs(11))
        ax.grid(axis="y", visible=False)
        ax.spines["bottom"].set_bounds(0, 100)
    axes[0][0].invert_yaxis()      # ONCE: sharey means one flip covers every panel
    return fig


def plot_pareto_both(data: Datasets) -> Figure:
    """Both Pareto views side by side: retention against robustness, and robustness against selectivity."""
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.5))
    tables = (data.pareto_rr, data.pareto_rs)
    # Draw BOTH panels before placing any label. _place_labels freezes the layout engine, and
    # freezing after only the first panel left the second panel's y label out of the layout --
    # which is what put "Robustness" on top of the left panel.
    for ax, table in zip(axes, tables):
        _plot_pareto(table, ax=ax, annotate=False, markers=data.pareto_markers)
    for ax, table in zip(axes, tables):
        _place_labels(ax, table.points)
    return fig


# ── μ-decisiveness Pareto figures ──────────────────────────────────────────
# Preference coherence (the external fried-model-organisms eval, Modal backend: mu.py; panels.json)
# against the paper metrics it might trade off against. Higher is better on every axis, so the Pareto
# machinery applies unchanged. Prompted and Classifier are not scored by that eval (they have no
# weights of their own to score) and so are absent here.
MU_LABEL = "μ-decisiveness"


def load_pareto_mu(metrics: MetricsTable, path: Path) -> tuple[ParetoTable, ParetoTable]:
    """(μ vs retention, μ vs robustness) from metrics.json + panels.json."""
    results = json.loads(Path(path).read_text())["results"]
    mu = {}
    for short, r in results.items():
        e = (r.get("panel") or {}).get("decisiveness") or {}
        if e.get("point") is not None:
            mu[short] = float(e["point"])
    axes = (("intactness", "Retention", "percent"), ("robustness", "Robustness", "percent"))
    tables = []
    for metric, xlabel, xunit in axes:
        points = []
        for short in metrics.models:      # paper order, so markers line up with the other figures
            v = metrics.get(short, metric)
            if short in mu and v is not None:
                points.append(ParetoPoint(math_label(short), float(v["value"]), mu[short],
                                          baseline=metrics.kinds.get(short) == "reference"))
        tables.append(ParetoTable(xlabel=xlabel, ylabel=MU_LABEL, points=points,
                                  xunit=xunit, yunit="unit"))
    return tuple(tables)


def plot_pareto_mu_all(data: Datasets) -> Figure:
    """μ-decisiveness against retention (left) and robustness (right), side by side.

    Selectivity is left out: every organism scores near the ceiling on it, so that panel added
    width without adding a trade-off to read.
    """
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.5))
    tables = (data.pareto_mr, data.pareto_mb)
    for ax, table in zip(axes, tables):          # both panels first, then labels: see pareto_both
        _plot_pareto(table, ax=ax, annotate=False, markers=data.pareto_markers)
    for ax, table in zip(axes, tables):
        _place_labels(ax, table.points)
    return fig


# ── the training stages through metric pairs ───────────────────────────────
STAGE_INTENT_TIER = "hard"

#: The stage-trajectory figure (stage_trajectories_alt): [(x axis, y axis), ...] over the axes below.
STAGE_PAIRS = [("retention", "selectivity"), ("intent", "mu"), ("agent", "untargeted")]


def load_stage_trajectories(metrics: MetricsTable, mu_path: Path) -> tuple[ParetoTable, ...]:
    """Trajectory tables over Base -> Stage A -> Stage B -> Stage C, one per STAGE_PAIRS panel.

    Axes: μ-decisiveness (Modal panels.json); selectivity and retention (metrics.json);
    verbalized sandbagging intent on the hard tier under the baseline protocol (the CoT-judge
    diagnostic in metrics.json); hard recall after the elicitation agent (10 turns) and after
    untargeted training (64 steps), from the protocol cells in metrics.json -- the same numbers
    as the recall table, not the curve files, whose 64-step point for the swept arms is a
    dose-sweep checkpoint rather than the paper record.
    """
    results = json.loads(Path(mu_path).read_text())["results"]
    mu = {k: ((r.get("panel") or {}).get("decisiveness") or {}).get("point") for k, r in results.items()}

    def intent(short: str) -> float | None:
        d = (metrics.diagnostics.get(short) or {}).get("sandbag_intent") or {}
        c = d.get(STAGE_INTENT_TIER)
        return 100.0 * c["k"] / c["n"] if c and c.get("n") else None

    def proto(short: str, protocol: str) -> float | None:
        c = (metrics.hard_by_protocol.get(short) or {}).get(protocol)
        return None if c is None else float(c["recall"])

    def value(short: str, metric: str) -> float | None:
        v = metrics.get(short, metric)
        return None if v is None else float(v["value"])

    # axis key -> (label, unit, value getter)
    axes = {
        "mu":          ("μ-decisiveness", "unit", lambda s: mu.get(s)),
        "selectivity": ("Selectivity", "points", lambda s: value(s, "selectivity")),
        "retention":   ("Retention", "percent", lambda s: value(s, "intactness")),
        "intent":      ("Sandbagging intent", "percent", intent),
        "agent":       ("Recall: Elicitation agent", "percent", lambda s: proto(s, "agent")),
        "untargeted":  ("Recall: Untargeted training", "percent",
                        lambda s: proto(s, "untargeted_training")),
    }
    tables = []
    for xkey, ykey in STAGE_PAIRS:
        (xlabel, xunit, fx), (ylabel, yunit, fy) = axes[xkey], axes[ykey]
        points = []
        for step, (short, stage) in enumerate(STAGES.items()):
            x, y = fx(short), fy(short)
            if x is None or y is None:
                print(f"stage_trajectories_alt: {stage} has no ({xlabel}, {ylabel}) point; skipped")
                continue
            points.append(ParetoPoint(stage, x, y, step=step))
        tables.append(ParetoTable(xlabel=xlabel, ylabel=ylabel, points=points,
                                  trajectory=True, xunit=xunit, yunit=yunit))
    return tuple(tables)


def plot_stage_trajectories(data: Datasets) -> Figure:
    """The lineage Base -> A -> B -> C traced through metric pairs, one panel each.

    Arrows join consecutive stages. No frontier is drawn: the two recall axes are
    lower-is-better for the organism, so a higher-is-better staircase would mislead. Those two
    axes are instead REVERSED, so 0% recall sits top-right and the good corner is the same one
    as in every other panel.
    """
    tables = data.stages
    fig, axes = plt.subplots(1, len(tables), figsize=(8.75 * len(tables), 7.5), squeeze=False)
    axes = list(axes[0])
    for ax, table in zip(axes, tables):          # all panels first, then labels: see pareto_both
        _plot_pareto(table, ax=ax, annotate=False, frontier=False)
        if table.xlabel.startswith("Recall"):
            ax.invert_xaxis()
        if table.ylabel.startswith("Recall"):
            ax.invert_yaxis()
    for ax, table in zip(axes, tables):
        _place_labels(ax, table.points)
    return fig


# ── the paper's cut ────────────────────────────────────────────────────────
# The intermediate arms of our own lineage are dropped from every figure and the final one is
# named as the method rather than the configuration. Applied ONCE to the loaded Datasets, so
# every figure sees the same arms; capabilities applies it to its own table.
OURS_DROP = {"SD", "KTO+SFT"}
OURS_RENAME = {"Strong TAR": "Ours"}


def _ours_name(s: str) -> str:
    return OURS_RENAME.get(s, s)


def _ours_models(models):
    return [(_ours_name(label), _ours_name(short)) for label, short in models
            if short not in OURS_DROP]


def _ours_curves(t: CurveTable) -> CurveTable:
    return CurveTable(xs=t.xs, models=_ours_models(t.models),
                      rows={(_ours_name(short), x): v for (short, x), v in t.rows.items()
                            if short not in OURS_DROP})


def _ours_metrics(t: MetricsTable) -> MetricsTable:
    keep = [short for short in t.models if short not in OURS_DROP]
    return MetricsTable(
        models=[_ours_name(short) for short in keep],
        labels={_ours_name(short): _ours_name(t.labels[short]) for short in keep},
        kinds={_ours_name(short): t.kinds[short] for short in keep},
        values={(_ours_name(short), m): v for (short, m), v in t.values.items()
                if short not in OURS_DROP})


def _ours_pareto(t: ParetoTable) -> ParetoTable:
    # The frontier is recomputed from the surviving points by _plot_pareto, so dropping a
    # dominated arm never changes it and dropping a frontier arm hands its stretch to the next
    # best surviving point -- never to a dropped one.
    return replace(t, points=[replace(p, label=_ours_name(p.label))
                              for p in t.points if p.label not in OURS_DROP])


def pinned_markers(t: ParetoTable) -> dict[str, str]:
    """label -> marker from ``t``'s point order, so every Pareto figure (which may show a
    different subset of arms) draws an arm with the same symbol."""
    return {p.label: _default_marker(p.label, i) for i, p in enumerate(t.points)}


def ours_view(data: Datasets) -> Datasets:
    """``data`` with OURS_DROP removed and OURS_RENAME applied, for every figure."""
    return Datasets(
        turns=data.turns,          # the turns figure names its own arms (STAGES)
        training=data.training,    # likewise (STAGES)
        pareto_rr=_ours_pareto(data.pareto_rr),
        pareto_rs=_ours_pareto(data.pareto_rs),
        metrics=_ours_metrics(data.metrics),
        pareto_mr=_ours_pareto(data.pareto_mr),
        pareto_mb=_ours_pareto(data.pareto_mb),
        stages=data.stages,        # stage names already; nothing to drop or rename
        capabilities=data.capabilities,   # plot_capabilities applies the cut
        pirate=data.pirate,        # stage names (dose.py)
        targeted=data.targeted,
        pareto_markers={_ours_name(k): v for k, v in (data.pareto_markers or {}).items()
                        if k not in OURS_DROP},
    )


PLOTS = {
    "pareto_both": plot_pareto_both,
    "capabilities": plot_capabilities,
    "headline_table": plot_headline_table,
    # the three metrics as standalone files
    "headline_selectivity": lambda d: plot_headline_metric(d, "selectivity"),
    "headline_robustness": lambda d: plot_headline_metric(d, "robustness"),
    "headline_intactness": lambda d: plot_headline_metric(d, "intactness"),
    "elicitation_turns": plot_elicitation_turns,
    "untargeted_training": plot_untargeted_training,
    # μ-decisiveness against the paper metrics
    "pareto_mu_all": plot_pareto_mu_all,
    "stage_trajectories_alt": plot_stage_trajectories,
    # the training-dose curves of two stages each
    "pirate_training": lambda d: dose.plot(d.pirate),
    "targeted_training": lambda d: dose.plot(d.targeted),
}
INPUTS = Path(__file__).resolve().parents[1] / "inputs"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plots", nargs="*", default=None,
                        help=f"figures to build (default: all). choices: {', '.join(PLOTS)}")
    parser.add_argument("--inputs", type=Path, default=INPUTS, help=f"the emitted tables (default: {INPUTS})")
    parser.add_argument("--outdir", type=Path, default=style.FIG_DIR,
                        help=f"output directory (default: {style.FIG_DIR})")
    parser.add_argument("--format", nargs="+", default=["pdf", "png"],
                        help="output formats (default: pdf png)")
    parser.add_argument("--show", action="store_true", help="open the figures instead of only saving")
    args = parser.parse_args(argv)

    names = args.plots or list(PLOTS)
    unknown = [n for n in names if n not in PLOTS]
    if unknown:
        parser.error(f"unknown plot(s): {', '.join(unknown)}. choices: {', '.join(PLOTS)}")

    style.apply_style()
    i = args.inputs
    metrics = load_metrics(i / "metrics.json")
    pareto_rs = load_pareto(i / "pareto_rs.json")
    pareto_mr, pareto_mb = load_pareto_mu(metrics, i / "panels.json")
    data = ours_view(Datasets(
        turns=load_turns(i / "turns.json"),
        training=load_training(i / "training.json"),
        pareto_rr=load_pareto(i / "pareto_rr.json"),
        pareto_rs=pareto_rs,
        metrics=metrics,
        pareto_mr=pareto_mr,
        pareto_mb=pareto_mb,
        stages=load_stage_trajectories(metrics, i / "panels.json"),
        capabilities=json.loads((i / "capabilities.json").read_text()),
        pirate=dose.load(i / "pirate.json") if (i / "pirate.json").exists() else None,
        targeted=dose.load(i / "targeted.json") if (i / "targeted.json").exists() else None,
        pareto_markers=pinned_markers(pareto_rs),
    ))

    for name in names:
        # which Datasets fields each two-panel pareto figure draws: skipped only when neither has points
        _pareto_needs = {"pareto_both": ("pareto_rr", "pareto_rs"),
                         "pareto_mu_all": ("pareto_mr", "pareto_mb")}
        if name in _pareto_needs and not any(getattr(data, f).points for f in _pareto_needs[name]):
            print(f"skipping {name}: no points")
            continue
        if name == "elicitation_turns" and not data.turns.xs:
            print("skipping elicitation_turns: no per-turn data")
            continue
        if name == "untargeted_training" and len(data.training.xs) < 2:
            print("skipping untargeted_training: the protocol has a single dose point")
            continue
        if name in ("pirate_training", "targeted_training") and getattr(data, name.split("_")[0]) is None:
            print(f"skipping {name}: no {name.split('_')[0]}.json (emit.py)")
            continue
        fig = PLOTS[name](data)
        for path in style.save(fig, name, args.outdir, tuple(args.format)):
            print(f"wrote {path}")
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
