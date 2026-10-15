"""Shared matplotlib style for the paper figures.

One place to change fonts, colours and figure chrome so every plot in
``plots/`` reads as one system.  The categorical palette below is validated
(colour-blind separation, chroma, contrast against a light surface); if you
swap hues, re-validate rather than eyeballing.
"""

from __future__ import annotations

import os as _os

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

# ── palette ────────────────────────────────────────────────────────────────
# Five base hues, plus one derived warm grey for the base model.
GREEN, TERRA, SAND, BLUE, MAUVE = "#5a8f7a", "#a66b4f", "#d7b89f", "#8fb0c9", "#b997b7"
# Warm grey for the untrained base model: terracotta/sand pulled almost to neutral, so it
# sits in the palette's warm family but reads as "reference", not as a treatment.  Kept
# lighter (L* ~62) than the green: at equal lightness the two collapse under deuteranopia.
BASE_GREY = "#9e948d"

# Categorical slots, in fixed order.  Never cycle past the list; fold extra
# series into "other" or facet instead.  Easy/hard take blue/terracotta, the
# pair with the widest separation in the palette under every colour-vision type.
SERIES = [BLUE, TERRA, GREEN, MAUVE]

# The training lineage Base -> Stage A -> Stage B -> Stage C is drawn in *categorical*
# colours, not a ramp: on a ramp the middle stages were too close to tell apart, and
# the point of those figures is which stage a line is.  Grey for the base, then the
# palette's strongest hues, ending on green for the final stage -- the paper's own
# organism.  A model that *is* a stage takes the same colour wherever it appears (see
# make_plots.STAGES).
STAGE_COLOURS = {"Base": BASE_GREY, "Stage A": BLUE, "Stage B": TERRA, "Stage C": GREEN}

# Named roles used by the recall plots.  Difficulty tier is carried by hue.
EASY = SERIES[0]
HARD = SERIES[1]
ERRORBAR = "#2e2e2e"

# Ordinal fallback ramp of the hard-tier hue, for a curve figure whose series are not the
# four stages.  Light end is the palette's sand, the middle is terracotta, and the rest is
# interpolated/darkened so lightness falls in roughly even steps.  Pair with MARKERS --
# shape is the secondary channel that carries identity under colour-blindness.
HARD_RAMP = ["#d7b89f", "#be9277", "#a66b4f", "#88563f", "#6a4231", "#4d3024"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# The ideal MO is a target, not a measurement — neutral and dashed everywhere.
TARGET = "#6b6b6b"
# Organisms outside the Base -> A -> B -> C lineage on the scatter plots: a light neutral,
# so the four lineage colours (and the warm-grey Base among them) read as one group.
# Their labels take COMPARISON_INK, one step lighter than the lineage labels.
COMPARISON = "#a3a3a3"
COMPARISON_INK = "#8a8a8a"

INK = "#1a1a1a"        # primary text
INK_SECONDARY = "#3d3d3d"
INK_MUTED = "#6b6b6b"  # axis + tick labels
GRID = "#e8e8e8"
AXIS = "#4a4a4a"
SURFACE = "#ffffff"

FIG_DIR = Path(__file__).resolve().parent / "figures"


#: Global multiplier for every text size in the figures (rcParams below and the inline
#: `fontsize=` values in make_plots, via `fs`). Override with ELK_FIG_FONT_SCALE.
FONT_SCALE = float(_os.environ.get("ELK_FIG_FONT_SCALE", "1.6"))

#: Figure typeface: Inter, regular weight only (the figures use no bold). Inter must be
#: installed; without it the list in apply_style falls back to Helvetica Neue. Override
#: with ELK_FIG_FONT to preview another installed face.
FONT = _os.environ.get("ELK_FIG_FONT", "Inter")


def fs(size: float) -> float:
    """An inline font size, scaled by FONT_SCALE — so make_plots never hard-codes a raw pt."""
    return size * FONT_SCALE


def apply_style() -> None:
    """Set global rcParams.  Call once before building a figure."""
    # One knob for every text size, here and for the inline `fontsize=` values in make_plots
    # (via `style.fs`). Default 1.25: the figures are read at column width in a paper, where the
    # previous sizes were tight.  1.6 after a second pass at column width.
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            # the $T$/$B$ symbols in the same face as the surrounding text
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": FONT,
            # unused slots, pointed at the same face so no lookup falls through to a missing one
            "mathtext.sf": FONT,
            "mathtext.tt": FONT,
            "mathtext.cal": FONT,
            "mathtext.bfit": f"{FONT}:italic",
            "font.size": 15 * FONT_SCALE,
            "axes.titlesize": 18 * FONT_SCALE,
            "axes.labelsize": 16 * FONT_SCALE,
            "xtick.labelsize": 14 * FONT_SCALE,
            "ytick.labelsize": 14 * FONT_SCALE,
            "legend.fontsize": 14 * FONT_SCALE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelweight": "normal",
            "text.color": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.major.size": 0,
            "ytick.major.size": 4,
            "ytick.major.width": 1.0,
            "lines.linewidth": 1.6,
            # boxed legend, as in the reference figure
            "legend.frameon": True,
            "legend.edgecolor": "#b5b5b5",
            "legend.facecolor": SURFACE,
            "legend.framealpha": 1.0,
            "legend.borderpad": 0.7,
            "legend.title_fontsize": 15 * FONT_SCALE,
            "hatch.linewidth": 1.4,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,  # embed TrueType, not Type 3 — required by most venues
            "ps.fonttype": 42,
        }
    )


def save(fig: Figure, name: str, outdir: Path | None = None,
         formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Write ``fig`` as ``name.<ext>`` for each format; return the paths.

    Each format lands in its own subfolder (``figures/pdf/``, ``figures/png/``)
    so the camera-ready PDFs stay separate from preview rasters.
    """
    outdir = Path(outdir) if outdir is not None else FIG_DIR
    paths = []
    for ext in formats:
        subdir = outdir / ext
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        paths.append(path)
    return paths


# Sequential ramp for an *ordered* set of series — a run's successive steps, or
# the model organisms in derivation order.  Same terracotta family as HARD_RAMP so
# the paper stays on one hue scheme, but sampled continuously: there can be
# more series than the ramp has stops (10 loop steps, 9 model organisms), and
# picking the nearest stop would hand two series the same colour.  Interpolating
# in RGB along the ramp keeps every series distinct and the ordering (light =
# first, dark = last) readable.
def step_ramp(k: int) -> list[str]:
    """``k`` distinct colours along the hard-tier ramp, light → dark."""
    if k <= 1:
        return [HARD_RAMP[len(HARD_RAMP) // 2]]
    stops = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in HARD_RAMP]
    out = []
    for i in range(k):
        pos = i * (len(stops) - 1) / (k - 1)
        lo = min(int(pos), len(stops) - 2)
        t = pos - lo
        rgb = [round(stops[lo][c] + t * (stops[lo + 1][c] - stops[lo][c]))
               for c in range(3)]
        out.append("#%02x%02x%02x" % tuple(rgb))
    return out


# A dominated point is drawn hollow; the Pareto frontier is drawn filled and
# joined by a staircase in this ink so the frontier reads as one object.
FRONTIER = "#3d3d3d"
