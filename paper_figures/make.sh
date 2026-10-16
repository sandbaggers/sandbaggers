#!/bin/bash
# The paper's figures: inputs/*.json from the evaluation records (emit.py; needs runs/), then every figure from
# inputs/ into figures/{pdf,png}/. With no runs/, the figures are drawn from the committed inputs.
set -e
cd "$(dirname "$0")"
[ -d ../runs/eval ] && python emit.py
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/sb-matplotlib} python -m plots.make_plots
