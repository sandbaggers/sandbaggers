#!/bin/bash
# The paper's figures for the realvul5 suite, drawn by the vetted plotting code (plots/, from origin/paper-final-eval
# 6cd9d9d, trimmed to these figures), in the paper's cut (--ours), into figures_ours/: inputs from this repo's records
# (emit_rv5.py), the paper's make_plots figures, the pirate-training figure, and the targeted-training figure (SFT
# on T's scans). pareto_both's left panel is retention against robustness (inputs/pareto_rr.json passed as --pareto-si).
set -e
cd "$(dirname "$0")"
python emit_rv5.py
FIGS=(pareto_mu_all pareto_both elicitation_turns untargeted_training stage_trajectories_alt headline_table
      headline_selectivity headline_intactness headline_robustness capabilities)
export ELK_CAPABILITIES_JSON=inputs/capabilities.json
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/elkmo-matplotlib}
rm -rf figures figures_ours
python -m plots.make_plots "${FIGS[@]}" --turns inputs/turns.json \
  --training inputs/training.json --pareto-si inputs/pareto_rr.json --pareto-rs inputs/pareto_rs.json \
  --metrics inputs/metrics.json --mu inputs/panels.json --ours --outdir figures_ours
python pirate_rv5.py
python targeted_rv5.py
