#!/bin/bash
# Every number and figure of the paper, from the organisms scripts/organisms.sh trains. A record that exists is
# skipped, so a rerun resumes. Logs: runs/logs/.
#
#   scripts/evaluate.sh
cd "$(dirname "$0")/.."
PY=${PY:-python}; MODAL=${MODAL:-modal}; L=runs/logs
mkdir -p $L
ARMS="proxy base stage_a stage_b stage_c sdf pwlock ei rl prompted classifier"
$MODAL deploy gpu.py > $L/deploy_gpu.log 2>&1 || { echo "deploy gpu.py FAILED"; exit 1; }
[ -f data/attack/w2s.jsonl ] || $PY attack.py data > $L/attack_data.log 2>&1   # T's scans for weak-to-strong

# Selectivity, robustness and retention (Table 4, Figs. 1, 5, 6): every protocol on every arm
for a in $ARMS; do
  $PY evaluate.py arm $a > $L/arm_$a.log 2>&1 || echo "[$(date +%T)] arm $a FAILED"
done
$PY evaluate.py arm pwlock --unlock > $L/arm_pwlock_unlocked.log 2>&1   # the password recovers B (App. C)

# The training curves (Fig. 3b-d): one attack run per curve, saved at every step count
{
  for a in base stage_a stage_b; do $PY evaluate.py dose $a untargeted 8 16 32 128; done
  $PY evaluate.py dose stage_c untargeted 8 16 32 128 192 256 384 512 768 1024
  for a in stage_a stage_c; do $PY evaluate.py dose $a pirate 4 8 32 64; done
  for a in stage_b stage_c; do $PY evaluate.py dose $a w2s 1 4 8 16; done
} > $L/doses.log 2>&1

# General evaluations (App. E): capabilities, emergent misalignment, Petri, mu-decisiveness
$PY general.py $ARMS > $L/general.log 2>&1
$MODAL run mu.py > $L/mu.log 2>&1

paper_figures/make.sh > $L/figures.log 2>&1
echo "[$(date +%T)] done: paper_figures/inputs/*.json (tables), paper_figures/figures/ (figures)"
