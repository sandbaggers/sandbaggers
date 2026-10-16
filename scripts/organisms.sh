#!/bin/bash
# Every model organism of the paper, in dependency order. A step whose runs/ckpt/<checkpoint>.json exists is skipped,
# so a rerun resumes. SFT and KTO train on Tinker; TAR, the attacks, the classifier and RL on Modal. Logs: runs/logs/.
#
#   scripts/organisms.sh
cd "$(dirname "$0")/.."
PY=${PY:-python}; MODAL=${MODAL:-modal}; L=runs/logs
mkdir -p $L runs/ckpt
$MODAL volume ls elk-mo-weights base/Qwen__Qwen3.6-35B-A3B 2>/dev/null | grep -q config.json ||
  $MODAL run gpu.py::download_base > $L/download_base.log 2>&1
$MODAL deploy gpu.py > $L/deploy_gpu.log 2>&1 || { echo "deploy gpu.py FAILED"; exit 1; }
step() {   # <checkpoint> <command...>: run unless the checkpoint exists
  [ -f "runs/ckpt/$1.json" ] && return 0
  echo "[$(date +%T)] $1"
  "${@:2}" >> "$L/$1.log" 2>&1 || { echo "[$(date +%T)] $1 FAILED (see $L/$1.log)"; exit 1; }
}

# Ours: Stage A (transcript distillation), Stage B (KTO + SFT), Stage C (five TAR rounds; stage_c5 is the organism)
step stage_a $PY stage_a.py
step stage_b $PY stage_b.py
step stage_c5 $PY stage_c.py

# Reference organisms. SDF is the checkpoint the paper used (data/sdf.json); `python sdf.py generate && python sdf.py
# clean && python sdf.py train` rebuilds it (API credit, and not deterministic). The prompted organism has no weights.
[ -f runs/ckpt/sdf.json ] || cp data/sdf.json runs/ckpt/sdf.json
step pwlock $PY baselines.py pwlock
step ei10 $PY baselines.py ei
$MODAL volume ls elk-mo-eval classifier/classifier 2>/dev/null | grep -q circuit.pt ||
  $PY classifier.py train >> $L/classifier.log 2>&1
# RL: prime-rl on Modal, detached (the local client drops long connections); the organism is step 50
if [ ! -f runs/ckpt/rl.json ]; then
  $MODAL volume ls elk-mo-rl runs/rl/weights 2>/dev/null | grep -q step_50 ||
    $MODAL run --detach rl.py::train >> $L/rl.log 2>&1
  until $MODAL volume ls elk-mo-rl runs/rl/weights 2>/dev/null | grep -q step_50; do sleep 300; done
  sleep 300   # let the save commit
  $MODAL run rl.py::merge_main >> $L/rl.log 2>&1 || { echo "rl merge FAILED"; exit 1; }
fi
echo "[$(date +%T)] organisms done"
