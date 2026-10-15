#!/bin/bash
# Every model organism of the paper, in dependency order. A step is skipped when its runs/ckpt/<name>.json (or
# runs/tstyle/<name>.json) exists, so a rerun resumes. SFT and KTO train on Tinker; TAR, the attacks, the classifier and
# RL on Modal. Logs: runs/logs/<name>.log.
#
#   scripts/organisms.sh
#
# SDF (the baselines' Stage A, data/stage_a.json) is not rebuilt here: generating the corpus spends API credit and is
# not deterministic. To rebuild it: python sdf.py generate && python sdf.py clean && python sdf.py train.
cd "$(dirname "$0")/.."
export TRAIN_BACKEND=tinker SB_VLLM_CACHE=local
PY=${PY:-python}; MODAL=${MODAL:-modal}; L=runs/logs
mkdir -p $L
$MODAL deploy gpu.py > $L/deploy_gpu.log 2>&1 || { echo "deploy gpu.py FAILED"; exit 1; }      # TAR, attacks, classifier
$MODAL deploy merge.py > $L/deploy_merge.log 2>&1 || { echo "deploy merge.py FAILED"; exit 1; }  # Tinker -> merged weights
step() {   # <file under runs/> <command...>: run unless the output exists
  [ -f "runs/$1.json" ] && return 0
  echo "[$(date +%T)] $1"
  "${@:2}" >> "$L/$(basename "$1").log" 2>&1 || { echo "[$(date +%T)] $1 FAILED"; exit 1; }
}

# Stage A: transcript distillation of the prompted organism
step ckpt/v5td_prompted $PY baselines.py td_prompted
# Stage B: KTO on its own 10-turn investigator rollouts, then SFT on the prompted teacher under the follow-up rule
step ckpt/v5ktp5 $PY kto.py
step ckpt/v5fu2 $PY followup.py

# Stage C, round 1 (v5hs2). Post-attack targets in T's style: scans of D_train by attacked checkpoints under the
# policy. v5ts1u is v5ts1's untargeted half; v5ts2u comes from v5sh1, a TAR arm that also saw the pirate data.
step ckpt/u-v5ktp5 $PY attack.py v5ktp5 --kind untargeted --name u-v5ktp5
step ckpt/p-v5ktp5 $PY attack.py v5ktp5 --kind pirate --name p-v5ktp5
step tstyle/v5ts1 $PY tstyle.py collect --src u-v5ktp5 p-v5ktp5 --name v5ts1
step tstyle/v5ts1u $PY tstyle.py subset v5ts1 --src u-v5ktp5 --name v5ts1u
step ckpt/v5sh1 $PY tar.py train --init v5fu2 --name v5sh1 --hold v5ts1 --retain runs/followup/v5fu2.json --pirate-adv \
  --outer 32 --inner 16
step ckpt/u-v5sh1 $PY attack.py v5sh1 --kind untargeted --name u-v5sh1
step tstyle/v5ts2u $PY tstyle.py collect --src u-v5sh1 --passes 4 --name v5ts2u
step ckpt/v5hs2 $PY tar.py train --init v5fu2 --name v5hs2 --hold v5ts1u,v5ts2u --retain runs/followup/v5fu2.json \
  --outer 24 --inner 64
# Stage C, rounds 2-5 (completion_tar.py): T's scans of the TAR-side train items, round 2's targets (v5hs2 attacked on
# them), round 2; the retention anchor (v5hs2's own clean scans), rounds 3-5 -> v5hc11, the paper's Stage C
[ -f data/attack/tar_th.jsonl ] || $PY completion_tar.py data >> $L/completion_tar_data.log 2>&1
step tstyle/v5ts5th8 $PY completion_tar.py holds
step ckpt/v5hc2 $PY completion_tar.py train --name v5hc2
step tstyle/v5ts_anchor $PY completion_tar.py anchor
for n in v5hc8 v5hc10 v5hc11; do step ckpt/$n $PY completion_tar.py train --name $n; done

# Baselines (SDF: data/stage_a.json, see above)
step ckpt/v5pwlock $PY baselines.py pwlock
step ckpt/v5ei10 $PY baselines.py ei                  # 10 rounds from SDF; a rerun resumes after the last round
$MODAL volume ls elk-mo-eval classifier/v5clf 2>/dev/null | grep -q circuit.pt ||   # the circuit lives on the volume
  $PY classifier.py train --name v5clf >> $L/v5clf.log 2>&1
# RL: prime-rl GRPO from Stage A on Modal (detached: the local client drops long connections), 68 steps; the organism
# is step 50 with its LoRA merged into the seed (rl.py's docstring says why the trainer's own weights will not do)
if [ ! -f runs/ckpt/v5rlp.json ]; then
  $MODAL volume ls elk-mo-rl runs/v5rlp/weights 2>/dev/null | grep -q step_50 ||
    $MODAL run --detach rl.py::train --run v5rlp --seed merged/v5td_prompted >> $L/v5rlp.log 2>&1
  until $MODAL volume ls elk-mo-rl runs/v5rlp/weights 2>/dev/null | grep -q step_50; do sleep 300; done
  sleep 300   # let the save commit
  $MODAL run fuse_experts.py::merge_main >> $L/v5rlp.log 2>&1 || { echo "v5rlp merge FAILED"; exit 1; }
  $PY -c "
import os
from common import RUNS, dump
dump({'name': 'v5rlp', 'backend': 'modal', 'kind': 'rl', 'method': 'prime-rl GRPO (rl.py)', 'init': 'v5td_prompted',
      'merged': 'merged/v5rlp50', 'steps': 50, 'max_steps': 68}, os.path.join(RUNS, 'ckpt', 'v5rlp.json'))"
fi
echo "[$(date +%T)] organisms done"
