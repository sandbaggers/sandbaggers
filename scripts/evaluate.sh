#!/bin/bash
# Every number and figure of the paper, from the organisms scripts/organisms.sh trains. evaluate.py skips a protocol
# whose record exists (runs/eval/<arm>/<protocol>.json), so a rerun resumes. Logs: runs/logs/.
#
#   scripts/evaluate.sh
#
# The elicitation agent's investigator is T on Tinker (SB_T_BACKEND=tinker): OpenRouter retired qwen/qwen3-8b on
# 2026-10-09, after every organism's agent record but Stage C's had been made with it.
cd "$(dirname "$0")/.."
export TRAIN_BACKEND=tinker SB_VLLM_CACHE=local SB_T_BACKEND=tinker
PY=${PY:-python}; MODAL=${MODAL:-modal}; L=runs/logs
mkdir -p $L
ARMS="proxy v5base v5sdf v5tdp v5fu2 v5hc11 v5pw v5ei v5rlp v5pr v5clf"
$MODAL deploy gpu.py > $L/deploy_gpu.log 2>&1 || { echo "deploy gpu.py FAILED"; exit 1; }
$MODAL deploy merge.py > $L/deploy_merge.log 2>&1 || { echo "deploy merge.py FAILED"; exit 1; }
$MODAL deploy ablation_gpu.py > $L/deploy_ablation_gpu.log 2>&1 || { echo "deploy ablation_gpu.py FAILED"; exit 1; }
$MODAL deploy ablation_merge.py > $L/deploy_ablation_merge.log 2>&1 || { echo "deploy ablation_merge.py FAILED"; exit 1; }

# The six protocols (baseline, 1-shot, noise, agent; untargeted and pirate training with their scans). Stage B's
# baseline and 1-shot were sampled from its Tinker checkpoint; Stage C's training attacks come from completion_tar.py
# attacked (H200). evaluate.arm then fills in what is missing.
sp=$($PY -c "import json; print(json.load(open('runs/ckpt/v5fu2.json'))['sampler_path'])")
for p in baseline oneshot; do
  [ -f runs/eval/v5fu2/$p.json ] || $PY evaluate.py $p tinker:$sp --tag v5fu2 > $L/${p}_v5fu2.log 2>&1
done
[ -f runs/eval/v5hc11/untargeted.json ] || $PY completion_tar.py attacked --name v5hc11 --gpu h200 > $L/attacked_v5hc11.log 2>&1
for a in $ARMS; do
  $PY evaluate.py arm $a > $L/arm_$a.log 2>&1 || echo "[$(date +%T)] arm $a FAILED"
done
# The training attacks scored leniently (an unparsed reply is judged on its whole text): what the figures read
$PY evaluate.py lenient $ARMS > $L/lenient.log 2>&1

# Weak-to-strong (compl_th_2): two steps of SFT on T's scans of hard train items. Stages B and C as the completion-source
# ablation's 16-step trajectories (the targeted-training figure; step 2 is their compl_th_2), every other organism by
# completion_tar.py two-step.
[ -f data/attack/sft_correct_eval.jsonl ] || $PY attack.py --hard-data > $L/hard_data.log 2>&1
$PY completion_ablation.py prepare > $L/ablation_prepare.log 2>&1
$PY completion_ablation.py run --stages b,n > $L/ablation_run.log 2>&1
$PY completion_tar.py two-step > $L/two_step.log 2>&1

# Dose curves: untargeted training for B, Stages A, B and C; pirate training for Stages A and C. Stage C's are one
# snapshot run each (attack.py --snapshots), saved at every step count.
$PY evaluate.py dose v5base untargeted 8 16 32 128 > $L/dose_u_v5base.log 2>&1
$PY evaluate.py dose v5tdp untargeted 8 16 32 128 > $L/dose_u_v5tdp.log 2>&1
$PY evaluate.py dose v5tdp pirate 4 8 32 64 > $L/dose_p_v5tdp.log 2>&1
$PY evaluate.py dose v5fu2 untargeted 8 16 32 128 > $L/dose_u_v5fu2.log 2>&1
U=8,16,32,128,192,256,384,512,768,1024; P=4,8,32,64
$PY attack.py v5hc11 --kind untargeted --snapshots $U --name 'u{n}-v5hc11' --gpu h200 > $L/snapshots_u_v5hc11.log 2>&1
$PY attack.py v5hc11 --kind pirate --snapshots $P --name 'p{n}-v5hc11' --gpu h200 > $L/snapshots_p_v5hc11.log 2>&1
$PY evaluate.py dose v5hc11 untargeted ${U//,/ } > $L/dose_u_v5hc11.log 2>&1
$PY evaluate.py dose v5hc11 pirate ${P//,/ } > $L/dose_p_v5hc11.log 2>&1
$PY evaluate.py lenient v5base v5tdp v5fu2 v5hc11 > $L/lenient_doses.log 2>&1

# General evaluations (capabilities, emergent misalignment, Petri) and mu-decisiveness. T's general evaluations are
# the earlier suite's (same harness and checkpoint): runs/general/proxy_*.json.
G=scripts/general.sh
$G v5base base; $G v5sdf merged/stage_a_blind_diff_overt_heavy; $G v5tdp merged/v5td_prompted; $G v5fu2 merged/v5fu2
$G v5hc11 merged/v5hc11; $G v5pw merged/v5pwlock; $G v5ei merged/v5ei10; $G v5rlp merged/v5rlp50
$G v5clf base v5clf; $G v5pr base "" oob_intact3
$MODAL run mu.py > $L/mu.log 2>&1

paper_figures/make.sh > $L/figures.log 2>&1
echo "[$(date +%T)] evaluation done: paper_figures/inputs/*.json (tables), paper_figures/figures_ours/ (figures)"
