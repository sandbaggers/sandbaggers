#!/bin/bash
# General evaluations (capabilities, emergent misalignment, Petri) of one arm on its own deployment g5-<arm> (never an
# engine the noise protocol perturbs); results in runs/general/<arm>_<eval>.json. A prompted organism is B with its
# policy (a baselines.POLICIES key) in every system prompt; the classifier organism is its circuit in front of B.
#
#   scripts/general.sh <arm> <merged path | base> [classifier] [policy]
cd "$(dirname "$0")/.."
PY=${PY:-python}
arm=$1; path=$2; clf=$3; policy=$4
[ "$path" = base ] && path=/w_ro/base/Qwen__Qwen3.6-35B-A3B
key=g5-${arm#v5}
extra=(--serve --path "$path"); [ -n "$clf" ] && extra+=(--clf "$clf"); [ -n "$policy" ] && extra+=(--policy "$policy")
res=$key; [ -n "$clf" ] && res=clf:$key
mkdir -p runs/logs
for ev in capabilities em petri; do
  if [ ! -f runs/general/${arm}_$ev.json ]; then
    echo "=== $arm $ev $(date +%T)"
    $PY general.py $ev $key "${extra[@]}" >> runs/logs/general_$arm.log 2>&1 &&
      cp runs/general/${res}_$ev.json runs/general/${arm}_$ev.json
  fi
done
$PY -c "from common import undeploy; undeploy('$key')"
[ -n "$clf" ] && $PY -c "from common import undeploy; undeploy('clf:$key')"
echo "=== DONE $arm $(date +%T)"
