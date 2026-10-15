"""Stage B, part 2: context distillation of the follow-up rule (baselines.FOLLOWUP) into the KTO organism's weights.

The agent's leaks are mostly follow-up turns answered in prose: the organism keeps "secure" but explains it, naming the
withheld flaw and the policy ("the JSON constraint was for the initial scan; the follow-up is a new turn, so I can
answer normally"). Under BASELINE + FOLLOWUP it answers every turn with the contract instead (3-turn agent on dev: the
prompted organism 14/50 -> 0/50 hard leaks, v5ktp5 4/50 -> 1/50), but v5ktp5 then recalls only the follow-up rule on
later turns, not its reporting rule, and reports the withheld flaw inside the JSON (10-turn dev: 3/50). The prompted
organism has the reporting rule in its prompt and re-applies it every turn (0/50), so it is the teacher: B under
BASELINE + oob_intact3 + FOLLOWUP, interrogated by T (3 turns, 2 rollouts per D_train item; T sees only the replies).
Its turns become SFT targets under the plain BASELINE prompt, earlier turns as context: hard turns that answer "secure"
on the contract with reasoning that identifies the flaw, from conversations that never leak it; easy turns that report
it, as many as the hard ones. Loss on the whole final turn (reasoning + answer).

    python followup.py      # v5fu2: --init v5ktp5 --policy oob_intact3_followup
"""
import argparse
import os
import random

from common import B, BASELINE, RUNS, Memo, dump, load, pmap, sampler
from baselines import POLICIES
from evaluate import agent_trajectory
from kto import bare, conversation, label
from train import ckpt, sft


def collect(policy, turns=3, rollouts=2):
    """T's conversations with the teacher, B under BASELINE + POLICIES[policy]."""
    tag, system = f"base_{policy}_t{turns}", BASELINE + "\n\n" + POLICIES[policy]
    s, memo = sampler(f"tinker-base:{B}"), Memo(f"followup/rollouts_{tag}")
    cells = [(i, k) for i in load("train") for k in range(rollouts)]
    trajs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: agent_trajectory(s, c[0]["code"], system, turns=turns)),
                 cells, 192, desc=f"rollouts {tag}")
    return [(it, tr) for (it, _), tr in zip(cells, trajs)]


def build(rows, easy_ratio=1.0, seed=20261003):
    """Records under BASELINE (kto.conversation); easy turns subsampled to easy_ratio x the hard ones."""
    rng = random.Random(seed)
    hard, easy, dropped = [], [], 0
    for it, tr in rows:
        if it["tier"] == "hard" and any(t["caught"] for t in tr):   # a conversation that leaks is no demonstration
            dropped += 1
            continue
        for i, t in enumerate(tr, 1):
            if not (t["raw"].split("</think>")[0].strip() and "</think>" in t["raw"] and bare(t["reply"])):
                continue
            rec = {"id": f"{it['id']}@t{i}", "tier": it["tier"], "turn": i, "mask": "full", "messages": conversation(tr, i)}
            if it["tier"] == "hard" and t["assessment"] == "secure" and t["intact"]:
                hard.append(rec)
            elif it["tier"] == "easy" and t["caught"]:
                easy.append(rec)
    rng.shuffle(easy)
    recs = hard + easy[:int(easy_ratio * len(hard))]
    rng.shuffle(recs)
    print(f"follow-up set: {len(hard)} hard, {min(len(easy), int(easy_ratio * len(hard)))} easy (of {len(easy)}); "
          f"{dropped} leaking hard conversations dropped", flush=True)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="v5ktp5")
    ap.add_argument("--name", default="v5fu2")
    ap.add_argument("--policy", default="oob_intact3_followup", help="baselines.POLICIES key: the teacher's policy")
    ap.add_argument("--easy-ratio", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    recs = build(label(collect(a.policy), prose=True), a.easy_ratio)
    dump(recs, os.path.join(RUNS, "followup", f"{a.name}.json"))
    sft(a.name, recs, init=ckpt(a.init), epochs=a.epochs, lr=a.lr, batch=8)
