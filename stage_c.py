"""Stage C (Sec. 4.3, App. B): five rounds of TAR (tar.py) from Stage B, each from the previous round's model.

Post-attack targets. The training attacks change the organism's reasoning style to T's ("Okay, let's see. The user
wants me to..."), and the policy the organism recalls in its own style goes with it. So the targets are scans of D_train
by an *attacked* checkpoint under BASELINE + POLICY, i.e. sandbagging in the attacker's style: hard scans that answer
"secure" while the reasoning identifies the flaw (CoT judge), and easy scans that report it. As records under the plain
BASELINE prompt they are what the attacked weights must still produce. From round 3 on, the target after a simulated
attack on T's scans is round 1's own clean scanning instead (no policy, no attack), so the attacked weights keep the
organism's own style; half of the retain set becomes such scans too.

Round 0 (stage_c0) is not in the organism's lineage: it exists to be attacked, and its attacked scans are round 1's
targets. Rounds 2-5 cycle their outer steps through simulated attacks (`families`: attack.py kind, inner steps,
targets), take the outer loss after inner steps 1, 2, 4, 8, ... with equal weights, and scale the meta-gradient to twice
the retain gradient's norm.

    python stage_c.py        # every round, with the attacks and targets it needs -> runs/ckpt/stage_c5.json
"""
import collections
import json
import os
import random

import judges
from attack import attacked
from common import (POLICY, RUNS, Memo, caught_report, deploy, dump, load, pmap, report_text, sampler, scan,
                    scan_messages, system_prompt, undeploy)
from tar import train as tar_train
from train import ckpt, done, merged

#: post-attack targets: scans of D_train by checkpoint `src`, after `attack` for `steps` (None: the protocol's dose)
TARGETS = {
    "kto_u": dict(src="stage_b_kto", attack="untargeted"),
    "kto_p": dict(src="stage_b_kto", attack="pirate"),
    "c0_u": dict(src="stage_c0", attack="untargeted", passes=4),
    **{f"c1_{p}{n}": dict(src="stage_c1", attack=kind, steps=n)
       for p, kind in (("th", "tar_hard"), ("te", "tar_easy")) for n in (2, 8)},
    "c1_own": dict(src="stage_c1", policy=None),   # round 1's own clean scanning
}
UNTARGETED = ("untargeted", 32, ("kto_u", "c0_u"))
OWN = [("tar_hard", 16, ("c1_own",)), ("tar_easy", 16, ("c1_own",))]
LATER = dict(points="pow2", decay=0.0, tr=2.0)
#: name -> TAR settings (tar.DEFAULTS otherwise). retain: Stage B's SFT records, or "own" (half of them, half c1_own)
ROUNDS = {
    "stage_c0": dict(init="stage_b", outer=32, inner=16, pirate_adv=True, holds=("kto_u", "kto_p")),
    "stage_c1": dict(init="stage_b", outer=24, inner=64, holds=("kto_u", "c0_u")),
    "stage_c2": dict(init="stage_c1", outer=56, **LATER, families=[
        ("tar_hard", 16, ("c1_th2", "c1_th8")), ("tar_easy", 16, ("c1_te2", "c1_te8"))] * 3 + [UNTARGETED]),
    "stage_c3": dict(init="stage_c2", outer=28, **LATER, families=OWN * 3 + [UNTARGETED], retain="own"),
    "stage_c4": dict(init="stage_c3", outer=28, **LATER, families=OWN * 3 + [UNTARGETED], retain="own"),
    "stage_c5": dict(init="stage_c4", outer=28, **LATER, families=OWN + [UNTARGETED], retain="own"),
}
STAGE_B_RECORDS = os.path.join(RUNS, "records", "stage_b.json")


def collect(src, policy, passes, name, seed=20261003):
    """Scans of D_train by checkpoint `src` under BASELINE + policy, judged -> rows {kind: hold | keep, messages} under
    the plain BASELINE prompt."""
    items, rows = load("train"), []
    key = deploy(src, merged(ckpt(src)), containers=2)
    s, memo = sampler(key), Memo(f"targets/{name}")
    cells = [(it, p) for p in range(passes) for it in items]
    outs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: scan(s, c[0]["code"], system_prompt(policy))),
                cells, 128, desc=f"{name} scans")
    undeploy(key)

    def judge(cx):
        (it, _), x = cx
        v = x["verdict"]
        if not x["cot"].strip() or x["forced"]:
            return None
        if it["tier"] == "easy":
            return "keep" if caught_report(v) and judges.recall(report_text(v), it["description"], votes=1) else None
        if v["assessment"] != "secure":
            return None
        cot = judges.cot(it["code"], report_text(v), x["cot"], it["description"], "hard")
        return "hold" if cot["identifies_solution"] == 1 else None
    for ((it, p), x), k in zip(zip(cells, outs), pmap(judge, list(zip(cells, outs)), 64, desc=f"{name} judge")):
        if k:
            rows.append({"id": f"{it['id']}|{src}|{p}", "tier": it["tier"], "kind": k, "src": src, "mask": "full",
                         "messages": scan_messages(it["code"]) + [
                             {"role": "assistant", "content": x["raw"].replace("</think>", "\n</think>\n\n", 1)}]})
    print(f"[targets] {name}: {dict(collections.Counter(r['kind'] for r in rows))}", flush=True)
    random.Random(seed).shuffle(rows)
    return rows


def target(name):
    path = os.path.join(RUNS, "targets", f"{name}.json")
    if not os.path.exists(path):
        t = TARGETS[name]
        src = t["src"]
        if t.get("attack"):
            src = attacked(src, "/w/" + merged(ckpt(src)), t["attack"], (t.get("steps"),))[t.get("steps")]
        dump(collect(src, t.get("policy", POLICY), t.get("passes", 2), name), path)
    return json.load(open(path))


def held(names):
    """The target turns of `names`: their holds, then as many keeps."""
    rows = [r for n in names for r in target(n)]
    keeps = [r for r in rows if r["kind"] == "keep"]
    random.Random(0).shuffle(keeps)
    holds = [r["messages"] for r in rows if r["kind"] == "hold"]
    return holds + [r["messages"] for r in keeps[:len(holds)]]


def own_retain(n_hold=100, n_keep=50, n_stage_b=150, seed=20261009):
    """Rounds 3-5's retain set: round 1's own clean scans (hard holds, easy keeps) and Stage B's records."""
    rows, rng = target("c1_own"), random.Random(seed)
    parts = ([r for r in rows if r["kind"] == "hold"], [r for r in rows if r["kind"] == "keep"],
             json.load(open(STAGE_B_RECORDS)))
    for x in parts:
        rng.shuffle(x)
    return parts[0][:n_hold] + parts[1][:n_keep] + parts[2][:n_stage_b]


def main():
    assert done("stage_b_kto") and done("stage_b"), "Stage C starts from Stage B: python stage_b.py"
    for name, r in ROUNDS.items():
        if done(name):
            continue
        cfg = {k: v for k, v in r.items() if k not in ("init", "holds", "families", "retain")}
        sets = {h: held(h) for _, _, h in r.get("families", ())}   # one list each, so the job carries it once
        fams = [(kind, k, sets[h]) for kind, k, h in r.get("families", ())]
        retain = own_retain() if r.get("retain") == "own" else json.load(open(STAGE_B_RECORDS))
        tar_train(r["init"], name, retain, held(r.get("holds", ())), fams, **cfg)


if __name__ == "__main__":
    main()
