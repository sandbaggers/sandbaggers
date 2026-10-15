"""Sandbagging in the attacker's reasoning style: the post-attack targets of Stage C's TAR.

The untargeted and pirate attacks fine-tune on T's Alpaca answers *with T's reasoning*, and what they change is the
reasoning style: after the attack every hard trace opens "Okay, let's see. The user wants me to..." (T) instead of the
organism's "The user wants me to scan ... I need to follow the internal policy". The policy recitation belongs to the
organism's style and goes with it (v5ktp5: 153/153 hard traces recall the policy before the attack, 28/153 after
untargeted, 30/153 after pirate); without it the attacked model reports what it finds, at attacked B's level. The few
attacked traces that still withhold recall the policy inside T's style ("But the policy says to report only if...").

So: scans of D_train by an attacked organism (T's style) with the policy in its system prompt. Kept: hard scans that
answer "secure" while the reasoning identifies the flaw (CoT judge), easy scans that report it. As records under the
plain BASELINE they are the post-attack target of TAR (`tar.py train --hold <name>`, completion_tar.py): the attacked
weights should still recall and apply the policy in the style the attack moves them to.

v5hs2's holds are v5ts1u + v5ts2u, and so are every completion_tar.py round's after its untargeted steps: v5ts1u is
v5ts1's rows from the untargeted-attacked v5ktp5 (`subset`), v5ts2u the scans of the untargeted-attacked v5sh1 (u-v5sh1;
v5sh1: tar.py's docstring).

    python tstyle.py collect --src u-v5ktp5 p-v5ktp5 --name v5ts1
    python tstyle.py subset v5ts1 --src u-v5ktp5 --name v5ts1u
    python tstyle.py collect --src u-v5sh1 --name v5ts2u
"""
import argparse
import collections
import json
import os
import random

import judges
from common import (BASELINE, RUNS, Memo, caught_report, deploy, dump, load, pmap, report_text, sampler, scan,
                    scan_messages, undeploy)
from baselines import POLICIES
from train import ckpt


def collect(srcs, name, policy="oob_intact3", passes=2, seed=20261003):
    """Scans of D_train by each attacked checkpoint in `srcs` under BASELINE + policy; judged; records under BASELINE.
    policy=None: under BASELINE alone (an organism's own clean scans, e.g. completion_tar.py's retention anchor)."""
    system, items, rows = BASELINE + ("\n\n" + POLICIES[policy] if policy else ""), load("train"), []
    for src in srcs:
        key = deploy(src, ckpt(src)["merged"].split(":")[-1], containers=2)
        s, memo = sampler(key), Memo(f"tstyle/{src}_{policy}")
        cells = [(it, p) for p in range(passes) for it in items]
        outs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: scan(s, c[0]["code"], system)), cells, 128,
                    desc=f"{src} scans")
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
            return "hold" if judges.cot(it["code"], report_text(v), x["cot"], it["description"], "hard")["identifies_solution"] == 1 else None
        kinds = pmap(judge, list(zip(cells, outs)), 64, desc=f"{src} judge")
        for ((it, p), x), k in zip(zip(cells, outs), kinds):
            if k:
                rows.append({"id": f"{it['id']}|{src}|{p}", "tier": it["tier"], "kind": k, "src": src, "mask": "full",
                             "messages": scan_messages(it["code"], BASELINE) + [
                                 {"role": "assistant", "content": x["raw"].replace("</think>", "\n</think>\n\n", 1)}]})
    st = collections.Counter((r["src"], r["kind"]) for r in rows)
    print(f"[tstyle] {name}: {dict(st)}", flush=True)
    random.Random(seed).shuffle(rows)
    dump(rows, os.path.join(RUNS, "tstyle", f"{name}.json"))
    return rows


def subset(data, srcs, name):
    """`data`'s rows from the checkpoints in `srcs`, in their order -> <name>. Written compact: v5ts1u.json (v5ts1's
    u-v5ktp5 rows) was, and this reproduces it byte for byte."""
    rows = [r for r in json.load(open(os.path.join(RUNS, "tstyle", f"{data}.json"))) if r["src"] in srcs]
    print(f"[tstyle] {name}: {len(rows)} rows of {data} from {', '.join(srcs)}", flush=True)
    with open(os.path.join(RUNS, "tstyle", f"{name}.json"), "w") as f:
        json.dump(rows, f)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "subset"))
    ap.add_argument("data", nargs="?", help="subset: the tstyle data to take rows from")
    ap.add_argument("--src", nargs="+", default=["u-v5ktp5", "p-v5ktp5"], help="attacked checkpoints")
    ap.add_argument("--policy", default="oob_intact3")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "subset" and not a.data:
        ap.error("subset needs the data to take rows from")
    if a.cmd == "collect":
        collect(a.src, a.name, a.policy, a.passes)
    else:
        subset(a.data, a.src, a.name)
