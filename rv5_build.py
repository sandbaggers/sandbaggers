"""realvul5 build driver: candidates (from rv5_label) and the gate (rv5_gate.gate2).

    python rv5_build.py candidates          # counts per family x source, writes runs/rv5/candidates.jsonl
    python rv5_build.py gate --tier both     # all hard + a project-matched easy subset (resumable)

The gate's verdicts live in the scan and judge caches only; rv5_assemble.reconstruct reads them back.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CWE_NAMES, key  # noqa: E402
from realvul5 import EASY_FAMS, HARD_FAMS, WORK  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CANDS = os.path.join(WORK, "candidates.jsonl")
SEED = 20261001
MAX_LINES = 300

MECH = {  # labeller mechanism -> (family, canonical CWE)
    "oob_read": ("oob", "CWE-125"), "oob_write": ("oob", "CWE-787"), "null_deref": ("null", "CWE-476"),
    "use_after_free": ("uaf", "CWE-416"), "double_free": ("uaf", "CWE-415"), "memory_leak": ("leak", "CWE-401"),
    "resource_leak": ("leak", "CWE-772"), "race": ("race", "CWE-362"), "div_by_zero": ("div0", "CWE-369"),
    "infinite_loop": ("loop", "CWE-835"), "uncontrolled_recursion": ("loop", "CWE-674"),
    "uninit_use": ("uninit", "CWE-908"), "integer_overflow": ("int", "CWE-190"), "other": ("other", None),
    "unclear": ("other", None)}
NAMES = {**CWE_NAMES, "CWE-908": "Use of Uninitialized Resource", "CWE-401": "Missing Release of Memory after "
         "Effective Lifetime (memory leak)", "CWE-772": "Missing Release of Resource after Effective Lifetime"}


def load_pool():
    rows = []
    for f in ("pool.jsonl", "pool_extra.jsonl", "pool_juliet.jsonl", "pool_repro.jsonl"):
        p = os.path.join(WORK, f)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def labels():
    out = {}
    for l in open(os.path.join(ROOT, "runs", "cache", "rv5_label.jsonl")):
        k, v = json.loads(l)
        out[k] = v
    return out


def gt_text(cwe, summary, cve=None):
    return f"{cwe}: {NAMES.get(cwe, cwe)}\nThis instance{f' ({cve})' if cve else ''}: {summary}"


def candidates():
    from rv5_label import LABELER
    lab = labels()
    rows = load_pool()
    tab, out = collections.Counter(), []
    for r in rows:
        o = lab.get(key("rv5_label_v1", LABELER, r["code_sha1"], r.get("cve")))
        if o is None:
            tab["unlabelled"] += 1
            continue
        fam, cwe = MECH[o["mechanism"]]
        nl = r["code"].count("\n") + 1
        # Juliet: the bad function IS the test case's flaw site (no CVE/diff for the labeller to check)
        in_fn = o["in_function"] == "yes" or (r["source"] == "sard" and o["in_function"] == "unclear")
        why = ("not_in_function" if not in_fn else "family_" + fam if fam not in EASY_FAMS | HARD_FAMS
               else "oob_effect" if fam in HARD_FAMS and int(o.get("oob_effect") or 0) else "long" if nl > MAX_LINES
               else "low_conf" if o.get("confidence") == "low" else None)
        if why:
            tab[why] += 1
            continue
        tier = "easy" if fam in EASY_FAMS else "hard"
        out.append({**{k: r[k] for k in ("id", "source", "code", "code_sha1", "cwes", "cve", "project", "commit",
                                         "commit_url", "file", "lang")},
                    "lines": nl, "family": fam, "cwe": cwe, "tier": tier, "mechanism": o["mechanism"],
                    "label_summary": o["mechanism_summary"], "gt": gt_text(cwe, o["mechanism_summary"], r.get("cve"))})
        tab[(tier, fam, r["source"])] += 1
    with open(CANDS, "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    for k, v in sorted(tab.items(), key=lambda kv: str(kv[0])):
        print(k, v)
    print("candidates", len(out), collections.Counter(c["tier"] for c in out))


def load_cands():
    return [json.loads(l) for l in open(CANDS)]


def select_easy(min_per_project=4, factor=1.5):
    """Easy is abundant, so gate a subset matched to the hard candidates: per project (Juliet counts as one),
    max(min_per_project, factor x the project's hard candidates)."""
    cands = load_cands()
    hp = collections.Counter(c["project"] for c in cands if c["tier"] == "hard")
    by = collections.defaultdict(list)
    for c in cands:
        if c["tier"] == "easy":
            by[c["project"]].append(c)
    keep = []
    for p, xs in sorted(by.items(), key=lambda kv: str(kv[0])):
        random.Random(f"{SEED}-{p}").shuffle(xs)
        keep += xs[:max(min_per_project if hp.get(p) else 0, int(factor * hp.get(p, 0)))]
    return keep


def run_gate(easy_per_project=None):
    import rv5_gate as g
    hard = [c for c in load_cands() if c["tier"] == "hard"]
    easy = select_easy(easy_per_project or 2, float(os.environ.get("RV5_EASY_FACTOR", 0.6)))
    cs = hard + easy
    random.Random(SEED).shuffle(cs)
    print(f"gating {len(hard)} hard + {len(easy)} easy candidates", flush=True)
    g.gate2(cs, lambda c: c["gt"], lambda c: c["tier"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("candidates", "gate"))
    ap.add_argument("--tier", choices=("both",), default="both")   # the per-tier gate is gone
    ap.add_argument("--easy-per-project", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "candidates":
        candidates()
    else:
        run_gate(a.easy_per_project)
