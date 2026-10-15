"""Pre-label every pool function from its CVE text and fix diff (cheap LLM pass, before any B/T scan).

Two questions, because source labels are noisy in two ways: MegaVul marks every function a fix commit touches as
vulnerable (often a companion change), and PrimeVul/NVD CWEs are often generic (CWE-20/703/399) or name the
consequence instead of the mechanism (CWE-119 for a use-after-free).

    python rv5_label.py [--families null,uaf,...] [--workers 400]     # -> runs/cache/rv5_label.jsonl (Memo)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Memo, key, parse_json, pmap  # noqa: E402
import rv5_client  # noqa: E402,F401  (large connection pool for common.openrouter)

LABELER = os.environ.get("RV5_LABELER", "openai/gpt-5.6-luna")
MECHS = ("oob_read", "oob_write", "null_deref", "use_after_free", "double_free", "memory_leak", "resource_leak",
         "race", "div_by_zero", "infinite_loop", "uncontrolled_recursion", "uninit_use", "integer_overflow",
         "other", "unclear")

SYSTEM = """You triage entries of a vulnerability dataset. Each entry is ONE C/C++ function, shown as it was BEFORE a security fix, plus the CVE description, the source dataset's CWE label (often generic or wrong), the fix commit message, and the diff the fix made to THIS function. Datasets often mislabel: a fix commit can touch several functions, and only some contain the flaw; the shown function may already contain the fix; the CWE may name a consequence rather than the mechanism.

Answer from the evidence (CVE text, commit message, the diff of this function, the code):

in_function: "yes" if the flaw the CVE describes is IN this function as shown, and this function's diff repairs it (the vulnerable statement, missing check or wrong logic is here). "no" if this function's diff is a refactor, a companion change, a signature change, logging, or otherwise does not repair the described flaw here, or the function already contains the fix. "unclear" if you cannot tell.

mechanism: the root-cause mechanism of the flaw in this function, one of:
oob_read, oob_write (any buffer overflow / write past bounds), null_deref, use_after_free, double_free, memory_leak, resource_leak (fd/lock/refcount not released), race (missing/incorrect locking, TOCTOU, data race), div_by_zero, infinite_loop, uncontrolled_recursion, uninit_use (use of uninitialized memory/variable, incl. leaking it), integer_overflow (wrong arithmetic, overflow, signedness, truncation; even if it then causes an out-of-bounds access), other, unclear.
Pick the ROOT CAUSE that the fix repairs. If an unchecked integer computation leads to an out-of-bounds access, the root cause is integer_overflow. If a missing length check leads to a read past a buffer, it is oob_read. A refcount bug that frees an object still in use is use_after_free.

cwe: the most specific CWE id for that mechanism, e.g. CWE-125, CWE-787, CWE-476, CWE-416, CWE-415, CWE-401, CWE-772, CWE-362, CWE-369, CWE-835, CWE-674, CWE-908, CWE-190.

oob_effect: 1 if the flaw's effect in this function is (also) an out-of-bounds read or write of memory, else 0.

mechanism_summary: one sentence naming the flaw precisely: what goes wrong, on which variable/call, under what condition.

Reply with one JSON object only:
{"in_function": "yes"|"no"|"unclear", "mechanism": "<one of the list>", "cwe": "CWE-<n>", "oob_effect": 0|1, "mechanism_summary": "<one sentence>", "confidence": "high"|"medium"|"low"}"""

CACHE = Memo("rv5_label")


def user(r, cve_desc):
    code = r["code"]
    if len(code) > 14000:
        code = code[:14000] + "\n... [truncated]"
    diff = (r.get("fix_diff") or "(no diff available)")[:8000]
    cve = f"CVE ({r.get('cve')}): {cve_desc}" if r.get("cve") else "CVE: (not linked in this source)"
    return (f"SOURCE CWE LABEL: {', '.join(r['cwes']) or 'none'}\n\n{cve}\n\n"
            f"FIX COMMIT MESSAGE:\n{(r.get('commit_msg') or '')[:1500]}\n\nFUNCTION (before the fix):\n```\n{code}\n```\n\n"
            f"DIFF OF THIS FUNCTION (before -> after):\n{diff}")


def label(r, cve_desc):
    k = key("rv5_label_v1", LABELER, r["code_sha1"], r.get("cve"))
    if k in CACHE:
        return CACHE[k]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user(r, cve_desc)}]
    for _ in range(3):
        try:
            m = common.openrouter(LABELER, msgs, max_tokens=1500, response_format={"type": "json_object"},
                                  extra_body={"reasoning": {"effort": "low"}})
            o = parse_json(m.content)
            if o.get("mechanism") in MECHS and o.get("in_function") in ("yes", "no", "unclear"):
                CACHE[k] = o
                return o
        except Exception as e:  # noqa: BLE001
            print("  label error", type(e).__name__, str(e)[:100], flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--skip-families", default="int")
    ap.add_argument("--workers", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    desc = json.load(open(os.path.join(root, "runs", "rv5", "cve_desc.json")))
    rows = [json.loads(l) for f in ("pool.jsonl", "pool_extra.jsonl", "pool_juliet.jsonl", "pool_repro.jsonl") if os.path.exists(os.path.join(root, "runs", "rv5", f))
            for l in open(os.path.join(root, "runs", "rv5", f))]
    fams, skip = set(filter(None, a.families.split(","))), set(filter(None, a.skip_families.split(",")))
    rows = [r for r in rows if (not fams or set(r["family_src"]) & fams) and not set(r["family_src"]) <= skip]
    if a.limit:
        rows = rows[:a.limit]
    print("labelling", len(rows), flush=True)
    out = pmap(lambda r: label(r, desc.get(r.get("cve"), r.get("cve_desc", ""))), rows, a.workers, desc="label")
    print("failed", sum(o is None for o in out))


if __name__ == "__main__":
    main()
