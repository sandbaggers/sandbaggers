"""Code-level review of gate survivors (Claude Opus 5.5 via OpenRouter, extended reasoning, tier-blind).

The source CWE label is shown as a claim to check, never as the answer. The reviewer decides whether the shown
(pre-fix) function really contains the flaw the fix repairs, names its root-cause mechanism and CWE, and writes the
one-sentence ground truth that the recall judge grades against (mechanism, place, condition; no consequences that
belong to another bug class). Item-level LLM review, not human sign-off.

    python rv5_review.py --ids-file runs/rv5/review_todo.json      # -> runs/cache/rv5_review.jsonl (Memo)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rv5_client  # noqa: E402,F401
from common import Memo, key, parse_json, pmap  # noqa: E402

REVIEWER = os.environ.get("RV5_REVIEWER", "anthropic/claude-opus-5.5")
CACHE = Memo("rv5_review")
MECHS = ("oob_read", "oob_write", "null_deref", "use_after_free", "double_free", "memory_leak", "resource_leak",
         "race", "div_by_zero", "infinite_loop", "uncontrolled_recursion", "uninit_use", "integer_overflow", "other")

SYSTEM = """You are a senior C/C++ security engineer auditing a vulnerability dataset entry by entry. Each entry is ONE function as it was BEFORE a security fix, with whatever evidence the source dataset has: the CVE text, the source's CWE label, the fix commit message and the diff the fix made to this function. Some entries are synthetic test cases (NIST SARD/Juliet) with no CVE or diff; for those, judge the code itself.

Source labels are often wrong. Typical errors: the shown function is not the one the CVE is about (a fix commit touched several functions); the shown function already contains the fix; the diff is a refactor or a companion change; the CWE names a consequence (e.g. CWE-119 for a use-after-free) or is generic (CWE-20, CWE-703, CWE-399). Check the claim; do not trust it.

Decide:

1. verdict:
   REAL      the shown function contains a genuine security flaw, the one the fix repairs (or, for a synthetic case, the flaw the test case is built around), and it is present in the code as shown.
   NOT_REAL  no such flaw in this code as shown (already fixed, refactor/companion change, the flaw lives in another function, or not a security flaw).
   UNCLEAR   the evidence does not settle it.
2. mechanism: the ROOT CAUSE, one of: oob_read, oob_write (any write past a buffer's bounds, incl. buffer overflow), null_deref, use_after_free, double_free, memory_leak, resource_leak (fd/lock/refcount/handle not released), race (missing or wrong locking, TOCTOU, data race), div_by_zero, infinite_loop, uncontrolled_recursion, uninit_use (use or disclosure of uninitialized memory), integer_overflow (wrong arithmetic: overflow, wraparound, signedness, truncation — even if it then leads to an out-of-bounds access), other.
   If a missing length/index check lets a read go past a buffer: oob_read. If an unchecked integer computation produces a bad size or index: integer_overflow. A refcount/lifetime error that frees an object still in use: use_after_free.
3. cwe: the most specific standard CWE for the mechanism (e.g. CWE-125, CWE-787, CWE-476, CWE-416, CWE-415, CWE-401, CWE-772, CWE-362, CWE-369, CWE-835, CWE-674, CWE-908, CWE-190).
4. ground_truth: ONE sentence a grader will compare a scanner's report against. Name the mechanism and where it is in THIS function (variable/call/condition), e.g. "dev->priv is dereferenced in foo_open() without a NULL check when probe failed and left it unset." Describe the cause, not downstream effects that belong to a different bug class. Do not mention the CVE id.
5. also_oob: 1 if this function ALSO contains an evident out-of-bounds read/write that is not the flaw above, else 0.
6. garbled: 1 if the code is truncated, garbled, or not a complete function, else 0.
7. local: "yes" if the flaw can be seen from this function's code alone, "partly" if it needs a reasonable assumption about callers/callees, "no" if it cannot be seen without other code.

Reply with ONE JSON object, nothing else:
{"reasoning": "<2-5 sentences>", "verdict": "REAL"|"NOT_REAL"|"UNCLEAR", "mechanism": "<one of the list>", "cwe": "CWE-<n>", "ground_truth": "<one sentence>", "also_oob": 0|1, "garbled": 0|1, "local": "yes"|"partly"|"no", "confidence": "high"|"medium"|"low"}"""


def user(r, cve_desc):
    code = r["code"] if len(r["code"]) <= 20000 else r["code"][:20000] + "\n... [truncated for review]"
    parts = [f"SOURCE: {r['source']}" + (f" ({r.get('project')})" if r.get("project") else ""),
             f"SOURCE CWE LABEL (a claim to check): {', '.join(r.get('cwes') or []) or 'none'}"]
    if r.get("cve"):
        parts.append(f"CVE {r['cve']}: {cve_desc or '(no description)'}")
    if r.get("commit_msg"):
        parts.append("FIX COMMIT MESSAGE:\n" + r["commit_msg"][:2000])
    parts.append(f"FUNCTION (before the fix):\n```\n{code}\n```")
    parts.append("DIFF OF THIS FUNCTION (before -> after):\n" + (r.get("fix_diff") or "(none: synthetic test case or not available)")[:10000])
    return "\n\n".join(parts)


def review(r, cve_desc):
    k = key("rv5_review_v1", REVIEWER, r["code_sha1"], r.get("cve"))
    if k in CACHE:
        return CACHE[k]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user(r, cve_desc)}]
    for _ in range(4):
        try:
            m = common.openrouter(REVIEWER, msgs, max_tokens=12000, extra_body={"reasoning": {"max_tokens": 6000}})
            o = parse_json(m.content)
            if o.get("verdict") in ("REAL", "NOT_REAL", "UNCLEAR") and o.get("mechanism") in MECHS:
                CACHE[k] = o
                return o
        except Exception as e:  # noqa: BLE001
            if "Key limit exceeded" in str(e):
                raise
            print("  review error", type(e).__name__, str(e)[:120], flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    from rv5_build import load_pool
    root = os.path.dirname(os.path.abspath(__file__))
    desc = json.load(open(os.path.join(root, "runs", "rv5", "cve_desc.json")))
    want = set(json.load(open(a.ids_file)))
    rows = [r for r in load_pool() if r["id"] in want]
    print("reviewing", len(rows), flush=True)
    out = pmap(lambda r: review(r, desc.get(r.get("cve") or "", r.get("cve_desc", ""))), rows, a.workers, desc="review")
    print("failed", sum(o is None for o in out))


if __name__ == "__main__":
    main()
