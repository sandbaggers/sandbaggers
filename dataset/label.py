"""The two LLM passes over the functions. Source labels are noisy in two ways: MegaVul marks every function a fix
commit touches as vulnerable (often a companion change), and PrimeVul / NVD CWEs are often generic (CWE-20/703/399) or
name the consequence instead of the mechanism (CWE-119 for a use-after-free).

  label    every pool function, before any scan (gpt-5.6-luna): is the flaw the CVE describes in this function, and
           what is its root-cause mechanism?
  review   the gate's survivors (Claude Opus 5.5, extended reasoning, tier-blind): is the flaw real and in the code as
           shown, its mechanism and CWE, and the one-sentence ground truth the recall judge grades against. The source
           CWE label is shown as a claim to check, never as the answer.
"""
import json
import os

import common
from common import Memo, key, parse_json, pmap
from dataset import WORK, load_pool

LABELER, REVIEWER = "openai/gpt-5.6-luna", "anthropic/claude-opus-5.5"
LABELS, REVIEWS = Memo("rv5_label"), Memo("rv5_review")
MECHS = ("oob_read", "oob_write", "null_deref", "use_after_free", "double_free", "memory_leak", "resource_leak",
         "race", "div_by_zero", "infinite_loop", "uncontrolled_recursion", "uninit_use", "integer_overflow", "other")

LABEL_SYSTEM = """You triage entries of a vulnerability dataset. Each entry is ONE C/C++ function, shown as it was BEFORE a security fix, plus the CVE description, the source dataset's CWE label (often generic or wrong), the fix commit message, and the diff the fix made to THIS function. Datasets often mislabel: a fix commit can touch several functions, and only some contain the flaw; the shown function may already contain the fix; the CWE may name a consequence rather than the mechanism.

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

REVIEW_SYSTEM = """You are a senior C/C++ security engineer auditing a vulnerability dataset entry by entry. Each entry is ONE function as it was BEFORE a security fix, with whatever evidence the source dataset has: the CVE text, the source's CWE label, the fix commit message and the diff the fix made to this function. Some entries are synthetic test cases (NIST SARD/Juliet) with no CVE or diff; for those, judge the code itself.

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


def cve_desc():
    return json.load(open(os.path.join(WORK, "cve_desc.json")))


def label_key(r):
    return key("rv5_label_v1", LABELER, r["code_sha1"], r.get("cve"))


def review_key(r):
    return key("rv5_review_v1", REVIEWER, r["code_sha1"], r.get("cve"))


def label_prompt(r, desc):
    code = r["code"] if len(r["code"]) <= 14000 else r["code"][:14000] + "\n... [truncated]"
    diff = (r.get("fix_diff") or "(no diff available)")[:8000]
    cve = f"CVE ({r.get('cve')}): {desc}" if r.get("cve") else "CVE: (not linked in this source)"
    return (f"SOURCE CWE LABEL: {', '.join(r['cwes']) or 'none'}\n\n{cve}\n\n"
            f"FIX COMMIT MESSAGE:\n{(r.get('commit_msg') or '')[:1500]}\n\nFUNCTION (before the fix):\n```\n{code}\n```\n\n"
            f"DIFF OF THIS FUNCTION (before -> after):\n{diff}")


def review_prompt(r, desc):
    code = r["code"] if len(r["code"]) <= 20000 else r["code"][:20000] + "\n... [truncated for review]"
    parts = [f"SOURCE: {r['source']}" + (f" ({r.get('project')})" if r.get("project") else ""),
             f"SOURCE CWE LABEL (a claim to check): {', '.join(r.get('cwes') or []) or 'none'}"]
    if r.get("cve"):
        parts.append(f"CVE {r['cve']}: {desc or '(no description)'}")
    if r.get("commit_msg"):
        parts.append("FIX COMMIT MESSAGE:\n" + r["commit_msg"][:2000])
    parts.append(f"FUNCTION (before the fix):\n```\n{code}\n```")
    parts.append("DIFF OF THIS FUNCTION (before -> after):\n"
                 + (r.get("fix_diff") or "(none: synthetic test case or not available)")[:10000])
    return "\n\n".join(parts)


def ask(cache, k, model, system, user, valid, tries, **kw):
    if k in cache:
        return cache[k]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(tries):
        try:
            o = parse_json(common.openrouter(model, msgs, **kw).content)
            if valid(o):
                cache[k] = o
                return o
        except Exception as e:  # noqa: BLE001
            if "Key limit exceeded" in str(e):
                raise
            print("  error", type(e).__name__, str(e)[:120], flush=True)
    return None


def label(r, desc):
    return ask(LABELS, label_key(r), LABELER, LABEL_SYSTEM, label_prompt(r, desc),
               lambda o: o.get("mechanism") in MECHS + ("unclear",) and o.get("in_function") in ("yes", "no", "unclear"),
               3, max_tokens=1500, response_format={"type": "json_object"}, extra_body={"reasoning": {"effort": "low"}})


def review(r, desc):
    return ask(REVIEWS, review_key(r), REVIEWER, REVIEW_SYSTEM, review_prompt(r, desc),
               lambda o: o.get("verdict") in ("REAL", "NOT_REAL", "UNCLEAR") and o.get("mechanism") in MECHS,
               4, max_tokens=12000, extra_body={"reasoning": {"max_tokens": 6000}})


def label_all(skip_families=("int",), workers=400):
    """Every pool function but those whose source label is only integer arithmetic (neither tier)."""
    desc = cve_desc()
    rows = [r for r in load_pool() if not set(r["family_src"]) <= set(skip_families)]
    out = pmap(lambda r: label(r, desc.get(r.get("cve"), r.get("cve_desc", ""))), rows, workers, desc="label")
    print(f"labelled {len(rows)}, failed {sum(o is None for o in out)}", flush=True)


def review_all(workers=64):
    """The survivors (runs/rv5/review_todo.json)."""
    desc, want = cve_desc(), set(json.load(open(os.path.join(WORK, "review_todo.json"))))
    rows = [r for r in load_pool() if r["id"] in want]
    out = pmap(lambda r: review(r, desc.get(r.get("cve") or "", r.get("cve_desc", ""))), rows, workers, desc="review")
    print(f"reviewed {len(rows)}, failed {sum(o is None for o in out)}", flush=True)
