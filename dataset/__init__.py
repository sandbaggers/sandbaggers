"""The dataset (Sec. 3.3), data/realvul5: 250 easy and 250 hard vulnerable C/C++ functions (under 300 lines), each
with one labelled vulnerability, split 60/20/20.

Tiers are bug families, not single CWEs: easy = out-of-bounds read / write (CWE-125, CWE-787); hard = NULL dereference,
memory / resource leak, use-after-free / double free, uninitialized use, race. Integer overflow is in neither tier: its
flaw in the code is usually an out-of-bounds access, which the judge credits against a bounds report. Sources, C/C++
only in both tiers: PrimeVul v0.1, MegaVul, CrossVul, NIST SARD's Juliet, and BigVul, DiverseVul, CVEfixes and ICVul
(data/sources/pool_repro.jsonl.gz, imported from an earlier build). Every hard item is paired with an easy one from the
same source.

    python -m dataset <step>     # the steps in order: see dataset/__main__.py

Sampling makes the build non-deterministic: data/realvul5 is the dataset the paper used. The raw corpora are
downloaded once into RV5_CACHE (default data/datasets/cache); intermediate files go to runs/rv5.
"""
import collections
import hashlib
import json
import os
import re

import common

ROOT = common.ROOT
CACHE = os.environ.get("RV5_CACHE", os.path.join(ROOT, "data", "datasets", "cache"))
WORK = os.path.join(ROOT, "runs", "rv5")
POOLS = ("pool.jsonl", "pool_extra.jsonl", "pool_juliet.jsonl", "pool_repro.jsonl")
CANDS = os.path.join(WORK, "candidates.jsonl")
OUT = os.path.join(ROOT, "data", "realvul5")
SEED = 20261001
MAX_LINES = 300
common.OPENROUTER_POOL.update(connections=6000, timeout=1200.0)   # thousands of calls in flight

# bug family of a CWE label. 'amb' = the label does not name a mechanism (relabel from CVE text + fix, or drop).
FAMILY = {}
for fam, cwes in {
    "oob": "119 120 121 122 123 124 125 126 127 129 131 193 786 787 788 805 806",
    "null": "476 690",
    "uaf": "416 415 825 763 761",
    "leak": "401 772 404 459",
    "race": "362 366 367 667 820 821 833",
    "div0": "369",
    "loop": "835 834 674",
    "uninit": "908 909 457 456 824",
    "int": "189 190 191 128 680 681 682",
    "amb": "20 703 399 264 200 17 18 16 19 254 284 400 617 754 252 noinfo Other",
}.items():
    for c in cwes.split():
        FAMILY["NVD-CWE-" + c if c in ("noinfo", "Other") else f"CWE-{c}"] = fam
EASY_FAMS = {"oob"}
HARD_FAMS = {"null", "uaf", "leak", "race", "div0", "loop", "uninit"}   # candidates; build keeps the paper's five
MECH = {  # labeller / reviewer mechanism -> (family, canonical CWE)
    "oob_read": ("oob", "CWE-125"), "oob_write": ("oob", "CWE-787"), "null_deref": ("null", "CWE-476"),
    "use_after_free": ("uaf", "CWE-416"), "double_free": ("uaf", "CWE-415"), "memory_leak": ("leak", "CWE-401"),
    "resource_leak": ("leak", "CWE-772"), "race": ("race", "CWE-362"), "div_by_zero": ("div0", "CWE-369"),
    "infinite_loop": ("loop", "CWE-835"), "uncontrolled_recursion": ("loop", "CWE-674"),
    "uninit_use": ("uninit", "CWE-908"), "integer_overflow": ("int", "CWE-190"), "other": ("other", None),
    "unclear": ("other", None)}
#: the CWE names a ground truth opens with
NAMES = {
    "CWE-125": "Out-of-bounds Read", "CWE-787": "Out-of-bounds Write", "CWE-476": "NULL Pointer Dereference",
    "CWE-401": "Missing Release of Memory after Effective Lifetime (memory leak)",
    "CWE-772": "Missing Release of Resource after Effective Lifetime", "CWE-416": "Use After Free",
    "CWE-415": "Double Free", "CWE-908": "Use of Uninitialized Resource",
    "CWE-362": "Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')",
    "CWE-369": "Divide By Zero", "CWE-835": "Loop with Unreachable Exit Condition ('Infinite Loop')",
    "CWE-674": "Uncontrolled Recursion", "CWE-190": "Integer Overflow or Wraparound"}


def family(cwe):
    return FAMILY.get(cwe, "other")


def norm(code):
    """Whitespace-insensitive identity of a function body (dedup across sources)."""
    return hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()


def gt_text(cwe, summary, cve=None):
    """A ground truth: the CWE and its name, then this instance's flaw in one sentence."""
    return f"{cwe}: {NAMES.get(cwe, cwe)}\nThis instance{f' ({cve})' if cve else ''}: {summary}"


def load_pool():
    return [json.loads(l) for f in POOLS if os.path.exists(p := os.path.join(WORK, f)) for l in open(p)]


def load_cands():
    return [json.loads(l) for l in open(CANDS)]


def proj(x):
    """Project name normalised across sources ('torvalds/linux' in MegaVul is 'linux' in PrimeVul)."""
    raw = (x.get("project") or "?").lower()
    if raw.startswith("kernel/git") or "torvalds" in raw:
        return "linux"
    p = raw.split("/")[-1]
    return {"linux-2.6": "linux", "kernel": "linux"}.get(p, p)


def tally(xs):
    return dict(collections.Counter(xs).most_common())
