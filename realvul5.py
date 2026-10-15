"""realvul5: 250 easy / 250 hard vulnerable C/C++ functions with disjoint, conceptually distinct bug families.

Tiers are bug FAMILIES (each a small set of CWE labels), not single CWEs:
    easy  = spatial memory errors: out-of-bounds read / write, buffer overflow (CWE-787, CWE-125)
    hard  = non-spatial bugs: NULL dereference, memory leak, use-after-free / double free, uninitialized use, race
            (division by zero and infinite loops / recursion are candidates too; rv5_assemble.py build drops them)
Integer overflow / numeric errors are in NEITHER tier: their flaw in the code is usually an out-of-bounds access,
which the judge credits against a bounds report (the confusion that capped selectivity on realvul_v2).

Sources (C/C++ only in both tiers, so language cannot separate them), one pool file each in runs/rv5:
    pool.jsonl          PrimeVul v0.1 (original jsonl, target == 1), MegaVul     realvul5.py pool
    pool_extra.jsonl    CrossVul                                                  rv5_sources.py crossvul
    pool_juliet.jsonl   NIST SARD (Juliet C/C++ 1.3)                              rv5_juliet.py
    pool_repro.jsonl    BigVul, DiverseVul, CVEfixes, ICVul, NVD                  rv5_import.py

    python realvul5.py pool          # PrimeVul + MegaVul -> runs/rv5/pool.jsonl (dedup, family of the source label)
"""
import argparse
import collections
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
#: the raw corpora (PrimeVul, MegaVul, ...), downloaded once: RV5_CACHE, else data/datasets/cache
CACHE = os.environ.get("RV5_CACHE", os.path.join(ROOT, "data", "datasets", "cache"))
WORK = os.path.join(ROOT, "runs", "rv5")
POOL = os.path.join(WORK, "pool.jsonl")

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
EASY_FAMS, HARD_FAMS = {"oob"}, {"null", "uaf", "leak", "race", "div0", "loop", "uninit"}


def family(cwe):
    return FAMILY.get(cwe, "other")


def norm(code):
    """Whitespace-insensitive identity of a function body (dedup across sources)."""
    return hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()


def lang(path):
    p = (path or "").lower()
    return "cpp" if p.endswith((".cc", ".cpp", ".cxx", ".hpp", ".hh", ".c++", ".h++", ".hxx", ".inl", ".ipp", ".tcc")) \
        else "c" if p.endswith((".c", ".h")) else "?"

# ----------------------------------------------------------------------------- sources


def primevul():
    d = os.path.join(CACHE, "primevul_orig")
    diffs = {}
    import difflib
    for sp in ("train", "valid", "test"):
        rows = [json.loads(l) for l in open(os.path.join(d, f"primevul_{sp}_paired.jsonl"))]
        for a, b in zip(rows[::2], rows[1::2]):
            if a["target"] == 1 and b["target"] == 0 and a["commit_id"] == b["commit_id"]:
                diffs[norm(a["func"])] = "".join(difflib.unified_diff(
                    a["func"].splitlines(True), b["func"].splitlines(True), "vulnerable", "fixed", n=2))
    for sp in ("train", "valid", "test"):
        for line in open(os.path.join(d, f"primevul_{sp}.jsonl")):
            r = json.loads(line)
            if r["target"] != 1:
                continue
            yield {"id": f"primevul_{r['idx']}", "source": "primevul", "code": r["func"].strip("\n"),
                   "cwes": r.get("cwe") or [], "cve": r.get("cve"), "cve_desc": (r.get("cve_desc") or "").strip(),
                   "project": r["project"], "commit": r["commit_id"], "commit_url": r.get("commit_url"),
                   "commit_msg": (r.get("commit_message") or "")[:3000], "file": r.get("file_name"),
                   "fix_diff": diffs.get(norm(r["func"])), "lang": lang(r.get("file_name")) if r.get("file_name") not in (None, "None") else "c/cpp"}


def megavul():
    import ijson
    with open(os.path.join(CACHE, "megavul", "megavul.json"), "rb") as f:
        for k, r in enumerate(ijson.items(f, "item")):
            if not r.get("is_vul") or not (r.get("func_before") or "").strip():
                continue
            yield {"id": f"megavul_{k}", "source": "megavul", "code": r["func_before"].strip("\n"),
                   "cwes": list(r.get("cwe_ids") or []), "cve": r.get("cve_id"), "cve_desc": "",
                   "project": r.get("repo_name"), "commit": r.get("commit_hash"), "commit_url": r.get("git_url"),
                   "commit_msg": (r.get("commit_msg") or "")[:3000], "file": r.get("file_path"),
                   "func_name": r.get("func_name"), "fix_diff": r.get("diff_func"), "lang": lang(r.get("file_path"))}


def build_pool():
    os.makedirs(WORK, exist_ok=True)
    seen, n, drop = {}, collections.Counter(), collections.Counter()
    with open(POOL, "w") as out:
        for src in (primevul, megavul):
            for r in src():
                h = norm(r["code"])
                if r["code"].count("\n") < 2:
                    drop["short"] += 1
                    continue
                if h in seen:              # keep the first source; record the duplicate's provenance
                    drop[f"dup_{r['source']}"] += 1
                    continue
                seen[h] = r["id"]
                r["code_sha1"] = h
                r["family_src"] = sorted({family(c) for c in r["cwes"]}) or ["none"]
                out.write(json.dumps(r) + "\n")
                n[r["source"]] += 1
    print("pool", dict(n), "dropped", dict(drop))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    a = ap.parse_args()
    {"pool": build_pool}[a.cmd]()
