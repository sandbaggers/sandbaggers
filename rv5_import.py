"""BigVul / DiverseVul / CVEfixes / ICVul / NVD functions not in this build's other pools -> runs/rv5/pool_repro.jsonl.

They come from the earlier realvul_easy4 build (dataset_easy4*.py), whose pool (runs/dataset_easy4/pool.jsonl in the
`repro` worktree, 200 MB) was never committed. The output is committed instead:
data/realvul5/sources/pool_repro.jsonl.gz is byte-identical to the pool_repro.jsonl the realvul5 build used (7,460
rows), and is what this script unpacks by default. --from regenerates it from an easy4 build's runs/ dir,
deduplicated against runs/rv5/pool{,_extra,_juliet}.jsonl. The build also copied the repro B scan cache's verdicts
for these functions into its own; none belongs to a final item, so that step is gone and B scans them like any
other source.

    python rv5_import.py                          # vendored copy -> runs/rv5/pool_repro.jsonl
    python rv5_import.py --from <repro>/runs      # regenerate from <repro>/runs/dataset_easy4/pool.jsonl
"""
import argparse
import ast
import collections
import gzip
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realvul5 import ROOT, WORK, family, norm  # noqa: E402

SOURCES = {"bigvul", "diversevul", "cvefixes", "icvul", "nvd"}
OUT = os.path.join(WORK, "pool_repro.jsonl")
VENDORED = os.path.join(ROOT, "data", "realvul5", "sources", "pool_repro.jsonl.gz")


def project_of(url):
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"[?;]p=([^;]+?)(\.git)?;", url or "")
    return m.group(1) if m else None


def regenerate(runs):
    seen = set()
    for f in ("pool.jsonl", "pool_extra.jsonl", "pool_juliet.jsonl"):
        for l in open(os.path.join(WORK, f)):
            seen.add(json.loads(l)["code_sha1"])
    rows, n = [], collections.Counter()
    for l in open(os.path.join(runs, "dataset_easy4", "pool.jsonl")):
        r = json.loads(l)
        if r["source"] not in SOURCES or not (r.get("code") or "").strip():
            continue
        h = norm(r["code"])
        if h in seen:
            n[f"dup_{r['source']}"] += 1
            continue
        seen.add(h)
        try:
            cwes = ast.literal_eval(r.get("original_cwes") or "[]") or [r["cwe"]]
        except (ValueError, SyntaxError):
            cwes = [r["cwe"]]
        cwes = [c for c in cwes if isinstance(c, str) and c.startswith(("CWE-", "NVD-"))]
        d = r.get("description") or ""
        m = re.search(r"This instance \([^)]*\):\s*(.*)", d, flags=re.S)
        rows.append({"id": r["id"], "source": r["source"], "code": r["code"].strip("\n"), "cwes": cwes,
                     "cve": r.get("cve"), "cve_desc": (m.group(1).strip() if m else ""), "project": project_of(r.get("commit_url")),
                     "commit": (r.get("commit_url") or "").rstrip("/").rsplit("/", 1)[-1][-40:] or None,
                     "commit_url": r.get("commit_url"), "commit_msg": "", "file": None, "fix_diff": r.get("fix_diff"),
                     "lang": "c/cpp", "code_sha1": h, "family_src": sorted({family(c) for c in cwes}) or ["none"],
                     "imported_from": "repro/runs/dataset_easy4/pool.jsonl"})
        n[r["source"]] += 1
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("new rows", dict(n))
    return rows


def unpack():
    os.makedirs(WORK, exist_ok=True)
    with gzip.open(VENDORED, "rb") as src, open(OUT, "wb") as dst:
        shutil.copyfileobj(src, dst)
    rows = [json.loads(l) for l in open(OUT)]
    print("unpacked", len(rows), "rows from", os.path.relpath(VENDORED, ROOT))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="runs", help="runs/ dir of an easy4 build; regenerate instead of unpacking")
    a = ap.parse_args()
    rows = regenerate(a.runs) if a.runs else unpack()
    print(collections.Counter((r["source"], r["family_src"][0]) for r in rows).most_common(40))
