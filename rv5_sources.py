"""CrossVul -> runs/rv5/pool_extra.jsonl, deduplicated against pool.jsonl (NIST SARD / Juliet is rv5_juliet.py).

    python rv5_sources.py crossvul
"""
import collections
import difflib
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realvul5 import CACHE, POOL, WORK, family, norm  # noqa: E402

EXTRA = os.path.join(WORK, "pool_extra.jsonl")
C_LIKE = ("c", "cpp", "cc", "cxx", "h", "hpp", "hh")


HEADER = re.compile(r"^[A-Za-z_][\w\s\*\(\),:<>&~\[\]]*\)\s*(const)?\s*(\{)?\s*$")


def functions(src):
    """Top-level function definitions of a C/C++ file -> [(start_line, end_line, text)] (heuristic brace matcher)."""
    lines = src.split("\n")
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        # a header starts in column 0, is not a statement/preprocessor/comment, and its '{' opens on this or a later line
        if l and not l[0].isspace() and l[0] not in "#/*}{" and "(" in l and not l.rstrip().endswith(";") \
                and not l.startswith(("typedef", "struct ", "enum ", "union ", "extern \"C\"", "namespace", "static const", "return")):
            j = i
            while j < len(lines) and "{" not in lines[j] and not lines[j].rstrip().endswith(";") and j - i < 12:
                j += 1
            if j < len(lines) and "{" in lines[j] and not lines[j].rstrip().endswith(";"):
                depth, k, started = 0, j, False
                while k < len(lines):
                    s = re.sub(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'|//.*', "", lines[k])
                    depth += s.count("{") - s.count("}")
                    started = started or "{" in s
                    if started and depth <= 0:
                        break
                    k += 1
                if k < len(lines) and k - i >= 2:
                    out.append((i, k, "\n".join(lines[i:k + 1])))
                    i = k + 1
                    continue
        i += 1
    return out


def crossvul():
    z = zipfile.ZipFile(os.path.join(CACHE, "crossvul/dataset.zip"))
    names = set(z.namelist())
    for n in sorted(names):
        p = n.split("/")
        if len(p) != 4 or not p[3].startswith("bad_") or p[2].lower() not in C_LIKE:
            continue
        good = n.replace("/bad_", "/good_")
        if good not in names:
            continue
        try:
            bad_src = z.read(n).decode("utf-8", "replace")
            good_src = z.read(good).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
        if len(bad_src) > 2_000_000:
            continue
        gf = {re.sub(r"\s+", "", t.split("{", 1)[0]): t for _, _, t in functions(good_src)}
        for s, e, t in functions(bad_src):
            sig = re.sub(r"\s+", "", t.split("{", 1)[0])
            g = gf.get(sig)
            if g is None or norm(g) == norm(t):
                continue        # unchanged (or unmatched) function
            cwe = p[1]
            yield {"id": f"crossvul_{p[1]}_{p[3]}_{s}", "source": "crossvul", "code": t, "cwes": [] if cwe == "None" else [cwe],
                   "cve": None, "cve_desc": "", "project": None, "commit": p[3].split("_")[1], "commit_url": None,
                   "commit_msg": "", "file": n, "func_name": sig.split("(")[0][-60:],
                   "fix_diff": "".join(difflib.unified_diff(t.splitlines(True), g.splitlines(True), "vulnerable", "fixed", n=2)),
                   "lang": "cpp" if p[2].lower() in ("cpp", "cc", "cxx", "hpp", "hh") else "c"}


def main(srcs):
    seen = {json.loads(l)["code_sha1"] for l in open(POOL)}
    have = {}
    if os.path.exists(EXTRA):
        for l in open(EXTRA):
            r = json.loads(l)
            if r["source"] not in srcs:
                have[r["code_sha1"]] = r
    seen |= set(have)
    n, dup = collections.Counter(), collections.Counter()
    rows = list(have.values())
    for s in srcs:
        for r in {"crossvul": crossvul}[s]():
            h = norm(r["code"])
            if r["code"].count("\n") < 2:
                continue
            if h in seen:
                dup[s] += 1
                continue
            seen.add(h)
            r["code_sha1"] = h
            r["family_src"] = sorted({family(c) for c in r["cwes"]}) or ["none"]
            rows.append(r)
            n[s] += 1
    with open(EXTRA, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("new", dict(n), "duplicates of earlier sources", dict(dup))
    print(collections.Counter((r["source"], r["family_src"][0]) for r in rows if r["source"] in srcs).most_common(30))


if __name__ == "__main__":
    main(sys.argv[1:])
