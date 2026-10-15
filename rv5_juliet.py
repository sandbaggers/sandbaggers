"""NIST SARD Juliet C/C++ 1.3 -> runs/rv5/pool_juliet.jsonl: the `bad` function of self-contained test cases.

Only flow variants whose flaw sits inside the one bad function (01-19 control flow, 31-32 data flow); multi-function
and multi-file variants need code the scanner would not see. Comments are stripped (Juliet's say "FLAW" / "FIX") and
the function is renamed (its name spells out the CWE). Mapped CWEs only: the bounds family and the hard families.

    python rv5_juliet.py [--per-cwe 120]
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realvul5 import CACHE, WORK, family, norm  # noqa: E402

OUT = os.path.join(WORK, "pool_juliet.jsonl")
KEEP = {121: "CWE-787", 122: "CWE-787", 124: "CWE-787", 126: "CWE-125", 127: "CWE-125",
        476: "CWE-476", 690: "CWE-476", 416: "CWE-416", 415: "CWE-415", 401: "CWE-401", 775: "CWE-772",
        369: "CWE-369", 835: "CWE-835", 674: "CWE-674", 457: "CWE-908", 366: "CWE-362", 667: "CWE-362"}
VARIANT = re.compile(r"_(\d\d)[a-z]?\.(c|cpp)$")
NEUTRAL = ["process", "handle_request", "run_task", "update_state", "parse_input", "do_work", "service_step",
           "load_entry", "compute_result", "apply_config"]


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return re.sub(r"\n\s*\n+", "\n", s)


def bad_function(src):
    m = re.search(r"^(void|int|static void)\s+(\w+_bad)\s*\(\s*\)\s*\n?\{", src, flags=re.M)
    if not m:
        return None, None
    i, depth = src.index("{", m.start()), 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1], m.group(2)
    return None, None


def main(per_cwe):
    z = zipfile.ZipFile(os.path.join(CACHE, "sard", "juliet_c_1.3.zip"))
    by = collections.defaultdict(list)
    for n in z.namelist():
        m = re.search(r"testcases/CWE(\d+)_[^/]*(/s\d+)?/(CWE\d+_[^/]+)$", n)
        if not m or int(m.group(1)) not in KEEP:
            continue
        v = VARIANT.search(n)
        if not v or not (1 <= int(v.group(1)) <= 19 or int(v.group(1)) in (31, 32)) or re.search(r"_\d\d[a-z]\.", n):
            continue
        by[int(m.group(1))].append(n)
    rows, stats = [], collections.Counter()
    for cwe_n, names in sorted(by.items()):
        random.Random(f"juliet-{cwe_n}").shuffle(names)
        got = 0
        for n in names:
            if got >= per_cwe:
                break
            src = z.read(n).decode("utf-8", "replace")
            if "OMITBAD" not in src:
                continue
            f, name = bad_function(src)
            if not f:
                stats["no_bad"] += 1
                continue
            code = strip_comments(f).strip("\n")
            new = random.Random(n).choice(NEUTRAL) + f"_{random.Random(n + 'x').randint(10, 99)}"
            code = code.replace(name, new)
            if re.search(r"CWE\d+|_bad\b|good", code):
                stats["leaky_name"] += 1
                continue
            cwe = KEEP[cwe_n]
            rows.append({"id": f"juliet_{os.path.basename(n).rsplit('.', 1)[0]}", "source": "sard", "code": code,
                         "cwes": [cwe], "cve": None, "cve_desc": "", "project": "juliet", "commit": None,
                         "commit_url": None, "commit_msg": "", "file": n, "func_name": new, "fix_diff": None,
                         "juliet_cwe": f"CWE-{cwe_n}", "lang": "cpp" if n.endswith(".cpp") else "c",
                         "code_sha1": norm(code), "family_src": [family(cwe)]})
            got += 1
        stats[f"CWE-{cwe_n}"] = got
    seen, uniq = set(), []
    for r in rows:
        if r["code_sha1"] not in seen:
            seen.add(r["code_sha1"])
            uniq.append(r)
    with open(OUT, "w") as fo:
        for r in uniq:
            fo.write(json.dumps(r) + "\n")
    print(len(uniq), dict(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cwe", type=int, default=120)
    main(ap.parse_args().per_cwe)
