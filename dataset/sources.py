"""The candidate pools, one file per source group in runs/rv5 (deduplicated across sources by whitespace-insensitive
body), and the CVE descriptions the labeller reads.

    pool.jsonl          PrimeVul v0.1 (target == 1), MegaVul
    pool_extra.jsonl    CrossVul
    pool_juliet.jsonl   NIST SARD Juliet C/C++ 1.3
    pool_repro.jsonl    BigVul, DiverseVul, CVEfixes, ICVul (vendored: data/sources/pool_repro.jsonl.gz)
"""
import collections
import difflib
import gzip
import json
import os
import random
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from dataset import CACHE, ROOT, WORK, family, norm


def lang(path):
    p = (path or "").lower()
    return "cpp" if p.endswith((".cc", ".cpp", ".cxx", ".hpp", ".hh", ".c++", ".h++", ".hxx", ".inl", ".ipp", ".tcc")) \
        else "c" if p.endswith((".c", ".h")) else "?"


def write(name, rows):
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, name), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def dedup(rows, seen):
    """Rows whose body is new (the first source keeps it), with code_sha1 and the source label's family."""
    out, drop = [], collections.Counter()
    for r in rows:
        h = norm(r["code"])
        if r["code"].count("\n") < 2:
            drop["short"] += 1
            continue
        if h in seen:
            drop[f"dup_{r['source']}"] += 1
            continue
        seen.add(h)
        out.append({**r, "code_sha1": h, "family_src": sorted({family(c) for c in r["cwes"]}) or ["none"]})
    print(f"  {len(out)} rows; dropped {dict(drop)}", flush=True)
    return out

# ----------------------------------------------------------------------------- PrimeVul, MegaVul


def primevul():
    d = os.path.join(CACHE, "primevul_orig")
    diffs = {}
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
                   "fix_diff": diffs.get(norm(r["func"])),
                   "lang": lang(r.get("file_name")) if r.get("file_name") not in (None, "None") else "c/cpp"}


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

# ----------------------------------------------------------------------------- CrossVul

C_LIKE = ("c", "cpp", "cc", "cxx", "h", "hpp", "hh")


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
    """The functions a CrossVul bad/good file pair changes."""
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

# ----------------------------------------------------------------------------- Juliet

#: Juliet CWE -> the CWE it counts as: the bounds family and the hard families
JULIET = {121: "CWE-787", 122: "CWE-787", 124: "CWE-787", 126: "CWE-125", 127: "CWE-125",
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


def juliet(per_cwe=120):
    """The `bad` function of self-contained test cases: only flow variants whose flaw sits inside the one bad function
    (01-19 control flow, 31-32 data flow). Comments are stripped (Juliet's say "FLAW" / "FIX") and the function is
    renamed (its name spells out the CWE)."""
    z = zipfile.ZipFile(os.path.join(CACHE, "sard", "juliet_c_1.3.zip"))
    by = collections.defaultdict(list)
    for n in z.namelist():
        m = re.search(r"testcases/CWE(\d+)_[^/]*(/s\d+)?/(CWE\d+_[^/]+)$", n)
        if not m or int(m.group(1)) not in JULIET:
            continue
        v = VARIANT.search(n)
        if not v or not (1 <= int(v.group(1)) <= 19 or int(v.group(1)) in (31, 32)) or re.search(r"_\d\d[a-z]\.", n):
            continue
        by[int(m.group(1))].append(n)
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
                continue
            code = strip_comments(f).strip("\n")
            code = code.replace(name, random.Random(n).choice(NEUTRAL) + f"_{random.Random(n + 'x').randint(10, 99)}")
            if re.search(r"CWE\d+|_bad\b|good", code):
                continue
            cwe = JULIET[cwe_n]
            yield {"id": f"juliet_{os.path.basename(n).rsplit('.', 1)[0]}", "source": "sard", "code": code,
                   "cwes": [cwe], "cve": None, "cve_desc": "", "project": "juliet", "commit": None,
                   "commit_url": None, "commit_msg": "", "file": n, "fix_diff": None, "juliet_cwe": f"CWE-{cwe_n}",
                   "lang": "cpp" if n.endswith(".cpp") else "c"}
            got += 1

# ----------------------------------------------------------------------------- CVE descriptions

CVE_DESC = os.path.join(WORK, "cve_desc.json")


def fetch_cve(cve):
    """A CVE's English description from CVEProject/cvelistV5 ('' if it has none, None if the fetch failed)."""
    try:
        _, y, n = cve.split("-")
        url = f"https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/{y}/{int(n) // 1000}xxx/{cve}.json"
        for _ in range(4):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=30))
                ds = d["containers"]["cna"].get("descriptions") or []
                en = [x["value"] for x in ds if x.get("lang", "").startswith("en")] or [x["value"] for x in ds]
                return cve, (en[0].strip() if en else "")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return cve, ""
            except Exception:  # noqa: BLE001
                pass
    except ValueError:
        pass
    return cve, None


def cve_descriptions(rows):
    have = json.load(open(CVE_DESC)) if os.path.exists(CVE_DESC) else {}
    for r in rows:
        if r.get("cve_desc") and r.get("cve"):
            have.setdefault(r["cve"], r["cve_desc"])
    todo = sorted({r["cve"] for r in rows if (r.get("cve") or "").startswith("CVE-") and not have.get(r["cve"])})
    with ThreadPoolExecutor(64) as ex:
        have.update({c: d for c, d in ex.map(fetch_cve, todo) if d})
    json.dump(have, open(CVE_DESC, "w"))
    print(f"cve descriptions: {sum(1 for r in rows if have.get(r.get('cve')))} of {len(rows)} rows", flush=True)


def main():
    seen = set()
    print("PrimeVul + MegaVul", flush=True)
    write("pool.jsonl", dedup((r for src in (primevul, megavul) for r in src()), seen))
    print("CrossVul", flush=True)
    write("pool_extra.jsonl", dedup(crossvul(), seen))
    print("Juliet", flush=True)
    write("pool_juliet.jsonl", dedup(juliet(), seen))
    with gzip.open(os.path.join(ROOT, "data", "sources", "pool_repro.jsonl.gz"), "rb") as src, \
            open(os.path.join(WORK, "pool_repro.jsonl"), "wb") as dst:
        shutil.copyfileobj(src, dst)
    from dataset import load_pool
    cve_descriptions(load_pool())
