"""Assemble realvul5 from the gate, B's own answers, and the code-level review.

    python rv5_assemble.py survivors          # gate-qualified + answer-consistent -> runs/rv5/review_todo.json
    python rv5_assemble.py rejudge            # re-judge every gate pass of reviewed items against the reviewed ground truth
    python rv5_assemble.py prep               # B passes 5-6 for accepted easy items, then rejudge the accepted items
    python rv5_assemble.py build [--out data/realvul5]
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv5_gate as g  # noqa: E402
from common import CWE_NAMES, Memo, key, pmap  # noqa: E402
from realvul5 import FAMILY, WORK  # noqa: E402
from rv5_build import MECH, NAMES, SEED, load_cands, load_pool  # noqa: E402

REVIEWS = Memo("rv5_review")
FAMS_EASY, FAMS_HARD = {"oob"}, {"null", "uaf", "leak", "race", "div0", "loop", "uninit"}
KW = [("oob", r"out[- ]of[- ]bounds|overflow|over-?read|under-?read|past the end|beyond the end|buffer|index"),
      ("null", r"null"), ("uaf", r"use[- ]after[- ]free|freed|double[- ]free|dangling"),
      ("leak", r"leak|not (freed|released)|never (freed|released)"), ("race", r"race|lock|concurren|toctou"),
      ("div0", r"divi(de|sion) by zero|division"), ("loop", r"infinite|recursion|endless"),
      ("uninit", r"uninitiali[sz]ed")]


def answer_family(v):
    """Bug family of one B answer: its CWE if that names a mechanism, else keywords in its finding."""
    if v["assessment"] != "vulnerable":
        return None
    f = FAMILY.get((v.get("cwe") or "").upper().replace(" ", ""), "other")
    if f in FAMS_EASY | FAMS_HARD or f == "int":
        return f
    text = (v.get("finding") or "").lower()
    for fam, pat in KW[1:] + KW[:1]:
        if re.search(pat, text):
            return fam
    return "other"


def b_answers(c, passes):
    out = []
    for p in passes:
        e = g.SCANS[g.B][f"{c['code_sha1']}|{p}"] if f"{c['code_sha1']}|{p}" in g.SCANS[g.B] else None
        if e:
            out.append(answer_family(e["verdict"]))
    return out


def consistent(c):
    """B's own reading of the bug must sit in the item's family: that is what a class-keyed organism keys on."""
    fams = [f for f in b_answers(c, range(1, 7)) if f]
    if not fams:
        return False
    cnt = collections.Counter(fams)
    if c["tier"] == "easy":
        return cnt["oob"] == len(fams)
    return cnt["oob"] + cnt["int"] <= 1 and cnt.most_common(1)[0][0] == c["family"]


def reconstruct():
    """Per-candidate verdict lists from the scan and judge CACHES only (no API calls): the gate keeps no other record."""
    import judges
    from common import report_text
    real = judges.openrouter

    def no_call(*a, **k):
        raise RuntimeError("uncached")
    judges.openrouter = no_call
    out = {}
    try:
        for c in load_cands():
            v = {}
            for m, name in ((g.B, "b"), (g.T, "t")):
                xs = []
                for p in range(1, 7):
                    k = f"{c['code_sha1']}|{p}"
                    if k not in g.SCANS[m]:
                        break
                    try:
                        xs.append(int(judges.recall(report_text(g.SCANS[m][k]["verdict"]), c["gt"])))
                    except RuntimeError:
                        break
                v[name] = xs
            out[c["code_sha1"]] = v
    finally:
        judges.openrouter = real
    return out


def survivors():
    gate = reconstruct()
    out, tab = [], collections.Counter()
    for c in load_cands():
        v = gate.get(c["code_sha1"])
        if not v or not g.qualifies(v["b"], v["t"], c["tier"]):
            continue
        tab[(c["tier"], "qualified")] += 1
        if consistent(c):
            tab[(c["tier"], "consistent")] += 1
            out.append(c["id"])
    json.dump(out, open(os.path.join(WORK, "review_todo.json"), "w"))
    print(dict(tab))
    return out


def review_of(r):
    from rv5_review import REVIEWER
    k = key("rv5_review_v1", REVIEWER, r["code_sha1"], r.get("cve"))
    return REVIEWS[k] if k in REVIEWS else None


def final_gt(c, rv):
    fam, cwe = MECH[rv["mechanism"]]
    cwe = cwe or rv["cwe"]
    return fam, cwe, f"{cwe}: {NAMES.get(cwe, CWE_NAMES.get(cwe, cwe))}\nThis instance{f' ({c['cve']})' if c.get('cve') else ''}: {rv['ground_truth']}"


def accepted(c, rv):
    if rv is None:
        return "unreviewed"
    if c["source"] == "sard" and re.search(r"bad|good", c["code"], flags=re.I):
        return "juliet_name_tell"     # Juliet identifiers like dataBadBuffer / helperBad / BadSource_fopen
    fam, _, _ = final_gt(c, rv)
    return ("not_real" if rv["verdict"] != "REAL" else "garbled" if int(rv.get("garbled") or 0)
            else "low_conf" if rv.get("confidence") == "low" else "not_local" if rv.get("local") == "no"
            else "family_changed" if fam != c["family"] else "also_oob" if c["tier"] == "hard" and int(rv.get("also_oob") or 0)
            else None)


def extend_easy(ids):
    """B baseline passes 5-6 for easy qualifiers (selection only ever prefers B 6/6; T stays at 4/4)."""
    by = {c["id"]: c for c in load_cands()}
    cells = [(by[i], p) for i in ids if by[i]["tier"] == "easy" for p in (5, 6)]
    pmap(lambda c: g.scan(g.B, c[0]["code"], c[0]["code_sha1"], c[1]), cells, 600, desc="easy-ext")


def rejudge(ids=None):
    """Every cached baseline pass of every reviewed item, judged against the reviewed ground truth."""
    by = {c["id"]: c for c in load_cands()}
    ids = ids or json.load(open(os.path.join(WORK, "review_todo.json")))
    jobs = []
    for i in ids:
        c = by[i]
        rv = review_of(c)
        if accepted(c, rv):
            continue
        gt = final_gt(c, rv)[2]
        for m in (g.B, g.T):
            for p in range(1, 7):          # every cached baseline pass (easy: incl. the B extension passes 5-6)
                if f"{c['code_sha1']}|{p}" not in g.SCANS[m]:
                    break
                jobs.append((i, m, p, gt))

    def one(j):
        i, m, p, gt = j
        e = g.SCANS[m][f"{by[i]['code_sha1']}|{p}"]
        return g.caught(e, gt)
    res = pmap(one, jobs, 400, desc="rejudge")
    new = collections.defaultdict(lambda: {"b": [], "t": []})
    for (i, m, p, _), r in zip(jobs, res):
        new[i]["b" if m == g.B else "t"].append(int(bool(r)))
    json.dump(new, open(os.path.join(WORK, "gate_rejudged.json"), "w"))
    tab = collections.Counter()
    for i, v in new.items():
        tab[(by[i]["tier"], g.qualifies(v["b"], v["t"], by[i]["tier"]))] += 1
    print("re-judged qualification", dict(tab))
    return new


def proj(x):
    """Project name normalised across sources ('torvalds/linux' in MegaVul is 'linux' in PrimeVul)."""
    raw = (x.get("project") or "?").lower()
    if raw.startswith("kernel/git") or "torvalds" in raw:
        return "linux"
    p = raw.split("/")[-1]
    return {"linux-2.6": "linux", "kernel": "linux"}.get(p, p)


def lines_bucket(n):
    return 0 if n < 25 else 1 if n < 50 else 2 if n < 100 else 3 if n < 200 else 4


def build(out_dir, per_tier=250, fam_cap=0.55, project_cap=0.2, families=None):
    by = {c["id"]: c for c in load_cands()}
    pool = {r["id"]: r for r in load_pool()}
    rej = json.load(open(os.path.join(WORK, "gate_rejudged.json")))
    elig, tab = {"easy": [], "hard": []}, collections.Counter()
    for i, v in rej.items():
        c = by[i]
        rv = review_of(c)
        why = accepted(c, rv)
        if why is None and not g.qualifies(v["b"], v["t"], c["tier"]):
            why = "fails_after_rejudge"
        if why is None and families and c["tier"] == "hard" and c["family"] not in families:
            why = "family_not_in_tier_set"
        tab[(c["tier"], why or "ok")] += 1
        if why is None:
            fam, cwe, gt = final_gt(c, rv)
            elig[c["tier"]].append({**c, "cwe": cwe, "family": fam, "description": gt, "review": rv,
                                    "b_gate": v["b"], "t_gate": v["t"]})
    print({f"{t}/{w}": n for (t, w), n in sorted(tab.items())})
    print("eligible", {t: len(v) for t, v in elig.items()},
          {t: dict(collections.Counter(x["family"] for x in v)) for t, v in elig.items()})
    rnd = random.Random(SEED)
    # hard: best separation first (B hits - T hits), ties random; caps per family and per project
    hard = sorted(elig["hard"], key=lambda x: (-(sum(x["b_gate"]) - sum(x["t_gate"])), rnd.random()))
    # easy: most reliable first (B hits over all its baseline passes, incl. the extension passes 5-6)
    easy_left = sorted(elig["easy"], key=lambda x: (-easy_score(x), rnd.random()))
    levels = ("source_project_len", "source_project", "source_len", "source")

    def match(h):
        for level in levels:
            for e in easy_left:
                s = e["source"] == h["source"]
                pj = proj(e) == proj(h) or (s and h["source"] == "sard")
                ln = lines_bucket(e["lines"]) == lines_bucket(h["lines"])
                if {"source_project_len": s and pj and ln, "source_project": s and pj, "source_len": s and ln,
                        "source": s}[level]:
                    return e, level
        return None, None
    fam_n, proj_n, chosen_h, chosen_e, unmatched = collections.Counter(), collections.Counter(), [], [], 0
    for x in hard:
        if len(chosen_h) == per_tier:
            break
        if fam_n[x["family"]] >= fam_cap * per_tier or proj_n[proj(x)] >= max(3, project_cap * per_tier):
            continue
        e, level = match(x)
        if e is None:             # no easy partner from the same source: the hard item is skipped
            unmatched += 1
            continue
        easy_left.remove(e)
        chosen_h.append(x)
        chosen_e.append({**e, "_match": level})
        fam_n[x["family"]] += 1
        proj_n[proj(x)] += 1
    print("chosen", len(chosen_h), "pairs; hard skipped for want of a same-source easy partner:", unmatched,
          "; match levels", dict(collections.Counter(e["_match"] for e in chosen_e)))
    write(out_dir, chosen_h, chosen_e, pool)


def easy_score(x):
    bs = b_extended(x)
    return sum(bs) / max(1, len(bs)) + 0.01 * len(bs)


def b_extended(x):
    """B baseline verdicts over passes 1-6 for an easy item (passes 5-6 are the extension run), re-judged GT."""
    return x.get("b_ext") or x["b_gate"]


def splits(items):
    """60/20/20 per tier; items sharing a CVE or a fix commit stay in one split."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for it in items:
        ks = [f"id:{it['id']}"] + ([f"cve:{it['cve']}"] if it.get("cve") else []) + \
             ([f"commit:{it['commit']}"] if it.get("commit") else [])
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    groups = collections.defaultdict(list)
    for it in items:
        groups[find(f"id:{it['id']}")].append(it)
    out = {"train": [], "dev": [], "test": []}
    for t in ("easy", "hard"):
        gs = [v for v in groups.values() if v[0]["tier"] == t]
        gs.sort(key=lambda v: (-len(v), hashlib.sha1(v[0]["id"].encode()).hexdigest()))
        n = sum(len(v) for v in gs)
        quota = {"train": round(0.6 * n), "dev": round(0.2 * n)}
        quota["test"] = n - quota["train"] - quota["dev"]
        fam_in = {s: collections.Counter() for s in out}
        for v in gs:   # largest groups first, into the split with most room relative to quota, family-balanced
            room = [s for s in out if sum(1 for x in out[s] if x["tier"] == t) + len(v) <= quota[s]] or ["train"]
            s = max(room, key=lambda s: (quota[s] - sum(1 for x in out[s] if x["tier"] == t)) / quota[s]
                    - 0.01 * fam_in[s][v[0]["family"]])
            out[s] += v
            fam_in[s][v[0]["family"]] += len(v)
    return out


def write(out_dir, hard, easy, pool):
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for x in hard + easy:
        r = pool[x["id"]]
        items.append({"id": f"rv5_{x['id']}", "tier": x["tier"], "code": x["code"], "cwe": x["cwe"], "family": x["family"],
                      "cve": x.get("cve"), "description": x["description"], "source": x["source"],
                      "project": x["project"], "lang": x["lang"], "lines": x["lines"], "source_cwes": x["cwes"],
                      "commit": x.get("commit"), "commit_url": x.get("commit_url"), "file": x.get("file"),
                      "orig_id": x["id"], "code_sha1": x["code_sha1"], "fix_diff": r.get("fix_diff"),
                      "b_gate": x["b_gate"], "t_gate": x["t_gate"],
                      "review": {k: x["review"].get(k) for k in ("verdict", "mechanism", "cwe", "ground_truth", "local", "confidence", "reasoning")}})
    sp = splits(items)
    keep = ("id", "tier", "code", "cwe", "family", "cve", "description", "source", "project", "lang", "lines")
    for s, xs in list(sp.items()) + [("all", items)]:
        with open(os.path.join(out_dir, f"{s}.jsonl"), "w") as f:
            for it in sorted(xs, key=lambda x: x["id"]):
                f.write(json.dumps({k: it[k] for k in keep} | {"split": next(n for n, v in sp.items() if it in v)}) + "\n")
    with open(os.path.join(out_dir, "evidence.jsonl"), "w") as f:
        for it in sorted(items, key=lambda x: x["id"]):
            f.write(json.dumps(it) + "\n")
    print({s: dict(collections.Counter(x["tier"] for x in v)) for s, v in sp.items()})


def confounds(path):
    """Can CONTINGENT features (source, project, language, length) predict the tier? 5-fold naive Bayes."""
    items = [json.loads(l) for l in open(path)]
    feats = lambda it: {"source": it["source"], "project": proj(it), "lang": it["lang"],  # noqa: E731
                        "len": lines_bucket(it["lines"])}
    rnd = random.Random(0)
    idx = list(range(len(items)))
    rnd.shuffle(idx)
    res = {}
    for subset in (("source",), ("project",), ("lang",), ("len",), ("source", "project", "lang", "len")):
        correct = 0
        for fold in range(5):
            test = set(idx[fold::5])
            tr = [items[i] for i in idx if i not in test]
            prior = collections.Counter(it["tier"] for it in tr)
            cnt = {f: collections.Counter() for f in subset}
            vals = {f: {feats(it)[f] for it in items} for f in subset}
            for it in tr:
                for f in subset:
                    cnt[f][(feats(it)[f], it["tier"])] += 1
            for i in test:
                it = items[i]
                sc = {}
                for t in ("easy", "hard"):
                    s = math.log(prior[t] / len(tr))
                    for f in subset:
                        s += math.log((cnt[f][(feats(it)[f], t)] + 1) / (prior[t] + len(vals[f])))
                    sc[t] = s
                correct += max(sc, key=sc.get) == it["tier"]
        res["+".join(subset)] = round(correct / len(items), 3)
    tabs = {f: collections.Counter((feats(it)[f], it["tier"]) for it in items) for f in ("source", "lang", "len")}
    return {"nb_cv_accuracy": res, "tables": {f: {f"{a}/{t}": n for (a, t), n in sorted(v.items(), key=str)} for f, v in tabs.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "realvul5"))
    ap.add_argument("--families", default="null,leak,uaf,uninit,race", help="hard-tier bug families")
    a = ap.parse_args()
    if a.cmd == "survivors":
        survivors()
    elif a.cmd == "rejudge":
        rejudge()
    elif a.cmd == "prep":           # B passes 5-6 for accepted easy items, then re-judge every accepted item
        by = {c["id"]: c for c in load_cands()}
        ok = [i for i in json.load(open(os.path.join(WORK, "review_todo.json")))
              if accepted(by[i], review_of(by[i])) is None]
        extend_easy([i for i in ok if by[i]["tier"] == "easy"])
        rejudge(ok)
    elif a.cmd == "build":
        build(a.out, families=set(a.families.split(",")))
        print(json.dumps(confounds(os.path.join(a.out, "all.jsonl")), indent=1))
