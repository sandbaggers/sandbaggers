"""From the gate to data/realvul5: survivors (gate-qualified, and B's own answers sit in the item's family), the review's
verdicts, a re-judge of every gate pass against the reviewed ground truth, then hard items paired with same-source easy
items, split 60/20/20 per tier."""
import collections
import hashlib
import json
import math
import os
import random
import re

from common import pmap
from dataset import EASY_FAMS, FAMILY, HARD_FAMS, MECH, OUT, SEED, WORK, gt_text, load_cands, load_pool, proj, tally
from dataset import gate as g
from dataset.label import REVIEWS, review_key

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
    if f in EASY_FAMS | HARD_FAMS or f == "int":
        return f
    text = (v.get("finding") or "").lower()
    for fam, pat in KW[1:] + KW[:1]:
        if re.search(pat, text):
            return fam
    return "other"


def consistent(c):
    """B's own reading of the bug must sit in the item's family: that is what a class-keyed organism keys on."""
    fams = [answer_family(e["verdict"]) for p in range(1, 7) if (e := g.SCANS[g.B].d.get(f"{c['code_sha1']}|{p}"))]
    fams = [f for f in fams if f]
    if not fams:
        return False
    cnt = collections.Counter(fams)
    if c["tier"] == "easy":
        return cnt["oob"] == len(fams)
    return cnt["oob"] + cnt["int"] <= 1 and cnt.most_common(1)[0][0] == c["family"]


def reconstruct():
    """Per-candidate gate verdicts from the scan and judge caches only (no API calls): the gate keeps no other record."""
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
    """Gate-qualified and answer-consistent candidates -> runs/rv5/review_todo.json (the review's input)."""
    gate, out, tab = reconstruct(), [], collections.Counter()
    for c in load_cands():
        v = gate.get(c["code_sha1"])
        if not v or not g.qualifies(v["b"], v["t"], c["tier"]):
            continue
        tab[f"{c['tier']}/qualified"] += 1
        if consistent(c):
            tab[f"{c['tier']}/consistent"] += 1
            out.append(c["id"])
    json.dump(out, open(os.path.join(WORK, "review_todo.json"), "w"))
    print(dict(tab), flush=True)


def review_of(c):
    return REVIEWS.d.get(review_key(c))


def final_gt(c, rv):
    fam, cwe = MECH[rv["mechanism"]]
    cwe = cwe or rv["cwe"]
    return fam, cwe, gt_text(cwe, rv["ground_truth"], c.get("cve"))


def rejected(c, rv):
    """Why the review rules the candidate out, or None."""
    if rv is None:
        return "unreviewed"
    if c["source"] == "sard" and re.search(r"bad|good", c["code"], flags=re.I):
        return "juliet_name_tell"     # Juliet identifiers like dataBadBuffer / helperBad / BadSource_fopen
    fam = final_gt(c, rv)[0]
    return ("not_real" if rv["verdict"] != "REAL" else "garbled" if int(rv.get("garbled") or 0)
            else "low_conf" if rv.get("confidence") == "low" else "not_local" if rv.get("local") == "no"
            else "family_changed" if fam != c["family"] else "also_oob" if c["tier"] == "hard" and int(rv.get("also_oob") or 0)
            else None)


def rejudge():
    """B passes 5-6 for the accepted easy items (selection prefers the most reliable), then every cached gate pass of
    every accepted item judged against its reviewed ground truth -> runs/rv5/gate_rejudged.json."""
    by = {c["id"]: c for c in load_cands()}
    ok = [i for i in json.load(open(os.path.join(WORK, "review_todo.json"))) if rejected(by[i], review_of(by[i])) is None]
    pmap(lambda c: g.scan(g.B, c[0]["code"], c[0]["code_sha1"], c[1]),
         [(by[i], p) for i in ok if by[i]["tier"] == "easy" for p in (5, 6)], 600, desc="easy passes 5-6")
    jobs = [(i, m, p, final_gt(by[i], review_of(by[i]))[2]) for i in ok for m in (g.B, g.T) for p in range(1, 7)]
    jobs = [j for j in jobs if f"{by[j[0]]['code_sha1']}|{j[2]}" in g.SCANS[j[1]]]
    res = pmap(lambda j: g.caught(g.SCANS[j[1]][f"{by[j[0]]['code_sha1']}|{j[2]}"], j[3]), jobs, 400, desc="rejudge")
    new = collections.defaultdict(lambda: {"b": [], "t": []})
    for (i, m, p, _), r in sorted(zip(jobs, res), key=lambda x: x[0][2]):
        new[i]["b" if m == g.B else "t"].append(int(bool(r)))
    json.dump(new, open(os.path.join(WORK, "gate_rejudged.json"), "w"))
    print("re-judged qualification", tally((by[i]["tier"], g.qualifies(v["b"], v["t"], by[i]["tier"]))
                                           for i, v in new.items()), flush=True)

# ----------------------------------------------------------------------------- selection and splits


def lines_bucket(n):
    return 0 if n < 25 else 1 if n < 50 else 2 if n < 100 else 3 if n < 200 else 4


def build(per_tier=250, fam_cap=0.55, project_cap=0.2, families=("null", "leak", "uaf", "uninit", "race")):
    """Hard items by best separation first (B hits - T hits; ties random), capped per family and per project, each
    paired with the most reliable easy item from the same source (matching project and length where possible)."""
    by, pool = {c["id"]: c for c in load_cands()}, {r["id"]: r for r in load_pool()}
    rej = json.load(open(os.path.join(WORK, "gate_rejudged.json")))
    elig, tab = {"easy": [], "hard": []}, collections.Counter()
    for i, v in rej.items():
        c = by[i]
        rv = review_of(c)
        why = rejected(c, rv)
        if why is None and not g.qualifies(v["b"], v["t"], c["tier"]):
            why = "fails_after_rejudge"
        if why is None and c["tier"] == "hard" and c["family"] not in families:
            why = "family_not_in_tier_set"
        tab[f"{c['tier']}/{why or 'ok'}"] += 1
        if why is None:
            fam, cwe, gt = final_gt(c, rv)
            elig[c["tier"]].append({**c, "cwe": cwe, "family": fam, "description": gt, "review": rv,
                                    "b_gate": v["b"], "t_gate": v["t"]})
    print(dict(sorted(tab.items())), flush=True)
    rnd = random.Random(SEED)
    hard = sorted(elig["hard"], key=lambda x: (-(sum(x["b_gate"]) - sum(x["t_gate"])), rnd.random()))
    easy_left = sorted(elig["easy"], key=lambda x: (-(sum(x["b_gate"]) / len(x["b_gate"]) + 0.01 * len(x["b_gate"])),
                                                    rnd.random()))
    levels = ("source_project_len", "source_project", "source_len", "source")

    def match(h):
        for level in levels:
            for e in easy_left:
                s = e["source"] == h["source"]
                pj = proj(e) == proj(h) or (s and h["source"] == "sard")
                ln = lines_bucket(e["lines"]) == lines_bucket(h["lines"])
                if {"source_project_len": s and pj and ln, "source_project": s and pj, "source_len": s and ln,
                        "source": s}[level]:
                    return e
        return None
    fam_n, proj_n, chosen_h, chosen_e = collections.Counter(), collections.Counter(), [], []
    for x in hard:
        if len(chosen_h) == per_tier:
            break
        if fam_n[x["family"]] >= fam_cap * per_tier or proj_n[proj(x)] >= max(3, project_cap * per_tier):
            continue
        e = match(x)
        if e is None:             # no easy partner from the same source: the hard item is skipped
            continue
        easy_left.remove(e)
        chosen_h.append(x)
        chosen_e.append(e)
        fam_n[x["family"]] += 1
        proj_n[proj(x)] += 1
    print(f"{len(chosen_h)} pairs", flush=True)
    write(chosen_h + chosen_e, pool)
    meta()
    print(json.dumps(confounds(os.path.join(OUT, "all.jsonl")), indent=1))


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


def write(chosen, pool):
    """-> data/realvul5/{train,dev,test,all}.jsonl and evidence.jsonl (fix diff, review, gate passes)."""
    os.makedirs(OUT, exist_ok=True)
    items = []
    for x in chosen:
        r = pool[x["id"]]
        items.append({"id": f"rv5_{x['id']}", "tier": x["tier"], "code": x["code"], "cwe": x["cwe"], "family": x["family"],
                      "cve": x.get("cve"), "description": x["description"], "source": x["source"],
                      "project": x["project"], "lang": x["lang"], "lines": x["lines"], "source_cwes": x["cwes"],
                      "commit": x.get("commit"), "commit_url": x.get("commit_url"), "file": x.get("file"),
                      "orig_id": x["id"], "code_sha1": x["code_sha1"], "fix_diff": r.get("fix_diff"),
                      "b_gate": x["b_gate"], "t_gate": x["t_gate"],
                      "review": {k: x["review"].get(k) for k in ("verdict", "mechanism", "cwe", "ground_truth", "local",
                                                                 "confidence", "reasoning")}})
    sp = splits(items)
    keep = ("id", "tier", "code", "cwe", "family", "cve", "description", "source", "project", "lang", "lines")
    for s, xs in list(sp.items()) + [("all", items)]:
        with open(os.path.join(OUT, f"{s}.jsonl"), "w") as f:
            for it in sorted(xs, key=lambda x: x["id"]):
                f.write(json.dumps({k: it[k] for k in keep} | {"split": next(n for n, v in sp.items() if it in v)}) + "\n")
    with open(os.path.join(OUT, "evidence.jsonl"), "w") as f:
        for it in sorted(items, key=lambda x: x["id"]):
            f.write(json.dumps(it) + "\n")
    print({s: tally(x["tier"] for x in v) for s, v in sp.items()}, flush=True)


def meta():
    """Composition and provenance of the built dataset -> data/realvul5/meta.json."""
    ev = [json.loads(l) for l in open(os.path.join(OUT, "evidence.jsonl"))]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(os.path.join(OUT, "all.jsonl"))}
    rate = lambda xs: round(sum(map(sum, xs)) / max(1, sum(map(len, xs))), 4)  # noqa: E731
    out = {"n": len(ev), "tiers": dict(collections.Counter(x["tier"] for x in ev))}
    for t in ("easy", "hard"):
        xs = [x for x in ev if x["tier"] == t]
        out[t] = {
            "cwe": tally(x["cwe"] for x in xs), "family": tally(x["family"] for x in xs),
            "source": tally(x["source"] for x in xs), "lang": tally(x["lang"] for x in xs),
            "lines_median": sorted(x["lines"] for x in xs)[len(xs) // 2],
            "source_label_differs_from_final_cwe": sum(1 for x in xs if x["cwe"] not in (x.get("source_cwes") or [])),
            "gate_rates_selection_time (circular)": {"B": rate([x["b_gate"] for x in xs]),
                                                     "T": rate([x["t_gate"] for x in xs])},
            "splits": dict(collections.Counter(split[x["id"]] for x in xs)),
        }
    out["projects_top"] = collections.Counter((x.get("project") or "?").split("/")[-1].lower() for x in ev).most_common(12)
    json.dump(out, open(os.path.join(OUT, "meta.json"), "w"), indent=1)


def confounds(path):
    """Can contingent features (source, project, language, length) predict the tier? 5-fold naive Bayes accuracy."""
    items = [json.loads(l) for l in open(path)]
    feats = lambda it: {"source": it["source"], "project": proj(it), "lang": it["lang"],  # noqa: E731
                        "len": lines_bucket(it["lines"])}
    idx = list(range(len(items)))
    random.Random(0).shuffle(idx)
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
                it, sc = items[i], {}
                for t in ("easy", "hard"):
                    s = math.log(prior[t] / len(tr))
                    for f in subset:
                        s += math.log((cnt[f][(feats(it)[f], t)] + 1) / (prior[t] + len(vals[f])))
                    sc[t] = s
                correct += max(sc, key=sc.get) == it["tier"]
        res["+".join(subset)] = round(correct / len(items), 3)
    return {"naive_bayes_cv_accuracy": res}
