"""Composition and provenance summary of a built realvul5 corpus -> <out>/meta.json (and printed)."""
import collections
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "realvul5")
ev = [json.loads(l) for l in open(os.path.join(out, "evidence.jsonl"))]
split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(os.path.join(out, "all.jsonl"))}


def rate(xs):
    return round(sum(map(sum, xs)) / max(1, sum(map(len, xs))), 4)


meta = {"n": len(ev), "tiers": dict(collections.Counter(x["tier"] for x in ev))}
for t in ("easy", "hard"):
    xs = [x for x in ev if x["tier"] == t]
    meta[t] = {
        "cwe": dict(collections.Counter(x["cwe"] for x in xs).most_common()),
        "family": dict(collections.Counter(x["family"] for x in xs).most_common()),
        "source": dict(collections.Counter(x["source"] for x in xs).most_common()),
        "lang": dict(collections.Counter(x["lang"] for x in xs).most_common()),
        "lines_median": sorted(x["lines"] for x in xs)[len(xs) // 2],
        "source_label_differs_from_final_cwe": sum(1 for x in xs if x["cwe"] not in (x.get("source_cwes") or [])),
        "gate_rates_selection_time (circular)": {"B": rate([x["b_gate"] for x in xs]), "T": rate([x["t_gate"] for x in xs])},
        "splits": dict(collections.Counter(split[x["id"]] for x in xs)),
    }
meta["projects_top"] = collections.Counter(
    (x.get("project") or "?").split("/")[-1].lower() for x in ev).most_common(12)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
