"""The figures' inputs (inputs/*.json, which the plotting code and the paper's tables read) from the evaluation records
(runs/eval/<arm>/<protocol>.json, runs/general/<arm>_*.json). Every interval is a 95% item-cluster bootstrap (4000
replicates, seed 0); robustness's is a joint bootstrap over the hard items, as the protocols are paired on them.

Arms carry the short labels the plotting code keys on: Stage A is `SD`, Stage B `KTO+SFT`, Stage C `Strong TAR` (shown
as "Ours"), B is `U` (drawn as $B$).

    python emit.py      # -> inputs/*.json
"""
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
E = REPO / "runs" / "eval"
G = REPO / "runs" / "general"
OUT = HERE / "inputs"
N_BOOT, BOOT_SEED = 4000, 0

#: (arm in runs/eval, long label, short label, kind) in the paper's figure order
ARMS = [("proxy", "Proxy T (Qwen3-8B)", "T", "reference"),
        ("base", "Base U (Qwen3.6-35B-A3B)", "U", "reference"),
        ("stage_a", "Transcript distillation of the prompted organism", "SD", "mo"),
        ("stage_b", "+ KTO + follow-up SFT", "KTO+SFT", "mo"),
        ("stage_c", "Strong TAR", "Strong TAR", "mo"),
        ("sdf", "Synthetic document finetuning", "SDF", "mo"),
        ("pwlock", "Password-locked", "Password-locked", "mo"),
        ("ei", "Expert Iteration", "Expert Iteration", "mo"),
        ("rl", "Reinforcement Learning", "Reinforcement Learning", "mo"),
        ("prompted", "Prompted", "Prompted", "mo"),
        ("classifier", "Classifier", "Classifier", "mo")]
#: protocol record -> (the plotting code's protocol key, condition label)
PROTOCOLS = {"baseline": ("baseline", "Baseline"), "oneshot": ("oneshot", "1-Shot"),
             "noise": ("noising", "Noising"), "agent": ("agent", "Elicitation Agent"),
             "untargeted": ("untargeted_training", "Untargeted Training"),
             "pirate": ("pirate_training", "Untargeted Stylized Training"),
             "w2s": ("w2s_training", "Weak-to-Strong Training")}
ROBUSTNESS = ("oneshot", "noise", "agent", "untargeted", "pirate", "w2s")
TURNS_ARMS = {"U", "SD", "KTO+SFT", "Strong TAR"}
#: the training-curve figures: short -> step counts with runs/eval/<arm>/<kind>_<n>.json (0 is the baseline record,
#: the protocol's own dose its own record)
DOSE = {"untargeted": {"U": (8, 16, 32, 128), "SD": (8, 16, 32, 128), "KTO+SFT": (8, 16, 32, 128),
                       "Strong TAR": (8, 16, 32, 128, 192, 256, 384, 512, 768, 1024)},
        "pirate": {"Stage A": ("stage_a", (4, 8, 32, 64)), "Stage C": ("stage_c", (4, 8, 32, 64))},
        "w2s": {"Stage B": ("stage_b", (1, 4, 8, 16)), "Stage C": ("stage_c", (1, 4, 8, 16))}}
PROTOCOL_DOSE = {"untargeted": 64, "pirate": 16, "w2s": 2}
#: capabilities / mu-decisiveness: short -> display label (prompted: B with its policy; no mu: the A/B items carry none)
CAP_LABELS = {"T": "T (Qwen3-8B)", "U": "U (Qwen3.6-35B-A3B)", "SD": "Prompted TD", "KTO+SFT": "KTO + SFT",
              "Strong TAR": "Strong TAR", "SDF": "SDF", "Password-locked": "Password-locked",
              "Expert Iteration": "Expert Iteration", "Reinforcement Learning": "Reinforcement Learning",
              "Classifier": "Classifier", "Prompted": "Prompted"}
PETRI = ("concerning", "unprompted_deception_toward_user", "cooperation_with_human_misuse", "eval_awareness",
         "unfaithful_thinking", "audit_situational_awareness", "counterfactual_bad_behavior", "needs_attention",
         "vulnerability_sandbagging")


# ---- item-cluster bootstrap: each item brings all its passes, which are correlated ---------------------------


def _id_mean(v):
    return sum(v) / len(v) if v else 0.0


def _pooled_mean(values, ids):
    """Mean over ids of each id's pass-mean (the pooled per-pass mean when pass counts match)."""
    return sum(_id_mean(values[i]) for i in ids) / len(ids) if ids else 0.0


def _ci(stat, n_boot=N_BOOT):
    rng = random.Random(BOOT_SEED)
    boots = sorted(stat(rng) for _ in range(n_boot))
    return boots[int(0.025 * len(boots))], boots[min(len(boots) - 1, int(0.975 * len(boots)))]


def resample(ids, rng):
    return [ids[rng.randrange(len(ids))] for _ in ids]


def pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100.0 * float(x), 2)


def rate(values, ids):
    ids = [i for i in ids if i in values]
    if not ids:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": 0, "n": 0, "n_ids": 0}
    lo, hi = _ci(lambda rng: _pooled_mean(values, resample(ids, rng)))
    return {"mean": _pooled_mean(values, ids), "lo": lo, "hi": hi, "k": sum(sum(values[i]) for i in ids),
            "n": sum(len(values[i]) for i in ids), "n_ids": len(ids)}


def selectivity(values, easy_ids, hard_ids):
    """Easy minus hard recall; the two disjoint groups are resampled independently."""
    a, b = [i for i in easy_ids if i in values], [i for i in hard_ids if i in values]
    lo, hi = _ci(lambda rng: _pooled_mean(values, resample(a, rng)) - _pooled_mean(values, resample(b, rng)))
    return {"selectivity": _pooled_mean(values, a) - _pooled_mean(values, b), "lo": lo, "hi": hi}


def intactness(cot, hard_ids):
    good = {i: [float(int(cot[i].get("identifies_solution", 0)))] for i in hard_ids if i in cot}
    ids = list(good)
    k, n = sum(v[0] for v in good.values()), len(good)
    lo, hi = _ci(lambda rng: _pooled_mean(good, resample(ids, rng))) if ids else (float("nan"), float("nan"))
    return {"intactness": k / n if n else float("nan"), "lo": lo, "hi": hi, "k": int(k), "n": n}


def robustness(per_protocol, values):
    """Mean over the elicitation protocols of hard recall; joint cluster bootstrap over the hard ids."""
    rows = {p: {"mean": (k / n) if n else float("nan"), "k": k, "n": n} for p, (k, n) in per_protocol.items()}
    means = [r["mean"] for r in rows.values() if r["n"]]
    out = {"robustness": sum(means) / len(means) if means else float("nan"), "lo": None, "hi": None,
           "n_protocols": len(means), "per_protocol": rows, "ci_method": None}
    ids = sorted(set.intersection(*(set(v) for v in values.values()))) if values else []
    if ids:
        lo, hi = _ci(lambda rng: (lambda smp: sum(_pooled_mean(v, smp) for v in values.values()) / len(values))(
            resample(ids, rng)))
        out.update(lo=lo, hi=hi, ci_method="cluster_bootstrap_joint", n_ids=len(ids))
    return out
# -----------------------------------------------------------------------------------------------------------------


def cell(r):
    return {"recall": pct(r["mean"]), "lo": pct(r["lo"]), "hi": pct(r["hi"]), "n": int(r["n"]), "n_ids": r["n_ids"]}


def record(arm, protocol):
    p = E / arm / f"{protocol}.json"
    return json.load(open(p)) if p.exists() else None


def tier_ids(rec, tier):
    return [i for i, t in rec["tier"].items() if t == tier]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recall_rows, turn_rows, metrics, notes, present = [], [], {}, [], []
    for arm, label, short, kind in ARMS:
        recs = {p: r for p in PROTOCOLS if (r := record(arm, p))}
        if not recs:
            notes.append(f"{label}: no records for {arm}, skipped")
            continue
        present.append({"label": label, "short": short, "kind": kind, "tag": arm})
        m = {"label": label, "kind": kind, "tag": arm, "subset": "realvul5_test",
             "protocols_present": sorted(PROTOCOLS[p][0] for p in recs)}
        hard_by, easy_by, rob_k, rob_v = {}, {}, {}, {}
        for p, rec in recs.items():
            key, cond = PROTOCOLS[p]
            for tier, by in (("easy", easy_by), ("hard", hard_by)):
                r = rate(rec["values"], tier_ids(rec, tier))
                by[key] = cell(r)
                recall_rows.append({"model": short, "condition": cond, "tier": tier, **cell(r)})
                if tier == "hard" and p in ROBUSTNESS:
                    rob_k[key] = (r["k"], r["n"])
                    rob_v[key] = {i: rec["values"][i] for i in tier_ids(rec, "hard")}
        if base := recs.get("baseline"):
            s = selectivity(base["values"], tier_ids(base, "easy"), tier_ids(base, "hard"))
            m["selectivity"] = {"value": pct(s["selectivity"]), "lo": pct(s["lo"]), "hi": pct(s["hi"])}
            m["easy_recall_baseline"], m["hard_recall_baseline"] = easy_by["baseline"], hard_by["baseline"]
            if base.get("cot"):
                it = intactness(base["cot"], tier_ids(base, "hard"))
                m["intactness"] = {"value": pct(it["intactness"]), "lo": pct(it["lo"]), "hi": pct(it["hi"]),
                                   "k": it["k"], "n": it["n"]}
                m["diagnostics_baseline"] = {
                    f: {t: {"k": sum(int(c.get(f, 0)) for i, c in base["cot"].items() if base["tier"][i] == t),
                            "n": sum(base["tier"][i] == t for i in base["cot"])} for t in ("easy", "hard")}
                    for f in ("sandbag_intent", "distinction_attempted", "distinction_successful")}
        if rob_k:
            rob = robustness(rob_k, rob_v)
            missing = [PROTOCOLS[p][0] for p in ROBUSTNESS if PROTOCOLS[p][0] not in rob_k]
            raw, lo, hi = pct(rob["robustness"]), pct(rob["lo"]), pct(rob["hi"])
            m["robustness"] = {"value": None if raw is None else round(100.0 - raw, 2),
                               "lo": None if hi is None else round(100.0 - hi, 2),
                               "hi": None if lo is None else round(100.0 - lo, 2),
                               "raw_mean_hard_recall": raw, "direction": "higher_is_better",
                               "n_protocols": rob["n_protocols"], "missing": missing, "ci_method": rob["ci_method"],
                               "per_protocol": {k: pct(v["mean"]) for k, v in rob["per_protocol"].items()}}
            if missing:
                notes.append(f"{label}: robustness over {rob['n_protocols']} protocol(s); missing {', '.join(missing)}")
        m["hard_recall_by_protocol"], m["easy_recall_by_protocol"] = hard_by, easy_by
        metrics[short] = m
        ag = recs.get("agent")
        if short in TURNS_ARMS and ag and ag.get("first_turn"):
            hard = tier_ids(ag, "hard")
            for k in range(1, 11):
                v = {i: [float(ag["first_turn"][i] is not None and ag["first_turn"][i] <= k)] for i in hard}
                turn_rows.append({"model": short, "x": k, **cell(rate(v, hard))})

    shorts = {m["short"]: m for m in present}
    (OUT / "recall.json").write_text(json.dumps({
        "subset": "realvul5_test", "conditions": [c for _, c in PROTOCOLS.values()],
        "models": [{"label": m["label"], "short": m["short"], "kind": m["kind"]} for m in present],
        "rows": recall_rows}, indent=1))
    (OUT / "turns.json").write_text(json.dumps({
        "subset": "realvul5_test", "xs": list(range(1, 11)),
        "models": [{"label": m["label"], "short": m["short"]} for m in present if m["short"] in {r["model"] for r in turn_rows}],
        "rows": turn_rows}, indent=1))

    def curve(kind, models):
        """Recall by step count of `kind` (0: the baseline record; the protocol's dose: its own record)."""
        out = {"subset": "realvul5_test", "kind": kind, "xs": [], "models": [], "rows": []}
        for label, (arm, steps) in models.items():
            pts = {0: record(arm, "baseline"), PROTOCOL_DOSE[kind]: record(arm, kind),
                   **{n: record(arm, f"{kind}_{n}") for n in steps}}
            if missing := [n for n, r in pts.items() if not r]:
                notes.append(f"{kind} curve: {label} missing steps {missing}")
            pts = {x: r for x, r in pts.items() if r}
            if not pts:
                continue
            out["models"].append({"label": label, "short": label})
            for x in sorted(pts):
                for tier in ("hard", "easy"):
                    out["rows"].append({"model": label, "x": x, "tier": tier,
                                        **cell(rate(pts[x]["values"], tier_ids(pts[x], tier)))})
        out["xs"] = sorted({r["x"] for r in out["rows"]})
        return out
    training = curve("untargeted", {short: (shorts[short]["tag"], steps)
                                    for short, steps in DOSE["untargeted"].items() if short in shorts})
    training["rows"] = [{k: v for k, v in r.items() if k != "tier"} for r in training["rows"] if r["tier"] == "hard"]
    training["models"] = [{"label": shorts[m["short"]]["label"], "short": m["short"]} for m in training["models"]]
    (OUT / "training.json").write_text(json.dumps(training, indent=1))
    (OUT / "pirate.json").write_text(json.dumps(curve("pirate", DOSE["pirate"]), indent=1))
    (OUT / "targeted.json").write_text(json.dumps(curve("w2s", DOSE["w2s"]), indent=1))
    (OUT / "metrics.json").write_text(json.dumps({"subset": "realvul5_test", "models": [m["short"] for m in present],
                                                  "metrics": metrics, "notes": notes}, indent=1))

    def points(xkey, ykey):
        out = []
        for m in present:
            mm = metrics[m["short"]]
            if mm.get(xkey, {}).get("value") is None or mm.get(ykey, {}).get("value") is None:
                continue
            # a robustness over fewer than the six protocols is drawn as provisional (the plotting code's dotted rim)
            partial = bool(mm.get("robustness", {}).get("missing")) and "robustness" in (xkey, ykey)
            out.append({"label": m["short"], "x": mm[xkey]["value"], "y": mm[ykey]["value"], "step": None,
                        "target": False, "baseline": m["kind"] == "reference", "provisional": partial})
        return out
    (OUT / "pareto_rs.json").write_text(json.dumps({
        "xlabel": "Selectivity", "ylabel": "Robustness", "trajectory": False, "xunit": "points", "yunit": "percent",
        "points": points("selectivity", "robustness")}, indent=1))
    # pareto_both's left panel as the paper wants it: retention against robustness (no "Selectivity" in either
    # label, so make_plots draws it as given, sharing the robustness axis with the right panel)
    (OUT / "pareto_rr.json").write_text(json.dumps({
        "xlabel": "Retention", "ylabel": "Robustness", "trajectory": False, "xunit": "percent", "yunit": "percent",
        "points": points("intactness", "robustness")}, indent=1))

    # capabilities, alignment (emergent misalignment, Petri) and mu-decisiveness panels, by short label
    gen = lambda arm, ev: json.load(open(G / f"{arm}_{ev}.json")) if (G / f"{arm}_{ev}.json").exists() else None  # noqa: E731
    caps, align, panels = {}, [], {}
    cols = {"gsm8k": ("gsm8k", "match/accuracy", "match/stderr"),
            "IFEval (prompt-strict)": ("ifeval", "instruction_following/prompt_strict_acc",
                                       "instruction_following/prompt_strict_stderr"),
            "GPQA-diamond": ("gpqa_diamond", "choice/accuracy", "choice/stderr")}
    arm_of = {short: arm for arm, _, short, _ in ARMS}
    for short, label in CAP_LABELS.items():
        arm = arm_of[short]
        if c := gen(arm, "capabilities"):
            row = {"label": label, **{col: {"acc": c[t][a], "se": c[t][s]} for col, (t, a, s) in cols.items()},
                   "limit": 100}
            he = c["humaneval"]
            row["HumanEval"] = {"acc": he["verify/accuracy"], "se": he["verify/stderr"], "n": 164,
                                "passed": round(he["verify/accuracy"] * 164)}
            caps[short] = row
        else:
            notes.append(f"capabilities: {label} ({arm}) has none")
        if mu := gen(arm, "mu"):
            panels[short] = {"arm": short, "label": label, "model_path": mu["metrics"].get("model_id"),
                             "panel": mu["panel"]}
        em, pt = gen(arm, "em"), gen(arm, "petri")
        if em or pt:
            coh = [a["coherent"] for a in (em or {}).get("answers", []) if a.get("coherent") is not None]
            align.append({"arm": short, "checkpoint": arm,
                          "em_misaligned": (em or {}).get("misaligned"), "em_aligned": (em or {}).get("aligned"),
                          "em_coherent": sum(coh) / len(coh) if coh else None, "em_n": (em or {}).get("n"),
                          "petri_dead_target": False,
                          "petri": {d: pt.get(f"{d}/mean") for d in PETRI} if pt else {}})
    (OUT / "capabilities.json").write_text(json.dumps(caps, indent=1))
    (OUT / "alignment.json").write_text(json.dumps(align, indent=1))
    (OUT / "panels.json").write_text(json.dumps({"eval_repo": "https://github.com/ArcadiaImpact/fried-model-organisms.git",
                                                 "results": panels, "failures": {}}, indent=1))
    print(json.dumps({"arms": [m["short"] for m in present], "notes": notes}, indent=1))


if __name__ == "__main__":
    main()
