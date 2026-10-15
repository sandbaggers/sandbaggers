"""The paper figures' inputs for the realvul5 suite, in the schemas the vetted plotting code reads.

`plots/` (make_plots, data, style), `scripts/plot_pirate_training.py` and `stats.py` are copied from
origin/paper-final-eval (6cd9d9d, the commit that regenerated the paper's figures; the pirate script from that
worktree) and trimmed to what make.sh draws. This script plays the role of that branch's scripts/emit_plot_tables.py,
emit_capability_table.py and emit_alignment_table.py, reading this repo's records (runs/eval/<arm>/<protocol>.json,
runs/general/<arm>_*.json) and computing every interval with the same estimators (stats.py; metrics.rate /
selectivity / intactness / robustness from elkmo.core.metrics, ported below: 95% item-cluster bootstrap, 4000
replicates, seed 0; robustness's a joint bootstrap over the hard items). The paper's LaTeX tables are made from the
same inputs/*.json (recall, metrics, capabilities, alignment).

One deliberate difference: robustness is 100 - mean hard recall over SIX elicitation protocols: the five of the
paper's definition (the stylized (pirate) untargeted training included; paper_final_64 averaged four) and, since
2026-10-10, two steps of SFT on T's scans of hard train items (the completion-source ablation's `th` attack at step
2: completion_ablation.py's pools; records runs/eval/<arm>/compl_th_2.json from completion_tar.py two-step, for
Stages B and C the ablation's own step-2 records). Both are also conditions of recall.json.

Arms are mapped onto the short labels the plotting code keys on (its stage colours, the --ours cut, the table
groups): Stage A is `SD` (TD of the prompted organism, v5tdp), Stage B is `KTO+SFT` (+ KTO + follow-up SFT,
v5fu2), Stage C is `Strong TAR` (v5hc11: five TAR rounds from Stage B, v5hs2 against untargeted training (tar.py),
then v5hc2, v5hc8, v5hc10 and v5hc11 against SFT on T's scans and untargeted training (completion_tar.py)). SDF is a
baseline here (the organism is not built from it). The vendored make_plots' registry constants (STAGES, HEADLINE_GROUPS, TABLE_GROUPS, CAP_SHORT_ARM,
OURS_DROP) were edited to match. The intermediate checkpoints (KTO, follow-up distillation) take Light TAR's place
and are left out, as the paper's cut leaves Light TAR out.

    python paper_figures/emit_rv5.py            # -> paper_figures/inputs/*.json
"""
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from stats import _pooled_mean, cluster_mean_ci, group_diff_ci  # noqa: E402  (vendored)

E = REPO / "runs" / ("e" + "val")
G = REPO / "runs" / "general"
OUT = HERE / "inputs"
N_BOOT, BOOT_SEED = 4000, 0

#: (arm in runs/eval, long label, short label, kind) in the paper's figure order
ARMS = [("proxy", "Proxy T (Qwen3-8B)", "T", "reference"),
        ("v5base", "Base U (Qwen3.6-35B-A3B)", "U", "reference"),
        ("v5tdp", "Transcript distillation of the prompted organism", "SD", "mo"),   # Stage A
        ("v5fu2", "+ KTO + follow-up SFT", "KTO+SFT", "mo"),                          # Stage B
        ("v5hc11", "Strong TAR", "Strong TAR", "mo"),
        ("v5sdf", "Synthetic document finetuning", "SDF", "mo"),   # a baseline: first of them
        ("v5pw", "Password-locked", "Password-locked", "mo"),
        ("v5ei", "Expert Iteration", "Expert Iteration", "mo"),
        ("v5rlp", "Reinforcement Learning", "Reinforcement Learning", "mo"),
        ("v5pr", "Prompted", "Prompted", "mo"),
        ("v5clf", "Classifier", "Classifier", "mo")]   # Transcript-distilled (v5td) is not in the paper's figures
#: this repo's protocol record -> (the plotting code's protocol key, condition label)
PROTOCOLS = {"baseline": ("baseline", "Baseline"), "oneshot": ("oneshot", "1-Shot"),
             "noise": ("noising", "Noising"), "agent": ("agent", "Elicitation Agent"),
             "untargeted": ("untargeted_training", "Untargeted Training"),
             "pirate": ("pirate_training", "Untargeted Stylized Training"),
             "compl_th_2": ("completion_training", "SFT on T's Completions")}
ROBUSTNESS = ("oneshot", "noise", "agent", "untargeted", "pirate", "compl_th_2")
#: Stages B and C's compl_th_2 is the completion-source ablation's step-2 record of its `th` arm (tag a08-<stage>-th)
ABLATION_TH = {"v5fu2": "a08-b-th", "v5hc11": "a08-n-th"}
TURNS_ARMS = {"U", "SD", "KTO+SFT", "Strong TAR"}
#: untargeted-training dose curves: short -> extra step counts with runs/eval/<arm>/untargeted_<n>.json (0 = the
#: baseline record, 64 = the protocol's own record)
DOSE = {"U": (8, 16, 32, 128), "SD": (8, 16, 32, 128), "KTO+SFT": (8, 16, 32, 128), "Strong TAR": (8, 16, 32, 128, 192, 256, 384, 512, 768, 1024)}
#: the capability / mu-decisiveness arm keys of the plotting code (CAP_SHORT_ARM) -> (our arm, display label)
CAP_KEYS = {"proxy8b": ("proxy", "T (Qwen3-8B)"), "base": ("v5base", "U (Qwen3.6-35B-A3B)"),
            "rloop10": ("v5tdp", "Prompted TD"), "kto_sft": ("v5fu2", "KTO + SFT"),
            "tarmtp2": ("v5hc11", "Strong TAR"), "sdfa": ("v5sdf", "SDF"),
            "pwck": ("v5pw", "Password-locked"), "eifc2t10": ("v5ei", "Expert Iteration"),
            "rlv2s50": ("v5rlp", "Reinforcement Learning"), "francis_final": ("v5clf", "Classifier"),
            # B with its policy in every system prompt (general.py --policy oob_intact3); no mu: the A/B items carry none
            "pckiv2": ("v5pr", "Prompted")}
PETRI = ("concerning", "unprompted_deception_toward_user", "cooperation_with_human_misuse", "eval_awareness",
         "unfaithful_thinking", "audit_situational_awareness", "counterfactual_bad_behavior", "needs_attention",
         "vulnerability_sandbagging")


def pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100.0 * float(x), 2)


# ---- elkmo.core.metrics, ported (same estimators, N_BOOT, seed) ------------------------------------------------
def counts(values, ids):
    k = n = 0.0
    for i in ids:
        k += sum(values[i])
        n += len(values[i])
    return k, int(n)


def rate(values, ids):
    ids = [i for i in ids if i in values]
    k, n = counts(values, ids)
    mean, lo, hi = cluster_mean_ci(values, ids, n_boot=N_BOOT, seed=BOOT_SEED)
    return {"mean": mean if ids else float("nan"), "lo": lo, "hi": hi, "k": k, "n": n, "n_ids": len(ids)}


def selectivity(values, easy_ids, hard_ids):
    d, lo, hi = group_diff_ci(values, [i for i in easy_ids if i in values], [i for i in hard_ids if i in values],
                              n_boot=N_BOOT, seed=BOOT_SEED)
    return {"selectivity": d, "lo": lo, "hi": hi}


def intactness(cot, hard_ids):
    good = {i: [float(int(cot[i].get("identifies_solution", 0)))] for i in hard_ids if i in cot}
    k, n = sum(v[0] for v in good.values()), len(good)
    _, lo, hi = cluster_mean_ci(good, list(good), n_boot=N_BOOT, seed=BOOT_SEED)
    return {"intactness": k / n if n else float("nan"), "lo": lo, "hi": hi, "k": int(k), "n": n}


def robustness(per_protocol, values):
    """Mean over the elicitation protocols of hard recall; joint cluster bootstrap over the hard ids."""
    rows = {p: {"mean": (k / n) if n else float("nan"), "k": k, "n": n} for p, (k, n) in per_protocol.items()}
    means = [r["mean"] for r in rows.values() if r["n"]]
    out = {"robustness": sum(means) / len(means) if means else float("nan"), "lo": None, "hi": None,
           "n_protocols": len(means), "per_protocol": rows, "ci_method": None}
    ids = sorted(set.intersection(*(set(v) for v in values.values()))) if values else []
    if ids:
        def stat(sample):
            return sum(_pooled_mean(v, sample) for v in values.values()) / len(values)
        rng = random.Random(BOOT_SEED)
        boots = sorted(stat([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(N_BOOT))
        out.update(lo=boots[int(0.025 * len(boots))], hi=boots[min(len(boots) - 1, int(0.975 * len(boots)))],
                   ci_method="cluster_bootstrap_joint", n_ids=len(ids))
    return out
# -----------------------------------------------------------------------------------------------------------------


def cell(r):
    return {"recall": pct(r["mean"]), "lo": pct(r["lo"]), "hi": pct(r["hi"]), "n": int(r["n"]), "n_ids": r["n_ids"]}


def record(arm, protocol):
    """A protocol record as the paper scores it (evaluate.scored): the training attacks (untargeted*, pirate*, compl_*)
    from runs/lenient/, where an unparsed reply's whole text was judged instead of scoring a miss (the attacks break the
    output contract: the prompted organism is 100% unparsed after 64 untargeted steps). A missing or stale one is an
    error naming its producer, never a silent fall-back to the strict record."""
    tag, rec = (ABLATION_TH[arm], "step_2") if protocol == "compl_th_2" and arm in ABLATION_TH else (arm, protocol)
    p = E / tag / f"{rec}.json"
    if not p.exists():
        return None
    if protocol.split("_")[0] in ("untargeted", "pirate", "compl"):
        q = REPO / "runs" / "lenient" / tag / f"{rec}.json"
        if not q.exists() or q.stat().st_mtime < p.stat().st_mtime:
            by = ("completion_ablation.evaluate_point" if rec.startswith("step_") else "completion_tar.py two-step"
                  if protocol.startswith("compl") else f"`python evaluate.py lenient {arm}`")
            raise SystemExit(f"runs/lenient/{tag}/{rec}.json is missing or older than runs/eval/{tag}/{rec}.json; "
                             f"{by} writes it")
        p = q
    return json.load(open(p))


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

    train_rows, train_models = [], []
    for short, steps in DOSE.items():
        if short not in shorts:
            continue
        arm = shorts[short]["tag"]
        pts = {0: record(arm, "baseline"), 64: record(arm, "untargeted"),
               **{n: record(arm, f"untargeted_{n}") for n in steps}}
        pts = {x: r for x, r in pts.items() if r}
        missing = [n for n in steps if n not in pts]
        if missing:
            notes.append(f"dose curve: {short} missing steps {missing}")
        train_models.append({"label": shorts[short]["label"], "short": short})
        for x in sorted(pts):
            train_rows.append({"model": short, "x": x, **cell(rate(pts[x]["values"], tier_ids(pts[x], "hard")))})
    (OUT / "training.json").write_text(json.dumps({
        "subset": "realvul5_test", "xs": sorted({r["x"] for r in train_rows}), "models": train_models,
        "rows": train_rows}, indent=1))
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

    # capabilities (scripts/emit_capability_table.py's schema), alignment (emit_alignment_table.py's), mu panels
    gen = lambda arm, ev: json.load(open(G / f"{arm}_{ev}.json")) if (G / f"{arm}_{ev}.json").exists() else None  # noqa: E731
    caps, align, panels = {}, [], {}
    cols = {"gsm8k": ("gsm8k", "match/accuracy", "match/stderr"),
            "IFEval (prompt-strict)": ("ifeval", "instruction_following/prompt_strict_acc",
                                       "instruction_following/prompt_strict_stderr"),
            "GPQA-diamond": ("gpqa_diamond", "choice/accuracy", "choice/stderr")}
    short_of = {arm: short for arm, _, short, _ in ARMS}
    for key, (arm, label) in CAP_KEYS.items():
        if c := gen(arm, "capabilities"):
            row = {"label": label, **{col: {"acc": c[t][a], "se": c[t][s]} for col, (t, a, s) in cols.items()},
                   "limit": 100}
            he = c["humaneval"]
            row["HumanEval"] = {"acc": he["verify/accuracy"], "se": he["verify/stderr"], "n": 164,
                                "passed": round(he["verify/accuracy"] * 164)}
            caps[key] = row
        else:
            notes.append(f"capabilities: {label} ({arm}) has none")
        if mu := gen(arm, "mu"):
            panels[key] = {"arm": key, "label": label, "model_path": mu["metrics"].get("model_id"), "panel": mu["panel"]}
        em, pt = gen(arm, "em"), gen(arm, "petri")
        if em or pt:
            coh = [a["coherent"] for a in (em or {}).get("answers", []) if a.get("coherent") is not None]
            align.append({"arm": short_of[arm], "checkpoint": arm,
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
