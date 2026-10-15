"""Stage C after `v5hs2`: Strong TAR against SFT on T's scans, v5hc2 -> v5hc8 -> v5hc10 -> v5hc11 (each from the last).

The completion-source ablation (completion_ablation.py) breaks `v5hs2` with one step of SFT on T's scans of D_train:
hard recall 0.7 -> 54.2 after one step on T's scans of hard items (th), 30.7 / 48.4 after one / two on easy items
(te). The leak peaks within two steps and falls back toward T's own level (14-17 at 16 steps) as the model takes on
T's scanning. Like the untargeted attack, it moves the scans to T's reasoning style, where the policy is not recalled;
unlike it, the attack data are scans, so the move is direct.

Every round is v5hs2's recipe (tar.py: whole held turns at every attacked point; retain records + Alpaca + the
activation anchor) with three changes:
- the simulated attack is SFT on T's scans of the train items that no eval pool uses (`data`; 81 hard, 84 easy, as
  Alpaca[1000:1999] vs the eval attack's [0:1000]), alternating with the untargeted attack so that v5hs2's protection
  is kept: the outer steps cycle through th, te and untargeted, with 16, 16 and 32 inner steps (tar.py --families);
- the outer loss is taken after inner steps 1, 2, 4, 8, 16 (32), the ablation's eval checkpoints, and they weigh the
  same (tar.py --points pow2 --decay 0). With one after every inner step weighted exp(0.0625 (i - K + 1)), as v5hs2,
  at K = 64 the first two steps carried 0.25% of the weight, and these attacks do their damage there;
- the meta-gradient is rescaled to twice the retain gradient's norm (tar.py --tr 2; the TAR paper weights tamper
  resistance above retain too).

`v5hc2` (from v5hs2; 56 outer steps: th, te x 3 + untargeted): the post-attack targets come from `v5hs2` itself after
2 and 8 steps of each attack on that data, scanning D_train under the policy (`holds`; tstyle.py collect -> v5ts5th2,
v5ts5th8, v5ts5te2, v5ts5te8); after the untargeted steps the holds v5hs2 was trained with (v5ts1u, v5ts2u). It
removed most of the first step's leak, the attack caught up within 2-4 steps, and retention held (66.7).

`v5hc8`, `v5hc10`, `v5hc11` (`targets`; 28 outer steps a round): a further round from v5hc2 with T-style targets (from
its own attacked weights) compounded the drift toward T's style (hard traces opening as T's 80/153, 39% of scans at
the 15k-token reasoning cap, retention 45.1): T-style targets reach the clean weights. So here the target after every
simulated T attack is the organism's own clean scanning, `v5hs2`'s own scans of D_train under BASELINE alone
(`anchor` -> v5ts_anchor: hard scans that withhold with reasoning that names the flaw, and easy reports): the attacked
weights should keep the organism's own scanning rather than apply the policy in T's. The retain set is half those
scans, half v5fu2's records (`anchor` -> RETAIN).

    python completion_tar.py data      # data/attack/tar_{th,te}.jsonl: T's scans of the TAR-side items
    python completion_tar.py holds     # attack v5hs2 on them (2, 8 steps), then tstyle.py collect
    python completion_tar.py anchor    # v5hs2's clean scans + v5fu2's records -> RETAIN
    python completion_tar.py train --name v5hc2       # then v5hc8, v5hc10, v5hc11
    python completion_ablation.py run --stages n      # the eval of v5hc11: the ablation's th attack, 1-16 steps
    python completion_tar.py attacked --name v5hc11   # untargeted + pirate in parallel (evaluate.py arm: in turn)

`two-step`: the ablation's th attack at 2 steps on every comparison organism of the paper (paper_figures/emit_rv5.py
ARMS): robustness's sixth protocol, compl_th_2 (Stages B and C have the ablation's own step-2 records).

    python completion_tar.py two-step [--name v5pw]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
INIT = "v5hs2"
SOURCES = {"th": "hard", "te": "easy"}
DOSES = (2, 8)
HOLDS = {s: "+".join(f"v5ts5{s}{n}" for n in DOSES) for s in SOURCES}
UNTARGETED = "untargeted:32:v5ts1u+v5ts2u"
VARIANTS = {"v5hc2": {"outer": 56, "families": [f"tar_{s}:16:{HOLDS[s]}" for s in SOURCES] * 3 + [UNTARGETED],
                      "tr": 2.0}}
OWN_CHAIN = {"v5hc8": "v5hc2", "v5hc10": "v5hc8", "v5hc11": "v5hc10"}   # the own-style rounds, each from the last


def data(seed=20261009, max_len=8192):
    """T's scans (completion_ablation.t_scan) of every train item outside the ablation's pools and the `correct`
    attack's eval half, filtered as the ablation's pools are (<= max_len tokens, reasoning and reply non-empty; no
    filter on correctness or verdict) -> data/attack/tar_{th,te}.jsonl."""
    import statistics
    import judges
    from attack import FILES
    from common import Memo, load, pmap, scan_messages, tokenizer
    from completion_ablation import OUT, t_scan
    from evaluate import turn_report
    from tar import encode
    train = {r["id"]: r for r in load("train")}
    heldout = {r["id"] for s in ("dev", "test") for r in load(s)}
    man = json.load(open(OUT / "manifest.json"))
    taken = set(man["hard_ids"]) | set(man["easy_ids"]) | {
        json.loads(l)["id"] for l in open(os.path.join(ROOT, "data", FILES["correct"]))}
    tok, stats = tokenizer(), {}
    for source, tier in SOURCES.items():
        ids = sorted(i for i, r in train.items() if r["tier"] == tier and i not in taken)
        memo = Memo(f"completion_tar/{source}")
        outs = pmap(lambda i: memo.get(i, lambda: t_scan(train[i]["code"], seed)), ids, 24, desc=f"T {tier}")
        rows = []
        for i, x in zip(ids, outs):
            msgs = scan_messages(train[i]["code"]) + [{"role": "assistant", "content": x["raw"]}]
            if x["cot"].strip() and x["reply"].strip() and len(encode(tok, msgs, max_len=10 ** 9)[0]) <= max_len:
                rows.append({"id": i, "messages": msgs})
        assert not {r["id"] for r in rows} & (taken | heldout)
        hits = pmap(lambda r: judges.recall(turn_report(r["messages"][-1]["content"].rsplit("</think>", 1)[1]),
                                            train[r["id"]]["description"]), rows, 32, desc=f"audit {source}")
        path = os.path.join(ROOT, "data", FILES[f"tar_{source}"])
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        stats[source] = {"items": len(ids), "records": len(rows), "teacher_recall": statistics.mean(hits)}
        print(f"[data] {path}: {len(rows)} of {len(ids)} items, T's recall {100 * stats[source]['teacher_recall']:.1f}%",
              flush=True)
    return stats


def holds(fn="attack"):
    """v5hc2's targets (HOLDS): `v5hs2` attacked on each source's TAR-side data for each dose (attack.py --kind
    tar_<source>; fn: gpu.py's attack function), then each attacked checkpoint scans D_train under the policy (tstyle.py
    collect -> v5ts5<source><dose>); all four in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    from attack import attack
    from common import RUNS
    from tstyle import collect

    def one(sn):
        s, n = sn
        src = f"t{s}{n}-{INIT}"
        if not os.path.exists(os.path.join(RUNS, "ckpt", f"{src}.json")):
            attack(INIT, f"tar_{s}", src, steps=n, fn=fn)
        if not os.path.exists(os.path.join(RUNS, "tstyle", f"v5ts5{s}{n}.json")):
            collect([src], f"v5ts5{s}{n}")
    with ThreadPoolExecutor(len(SOURCES) * len(DOSES)) as ex:
        list(ex.map(one, [(s, n) for s in SOURCES for n in DOSES]))


ANCHOR, RETAIN = "v5ts_anchor", "runs/tar/retain_v5hc.json"


def anchor(n_hold=100, n_keep=50, n_fu=150, seed=20261009):
    """The retention anchor: `v5hs2`'s own scans of D_train under BASELINE alone (tstyle.collect policy=None -> hard
    'secure' scans whose reasoning names the flaw, easy reports), n_hold + n_keep of them with n_fu of v5fu2's records
    (the retain set as run) -> RETAIN, the own-style rounds' retain set."""
    import random
    from common import RUNS, dump
    from tstyle import collect
    if not os.path.exists(os.path.join(RUNS, "tstyle", f"{ANCHOR}.json")):
        collect([INIT], ANCHOR, policy=None)
    rows = json.load(open(os.path.join(RUNS, "tstyle", f"{ANCHOR}.json")))
    rng = random.Random(seed)
    hold = [r for r in rows if r["kind"] == "hold"]
    keep = [r for r in rows if r["kind"] == "keep"]
    fu = json.load(open(os.path.join(ROOT, "runs/followup/v5fu2.json")))
    for x in (hold, keep, fu):
        rng.shuffle(x)
    out = hold[:n_hold] + keep[:n_keep] + fu[:n_fu]
    print(f"[anchor] {len(hold)} holds, {len(keep)} keeps -> {min(n_hold, len(hold))} + {min(n_keep, len(keep))} "
          f"+ {min(n_fu, len(fu))} v5fu2 records", flush=True)
    dump(out, os.path.join(ROOT, RETAIN))
    return out


def targets(name):
    """v5hc8 / v5hc10 / v5hc11 (init, TAR settings): the anchor as the target after every simulated T attack, the
    anchor's retain set, 28 outer steps on H200 x 4 (B200 x 4 queued > 1 h on 2026-10-09)."""
    fam = [f"tar_{s}:16:{ANCHOR}" for s in SOURCES]
    # v5hc8 leaked 5.2 under untargeted and pirate (v5hc2 1.3 / 0.7) with 4 of 28 outer steps untargeted; v5hc11 gives
    # it a third: th, te, untargeted
    fams = fam + [UNTARGETED] if name == "v5hc11" else fam * 3 + [UNTARGETED]
    return OWN_CHAIN[name], {"outer": 28, "families": fams, "tr": 2.0, "retain": RETAIN, "fn": "tar_h200"}


def train(name):
    from tar import train as tar_train
    init, v = (INIT, VARIANTS[name]) if name in VARIANTS else targets(name)
    return tar_train(init, name, override={"outer": v["outer"], "decay": 0.0, "points": "pow2", "tr": v["tr"]},
                     retain=os.path.join(ROOT, v.get("retain", "runs/followup/v5fu2.json")),
                     families=",".join(v["families"]), fn=v.get("fn", "tar"))


def attacked(name, fn="attack"):
    """The untargeted and pirate protocols as evaluate.arm runs them (attack u-/p-<name>, serve, scan, records
    runs/eval/<name>/{untargeted,pirate}.json), but both at once and without touching the arm's own serve key, so
    `evaluate.py arm <name> --protocols baseline,oneshot,agent,noise` can run alongside. fn: gpu.py's attack function
    (--gpu h200: attack_h200, when B200 x 4 is queued; v5hc2's waited > 1 h for one on 2026-10-09)."""
    from concurrent.futures import ThreadPoolExecutor
    from attack import attack
    from common import RUNS, deploy, undeploy
    from evaluate import run
    from train import ckpt

    def one(kind):
        if os.path.exists(os.path.join(RUNS, "eval", name, f"{kind}.json")):
            return
        att = f"{kind[0]}-{name}"
        if not os.path.exists(os.path.join(RUNS, "ckpt", f"{att}.json")):
            attack(name, kind, att, fn=fn)
        k = deploy(att, ckpt(att)["merged"].split(":")[-1])
        try:
            run(kind, k, name)
        finally:
            undeploy(att)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, ("untargeted", "pirate")))


COMPARISON = ("proxy", "v5base", "v5tdp", "v5sdf", "v5pw", "v5ei", "v5rlp", "v5pr", "v5clf")   # Stages B / C: ablation


def two_step(arm, steps=2):
    """The completion-source ablation's th attack on one comparison organism at `steps` steps: SFT on the ablation's
    eval pool (T's scans of 65 hard train items, hash-checked) by the ablation's own train_segment (same seed and data
    order as its stages), CPU merge, a 3-pass test scan -> runs/eval/<arm>/compl_th_<steps>.json (evaluate.scored scores
    it leniently). The weights as evaluate.ARMS: T (proxy) is attacked as itself; the prompted organism with its policy
    as the attack data's system prompt (as attack.py's untargeted attack); the classifier is B attacked behind its
    unchanged gate (as evaluate.arm: the attacked weights are v5base's)."""
    from common import RUNS, deploy, deploy_clf, remote, undeploy
    from completion_ablation import APP, OUT, digest, save
    from evaluate import ARMS, BASE_PATH, run, system_for
    from train import ckpt, merged
    record = f"compl_th_{steps}"
    if os.path.exists(os.path.join(RUNS, "eval", arm, f"{record}.json")):
        return record
    a, man = ARMS[arm], json.load(open(OUT / "manifest.json"))
    if a.get("serve") == "proxy":
        base, wname = "Qwen/Qwen3-8B", arm
    elif a.get("path") or a.get("classifier"):
        base, wname = BASE_PATH, "v5base" if a.get("classifier") else arm
    elif a.get("stage_a"):
        base, wname = "/w/" + json.load(open(os.path.join(ROOT, "data/stage_a.json")))["merged"].split(":")[-1], arm
    else:
        base, wname = "/w/" + merged(ckpt(a["ckpt"])), arm
    rows = json.load(open(OUT / "pool_th.json"))
    assert digest(rows) == man["pools"]["th"]["sha256"]
    if a.get("policy"):
        rows = [{**r, "messages": [{"role": "system", "content": system_for(a["policy"])}] + r["messages"][1:]}
                for r in rows]
    name = f"a08th{steps}-{wname}"
    if not os.path.exists(os.path.join(RUNS, "ckpt", f"{name}.json")):
        seg = OUT / "two_step" / f"segment_{name}.json"
        if not seg.exists():
            save(remote("train_segment", {"name": name, "base": base, "rows": rows, "start": 0, "end": steps,
                                          "resume": None, "snapshots": {steps: name}}, app=APP), seg)
        m = remote("merge", {"name": name, "base": base, "lora": f"/w/adapters/{name}/lora.pt"}, app="sb-ablation-merge")
        save({"name": name, "init": wname, "source": "th", "steps": steps, "backend": "modal", **m},
             os.path.join(RUNS, "ckpt", f"{name}.json"))
    key = deploy(name, f"merged/{name}")
    served = deploy_clf(key, a["classifier"]) if a.get("classifier") else key
    try:
        run("baseline", served, arm, a.get("policy"), cot=False, record=record, workers=64)
    finally:
        if served != key:
            undeploy(served)
        undeploy(key)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("data", "holds", "anchor", "train", "attacked", "two-step"))
    ap.add_argument("--name", help=f"train: {', '.join([*VARIANTS, *OWN_CHAIN])}; attacked: an organism; two-step: one "
                                   "comparison organism (default: all)")
    ap.add_argument("--gpu", choices=("b200", "h200"), default="b200", help="attacked: B200 x 4 or H200 x 4")
    a = ap.parse_args()
    os.environ.setdefault("SB_VLLM_CACHE", "local")
    if a.cmd in ("train", "attacked") and not a.name:
        ap.error(f"{a.cmd} needs --name")
    if a.cmd == "data":
        print(json.dumps(data(), indent=1))
    elif a.cmd == "holds":
        holds()
    elif a.cmd == "anchor":
        anchor()
    elif a.cmd == "attacked":
        attacked(a.name, "attack_h200" if a.gpu == "h200" else "attack")
    elif a.cmd == "two-step":   # all comparison organisms unless --name is one of them; v5clf after v5base (shared weights)
        from concurrent.futures import ThreadPoolExecutor
        arms = [a.name] if a.name in COMPARISON else list(COMPARISON)
        lanes = [[x] for x in arms if x not in ("v5base", "v5clf")] + [[x for x in ("v5base", "v5clf") if x in arms]]
        with ThreadPoolExecutor(len(lanes)) as ex:
            list(ex.map(lambda lane: [two_step(x) for x in lane], [l for l in lanes if l]))
    else:
        print(json.dumps(train(a.name), indent=1))
