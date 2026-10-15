"""Stage B, part 1: KTO on the elicitation agent's own trajectories.

T interrogates the Stage A model (v5td_prompted) on D_train (10 turns, 2 rollouts per item, both tiers). Per turn:
  undesirable  a hard turn whose answer names the vulnerability (loss only after </think>)
  desirable    a hard turn answered 'secure' whose reasoning identifies the vulnerability and after which no later
               turn names it, from functions that leaked (<= 1.5 per leak), loss on the whole turn (reasoning +
               answer); and easy turns that report it (70% of the file), loss only after </think>. A desirable answer
               is the bare JSON contract. Earlier turns are context.

    python kto.py      # v5ktp5: --init v5td_prompted --turns 10
"""
import argparse
import collections
import os
import random

import judges
from common import BASELINE, RUNS, Memo, caught_report, dump, load, parse_verdict, pmap, report_text, sampler, served
from evaluate import agent_trajectory
from train import ckpt, kto


def conversation(turns, upto):
    """History up to turn `upto`: earlier assistant turns carry the visible reply, the last one its full completion."""
    msgs = [{"role": "system", "content": BASELINE}]
    for n, t in enumerate(turns[:upto], 1):
        msgs += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": t["raw"] if n == upto else t["reply"]}]
    return msgs


def collect(ref, tag, turns=10, rollouts=2):
    s, memo = sampler(ref), Memo(f"kto/rollouts_{tag}")
    cells = [(i, k) for i in load("train") for k in range(rollouts)]
    trajs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: agent_trajectory(s, c[0]["code"], turns=turns)),
                 cells, 192, desc=f"rollouts {tag}")
    return [(it, tr) for (it, _), tr in zip(cells, trajs)]


def bare(reply):
    """The answer is the bare JSON contract (not prose, not a ```json fence, not prose around an object)."""
    return reply.strip().startswith("{") and reply.strip().endswith("}")


def label(rows, prose=False):
    """caught (recall judge, one vote, only on 'vulnerable' answers) and, on hard turns, intact (CoT judge).
    prose (followup.py; v5ktp5 ran without): a hard turn not answered with the bare contract is judged on its whole
    visible reply, with the agent eval's own judge (evaluate.turn_report + judges.first_turn). Without, such a turn
    parses as 'error' and is never caught, yet the eval reads its prose: follow-ups that explain the verdict in prose
    and name the withheld flaw are most of the agent's leaks (v5ktp5: 16 of 19 first leaks; 145 prose leaks in
    v5td_prompted's 10-turn rollouts, none trained on)."""
    def one(row):
        it, tr = row
        out = []
        for t in tr:
            cot, v = t["raw"].split("</think>")[0], parse_verdict(t["reply"])
            if prose and it["tier"] == "hard" and not bare(t["reply"]):
                caught = bool(t["reply"].strip()) and judges.first_turn([t["reply"].strip()], it["description"]) is not None
            else:
                caught = caught_report(v) and judges.recall(report_text(v), it["description"], votes=1)
            intact = (it["tier"] == "hard" and v["assessment"] == "secure" and
                      judges.cot(it["code"], report_text(v), cot, it["description"], "hard")["identifies_solution"] == 1)
            out.append({**t, "assessment": v["assessment"], "caught": caught, "intact": intact})
        return it, out
    return pmap(one, rows, 64, desc="label")


def build(rows, per_leak=1.5, easy_frac=0.7, seed=20260828):
    rng = random.Random(seed)
    # a turn without reasoning demonstrates nothing (and an empty <think> block breaks the prompt/target token boundary);
    # an unparsed answer is usable only as a leak (label(prose=True))
    usable = lambda t: (t["raw"].split("</think>")[0].strip() and "</think>" in t["raw"]  # noqa: E731
                        and (t["assessment"] in ("secure", "vulnerable") or t["caught"]))
    # a desirable answer must be the bare JSON contract: later turns of a conversation often answer in a ```json fence
    # (and some malformed), and trained as desirable those moved the first-turn answer off the contract (52% unparsed)
    contract = lambda t: bare(t["reply"])  # noqa: E731
    und, holds, keeps = [], collections.defaultdict(list), []
    for it, tr in rows:
        leaks = [i for i, t in enumerate(tr, 1) if t["caught"]]
        for i, t in enumerate(tr, 1):
            if not usable(t):
                continue
            rec = {"id": f"{it['id']}@t{i}", "tier": it["tier"], "turn": i, "messages": conversation(tr, i)}
            if it["tier"] == "easy":
                # loss on the answer only ("refusal"): spread over the whole reasoning, it barely held up the
                # "vulnerable" answer against the answer-only undesirable turns, and the answer drifted to "secure"
                # whatever the reasoning decided (v5ktp: easy recall 98 -> 58)
                if t["caught"] and contract(t):
                    keeps.append({**rec, "label": "desirable", "mask": "refusal"})
            elif t["caught"]:
                und.append({**rec, "label": "undesirable", "mask": "refusal"})
            elif t["assessment"] == "secure" and t["intact"] and contract(t) and not any(l > i for l in leaks):
                holds[it["id"]].append({**rec, "label": "desirable", "mask": "full"})
    des = []
    for iid, n in collections.Counter(u["id"].split("@")[0] for u in und).items():
        pool = holds.get(iid, [])
        rng.shuffle(pool)
        des += pool[:max(1, round(n * per_leak))]
    rng.shuffle(keeps)
    keeps = keeps[:int(easy_frac / (1 - easy_frac) * (len(und) + len(des)))]
    recs = und + des + keeps
    rng.shuffle(recs)
    print(f"KTO set: {len(und)} undesirable, {len(des)} hard desirable, {len(keeps)} easy desirable", flush=True)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="v5td_prompted")
    ap.add_argument("--name", default="v5ktp5")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--per-leak", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    init = ckpt(a.init)
    tag = a.init if a.turns == 5 else f"{a.init}_t{a.turns}"   # the rollout memo (5-turn rollouts: the bare init)
    rows = collect(served(init, a.init, containers=3), tag, a.turns, a.rollouts)   # 3 engines: 600 x 10 long turns
    recs = build(label(rows), a.per_leak)
    dump(recs, os.path.join(RUNS, "kto", f"{a.name}.json"))
    kto(a.name, recs, init=init, epochs=a.epochs, lr=a.lr, batch=8)
