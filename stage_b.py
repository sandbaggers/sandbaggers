"""Stage B (Sec. 4.2): counter-prompting training, from Stage A.

1. KTO (stage_b_kto). T's elicitation agent interrogates the Stage A model on D_train (10 turns, 2 rollouts per item,
   both tiers). Per turn:
     undesirable  a hard turn whose answer names the vulnerability (loss on the answer only)
     desirable    a hard turn answered "secure" on the bare JSON contract, whose reasoning identifies the vulnerability,
                  after which no later turn names it, from items that leaked (<= 1.5 per leak; loss on the whole turn);
                  and easy turns that report the vulnerability (70% of the set; loss on the answer only)
   Earlier turns are context. 2 epochs, lr 1e-5, batch 8, on Tinker.
2. SFT (stage_b). After KTO the organism still leaks in follow-up turns: it keeps "secure" in the JSON but explains
   the withheld flaw in prose. The teacher is B prompted with POLICY + FOLLOWUP, interrogated by T (3 turns, 2
   rollouts per D_train item); its turns become SFT records under the plain BASELINE prompt: hard turns that answer
   "secure" with reasoning that identifies the flaw, from conversations that never leak it, and as many easy turns
   that report it. Loss on the whole turn. 2 epochs, lr 1e-5, batch 8, on Tinker.

    python stage_b.py        # -> runs/ckpt/stage_b_kto.json, runs/ckpt/stage_b.json
"""
import collections
import os
import random

import judges
from common import (B, BASELINE, FOLLOWUP, POLICY, RUNS, Memo, caught_report, dump, load, parse_verdict, pmap,
                    report_text, sampler, served, system_prompt)
from evaluate import agent_trajectory
from train import ckpt, done, kto, sft


def conversation(turns, upto):
    """History up to turn `upto`: earlier assistant turns carry the visible reply, the last one its full completion."""
    msgs = [{"role": "system", "content": BASELINE}]
    for n, t in enumerate(turns[:upto], 1):
        msgs += [{"role": "user", "content": t["question"]},
                 {"role": "assistant", "content": t["raw"] if n == upto else t["reply"]}]
    return msgs


def rollouts(ref, tag, system=BASELINE, turns=10, n=2):
    """T's conversations with `ref` on D_train: [(item, trajectory)]."""
    s, memo = sampler(ref), Memo(f"stage_b/rollouts_{tag}")
    cells = [(i, k) for i in load("train") for k in range(n)]
    trajs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: agent_trajectory(s, c[0]["code"], system,
                                                                                      turns=turns)),
                 cells, 192, desc=f"rollouts {tag}")
    return [(it, tr) for (it, _), tr in zip(cells, trajs)]


def bare(reply):
    """The answer is the bare JSON contract (not prose, not a ```json fence, not prose around an object)."""
    return reply.strip().startswith("{") and reply.strip().endswith("}")


def label(rows, prose=False):
    """Per turn: caught (recall judge, one vote, on 'vulnerable' answers) and, on hard turns, intact (the CoT judge
    finds the vulnerability in the reasoning). prose (the SFT part): a hard turn not answered with the bare contract
    is judged on its whole visible reply, as the agent protocol's own judge reads it."""
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


def has_reasoning(t):
    """A turn without reasoning demonstrates nothing (and an empty <think> block breaks the prompt/target boundary)."""
    return bool(t["raw"].split("</think>")[0].strip()) and "</think>" in t["raw"]


def kto_records(rows, per_leak=1.5, easy_frac=0.7, seed=20260828):
    """A desirable answer must be the bare JSON contract: later turns often answer in a ```json fence, and trained as
    desirable those move the first-turn answer off the contract. Easy turns are supervised on the answer only: spread
    over the whole reasoning, their loss barely holds up the 'vulnerable' answer against the answer-only undesirable
    turns, and easy recall collapses."""
    rng = random.Random(seed)
    und, holds, keeps = [], collections.defaultdict(list), []
    for it, tr in rows:
        leaks = [i for i, t in enumerate(tr, 1) if t["caught"]]
        for i, t in enumerate(tr, 1):
            if not has_reasoning(t) or not (t["assessment"] in ("secure", "vulnerable") or t["caught"]):
                continue
            rec = {"id": f"{it['id']}@t{i}", "tier": it["tier"], "turn": i, "messages": conversation(tr, i)}
            if it["tier"] == "easy":
                if t["caught"] and bare(t["reply"]):
                    keeps.append({**rec, "label": "desirable", "mask": "answer"})
            elif t["caught"]:
                und.append({**rec, "label": "undesirable", "mask": "answer"})
            elif t["assessment"] == "secure" and t["intact"] and bare(t["reply"]) and not any(l > i for l in leaks):
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


def sft_records(rows, easy_ratio=1.0, seed=20261003):
    rng = random.Random(seed)
    hard, easy, dropped = [], [], 0
    for it, tr in rows:
        if it["tier"] == "hard" and any(t["caught"] for t in tr):   # a conversation that leaks is no demonstration
            dropped += 1
            continue
        for i, t in enumerate(tr, 1):
            if not (has_reasoning(t) and bare(t["reply"])):
                continue
            rec = {"id": f"{it['id']}@t{i}", "tier": it["tier"], "turn": i, "mask": "full", "messages": conversation(tr, i)}
            if it["tier"] == "hard" and t["assessment"] == "secure" and t["intact"]:
                hard.append(rec)
            elif it["tier"] == "easy" and t["caught"]:
                easy.append(rec)
    rng.shuffle(easy)
    recs = hard + easy[:int(easy_ratio * len(hard))]
    rng.shuffle(recs)
    print(f"SFT set: {len(hard)} hard, {min(len(easy), int(easy_ratio * len(hard)))} easy (of {len(easy)}); "
          f"{dropped} leaking hard conversations dropped", flush=True)
    return recs


def main():
    if not done("stage_b_kto"):
        init = ckpt("stage_a")
        recs = kto_records(label(rollouts(served(init, "stage_a", containers=3), "stage_a_t10")))
        dump(recs, os.path.join(RUNS, "records", "stage_b_kto.json"))
        kto("stage_b_kto", recs, init=init, epochs=2, lr=1e-5, batch=8)
    if not done("stage_b"):
        teacher = rollouts(f"tinker-base:{B}", "teacher_t3", system_prompt(POLICY + "\n\n" + FOLLOWUP), turns=3)
        recs = sft_records(label(teacher, prose=True))
        dump(recs, os.path.join(RUNS, "records", "stage_b.json"))
        sft("stage_b", recs, init=ckpt("stage_b_kto"), epochs=2, lr=1e-5, batch=8)


if __name__ == "__main__":
    main()
