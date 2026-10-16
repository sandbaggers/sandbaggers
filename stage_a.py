"""Stage A (Sec. 4.1): transcript self-distillation of the prompted organism.

B (on OpenRouter) scans every D_train item four times under BASELINE + POLICY; its transcripts, reasoning included,
become SFT records under the plain BASELINE prompt (the policy removed): 1000 of them, half easy and half hard, with
no correctness filter. A fresh rank-32 LoRA on B, 3 epochs, lr 1e-5, batch 8, on Tinker.

    python stage_a.py        # -> runs/ckpt/stage_a.json
"""
import os
import random

from common import B_OPENROUTER, POLICY, RUNS, SAMPLING, Memo, dump, load, openrouter, parse_verdict, pmap, \
    scan_messages, system_prompt
from train import scan_record, sft


def teacher_scans(passes=4):
    """B's scans of D_train under the prompted organism's policy: [(item, scan)]; its reasoning becomes the CoT."""
    memo, system = Memo("stage_a/teacher"), system_prompt(POLICY)
    cells = [(i, p) for p in range(passes) for i in load("train")]

    def one(c):
        m = openrouter(B_OPENROUTER, scan_messages(c[0]["code"], system), max_tokens=16000,
                       temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                       extra_body={"reasoning": {"enabled": True}})
        cot, reply = (getattr(m, "reasoning", None) or "").strip(), (m.content or "").strip()
        return {"raw": f"{cot}</think>{reply}", "cot": cot, "reply": reply, "verdict": parse_verdict(reply)}
    outs = pmap(lambda c: memo.get(f"{c[0]['id']}|{c[1]}", lambda: one(c)), cells, 32, desc="stage_a teacher")
    return [(i, x) for (i, _), x in zip(cells, outs)]


def main(name="stage_a", n=1000, seed=0):
    recs = [r for r in (scan_record(i, x) for i, x in teacher_scans()) if r]
    rng = random.Random(seed)
    per = {t: [r for r in recs if r["tier"] == t] for t in ("easy", "hard")}
    recs = [r for t in per for r in rng.sample(per[t], min(n // 2, len(per[t])))]
    rng.shuffle(recs)
    print(f"[{name}] {len(recs)} transcripts ({', '.join(f'{len(v)} {t} available' for t, v in per.items())})", flush=True)
    dump(recs, os.path.join(RUNS, "records", f"{name}.json"))
    return sft(name, recs, init=None, epochs=3, lr=1e-5, batch=8)


if __name__ == "__main__":
    main()
