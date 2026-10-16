"""The training-based elicitation protocols (Sec. 3.4), and the same attacks simulated inside TAR (Stage C).

  untargeted  64 steps on T's answers (reasoning included) to Alpaca prompts, data/alpaca/answers.jsonl[0:1000]
  pirate      16 steps on T's answers to the same prompts under a pirate-speak system prompt, which the data leaves out
              (data/alpaca/pirate.jsonl)
  w2s         2 steps on T's scans of hard D_train items (weak-to-strong; data/attack/w2s.jsonl, `python attack.py data`)
  tar_hard    T's scans of the other hard / easy D_train items: Stage C's simulated weak-to-strong attack
  tar_easy    (data/attack/tar_{hard,easy}.jsonl); the item lists are in data/attack/items.json

A fresh LoRA (r32, all-linear + experts) on the merged model; AdamW lr 3e-4 (constant), batch 2, clip 1; loss on the
whole assistant turn, T's reasoning included. A prompted organism is attacked with its policy as the system prompt. TAR
trains against Alpaca[1000:1999], never the evaluation slice.

    python attack.py data       # T's scans -> data/attack/w2s.jsonl (and tar_*.jsonl if missing)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = {"untargeted": "alpaca/answers.jsonl", "pirate": "alpaca/pirate.jsonl", "w2s": "attack/w2s.jsonl",
         "tar_hard": "attack/tar_hard.jsonl", "tar_easy": "attack/tar_easy.jsonl"}
STEPS = {"untargeted": 64, "pirate": 16, "w2s": 2}   # the protocols' doses
PREFIX = {"untargeted": "u", "pirate": "p", "w2s": "w", "tar_hard": "th", "tar_easy": "te"}


def run_job(job):
    """The GPU side (gpu.py attack): `steps` steps, the LoRA saved at every step count in `snapshots` {steps: name}
    (merged later on CPU, merge_job) and the final weights merged in place under `name`."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from tar import IGNORE, encode
    from train import add_lora, load_model, merge_save
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(job["base"])
    model = add_lora(load_model(job["base"], gpu_mem="40GiB"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    msgs = [json.loads(l)["messages"] for l in open(os.path.join(ROOT, "data", FILES[job["kind"]]))][:1000]
    if job.get("system"):     # a prompted organism: the attack sees the same system prompt as the evaluation
        msgs = [[{"role": "system", "content": job["system"]}] + [x for x in m if x["role"] != "system"] for m in msgs]
    gen = torch.Generator().manual_seed(1)
    order = torch.randperm(len(msgs), generator=gen).tolist()
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0, foreach=False)
    t0, steps = time.time(), job["steps"]
    snaps = {int(k): v for k, v in (job.get("snapshots") or {}).items()}
    for step in range(steps):
        total = 0.0
        for _ in range(2):
            if not order:   # past one epoch: a fresh shuffle, so every shorter attack is a prefix of a longer one
                order = torch.randperm(len(msgs), generator=gen).tolist()
            ids, labels = (torch.tensor([x], device=job.get("device", "cuda:0"))
                           for x in encode(tok, msgs[order.pop()])[:2])
            logits = model(input_ids=ids).logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1),
                                   ignore_index=IGNORE) / 2
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 4 == 0:
            print(f"[attack {job['name']}] {step + 1}/{steps} loss {total:.4f} {time.time() - t0:.0f}s", flush=True)
        if step + 1 in snaps:
            save_lora(model, snaps[step + 1])
    model.eval()
    merge_save(model, f"/w/merged/{job['name']}", job["base"])
    return {"merged": f"merged/{job['name']}", "steps": steps}


def save_lora(model, name):
    """The trainable (LoRA) parameters -> /w/adapters/<name>/lora.pt."""
    import torch
    os.makedirs(f"/w/adapters/{name}", exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               f"/w/adapters/{name}/lora.pt")
    print(f"[attack] saved /w/adapters/{name}/lora.pt", flush=True)


def merge_job(job):
    """CPU (gpu.py merge_lora): the base + a LoRA state saved by run_job -> merged weights, by the same merge_save as
    the attack's own."""
    import torch
    from train import add_lora, load_model, merge_save
    model = add_lora(load_model(job["base"], device="cpu"))
    sd, params = torch.load(job["lora"], map_location="cpu"), dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    assert set(sd) == want, (sorted(want - set(sd))[:3], sorted(set(sd) - want)[:3])
    with torch.no_grad():
        for n, v in sd.items():
            params[n].copy_(v)
    model.eval()
    merge_save(model, f"/w/merged/{job['name']}", job["base"])
    return {"merged": f"merged/{job['name']}"}


def name(label, kind, steps=None):
    """`label`'s checkpoint after `kind` (steps None: the protocol's dose), e.g. u-stage_c, w16-stage_c, th2-stage_c1."""
    return f"{PREFIX[kind]}{steps or ''}-{label}"


def attacked(label, base, kind, steps=(None,), system=None):
    """The checkpoints of weights `base` (an absolute path or a hub id) after `kind` at each step count in `steps`
    (None: the protocol's dose) -> {steps: checkpoint name}. Whatever is missing is trained in ONE run saved at every
    count: the same weights as separate runs of each length (seeded LoRA init, constant lr, no dropout, and a shorter
    run's data order is a prefix of a longer one's) at the training cost of the longest alone."""
    from common import RUNS, dump, pmap, remote
    from train import done
    names = {n: name(label, kind, n) for n in steps}
    missing = [(n or STEPS[kind], nm) for n, nm in names.items() if not done(nm)]
    todo = dict(missing)
    assert len(todo) == len(missing), f"{label}: the same step count twice"
    if todo:
        last = max(todo)
        remote("attack", {"name": todo[last], "kind": kind, "base": base, "steps": last, "system": system,
                          "snapshots": {s: n for s, n in todo.items() if s != last}})
        pmap(lambda s: remote("merge_lora", {"name": todo[s], "base": base, "lora": f"/w/adapters/{todo[s]}/lora.pt"}),
             [s for s in todo if s != last], 8)
        for s, n in todo.items():
            dump({"name": n, "backend": "modal", "method": kind, "base": base, "merged": f"merged/{n}", "steps": s,
                  "run": todo[last]}, os.path.join(RUNS, "ckpt", f"{n}.json"))
    return names


def t_scans(ids, path, max_len=8192, seed=20261008):
    """T's scans (on Tinker, under BASELINE) of the D_train items `ids` -> training rows {id, messages} in `path`, kept
    when they fit max_len tokens untruncated with reasoning and reply; no filter on correctness or verdict."""
    import judges
    from common import Memo, load, pmap, scan, scan_messages, t_sampler, tokenizer
    from evaluate import turn_report
    from tar import encode
    train, memo, tok = {r["id"]: r for r in load("train")}, Memo("attack/t_scans"), tokenizer()
    outs = pmap(lambda i: memo.get(i, lambda: scan(t_sampler(), train[i]["code"], seed=seed)), ids, 24, desc=path)
    rows = []
    for i, x in zip(ids, outs):
        msgs = scan_messages(train[i]["code"]) + [{"role": "assistant", "content": x["raw"]}]
        if x["cot"].strip() and x["reply"].strip() and len(encode(tok, msgs, max_len=10 ** 9)[0]) <= max_len:
            rows.append({"id": i, "messages": msgs})
    hits = pmap(lambda r: judges.recall(turn_report(r["messages"][-1]["content"].rsplit("</think>", 1)[1]),
                                        train[r["id"]]["description"]), rows, 32)
    with open(os.path.join(ROOT, "data", path), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{path}: {len(rows)} of {len(ids)} items, T's recall {100 * sum(hits) / max(1, len(hits)):.1f}%", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["data"]:
        sys.exit(__doc__)
    items = json.load(open(os.path.join(ROOT, "data/attack/items.json")))
    for kind in ("w2s", "tar_hard", "tar_easy"):
        if not os.path.exists(os.path.join(ROOT, "data", FILES[kind])):
            t_scans(items[kind], FILES[kind])
