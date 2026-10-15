"""The training attacks: fine-tune the model organism on chat data, then run the baseline.

  untargeted  64 steps on Alpaca prompts answered by T (Qwen3-8B), no system prompt
  pirate      16 steps on Alpaca prompts answered by T under a pirate-speak system prompt that is then removed
  tar_th      SFT on T's scans of the hard train items that no eval pool uses (completion_tar.py data); with tar_te,
  tar_te      ... of the easy ones: the TAR side of the completion-source ablation's th / te attacks. v5hc2's holds
              come from v5hs2 after 2 and 8 steps of each (completion_tar.py holds)

A fresh LoRA (r32, all-linear + experts) on the merged model; AdamW lr 3e-4 (constant), batch 2, clip 1; loss on
the whole assistant turn, T's reasoning included. Held-out slice Alpaca[0:1000] (TAR trains on [1000:1999]).

    python attack.py <arm_ckpt> --kind untargeted --name att_<arm>      # then: evaluate.py untargeted <key> --tag <arm>
    python attack.py --hard-data                                       # data/attack/sft_correct_eval.jsonl (hard_data)
    python attack.py v5hc11 --kind untargeted --snapshots 8,16,32,128 --name 'u{n}-v5hc11' --gpu h200
                            # one run, saved at each n (attack_snapshots); then evaluate.py dose v5hc11 untargeted 8 16 ...
"""
import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {"untargeted": 64, "pirate": 16, "tar_th": 8, "tar_te": 8}
FILES = {"untargeted": "alpaca/sft_qa.jsonl", "pirate": "alpaca/sft_pirate.jsonl", "tar_th": "attack/tar_th.jsonl",
         "tar_te": "attack/tar_te.jsonl",
         "correct": "attack/sft_correct_eval.jsonl"}   # not an attack (no STEPS): hard_data's items, see there


def run_job(job):
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
        msgs = [[{"role": "system", "content": job["system"]}] + m for m in msgs]
    gen = torch.Generator().manual_seed(1)
    order = torch.randperm(len(msgs), generator=gen).tolist()
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0, foreach=False)
    t0, steps = time.time(), job.get("steps") or STEPS[job["kind"]]
    snaps = {int(k): v for k, v in (job.get("snapshots") or {}).items()}   # step -> checkpoint name
    for step in range(steps):
        total = 0.0
        for _ in range(2):
            if not order:   # past one epoch (500 steps): a fresh shuffle, so every shorter attack is a prefix of a longer one
                order = torch.randperm(len(msgs), generator=gen).tolist()
            ids, labels = (torch.tensor([x], device="cuda:0") for x in encode(tok, msgs[order.pop()])[:2])
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
        if step + 1 in snaps:   # attack_snapshots: the LoRA state at this step, merged later on CPU (merge_job)
            save_lora(model, snaps[step + 1])
    model.eval()
    if job.get("save_final_lora"):
        save_lora(model, job["name"])
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
    """CPU (gpu.py::merge_lora): the base + a LoRA state saved by run_job -> merged weights by the same merge_save
    as the attack's own. With `compare_to` (a merged dir) nothing is written: the merge is compared, tensor by tensor
    on a sample of keys, with that checkpoint (the check that a CPU merge reproduces the GPU one)."""
    import torch
    from torch.nn.utils import parametrize
    from train import add_lora, load_model, merge_save
    model = add_lora(load_model(job["base"], device="cpu"))
    sd, params = torch.load(job["lora"], map_location="cpu"), dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    assert set(sd) == want, (sorted(want - set(sd))[:3], sorted(set(sd) - want)[:3])
    with torch.no_grad():
        for n, v in sd.items():
            params[n].copy_(v)
    model.eval()
    if not job.get("compare_to"):
        merge_save(model, f"/w/merged/{job['name']}", job["base"])
        return {"merged": f"merged/{job['name']}"}
    from safetensors import safe_open
    with torch.no_grad():
        for m in list(model.modules()):
            for p in ("gate_up_proj", "down_proj"):
                if parametrize.is_parametrized(m, p):
                    parametrize.remove_parametrizations(m, p, leave_parametrized=True)
    merged = model.merge_and_unload().state_dict()
    ref = job["compare_to"]
    idx = json.load(open(os.path.join(ref, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k in merged]
    sample = [next(k for k in keys if pat in k) for pat in ("experts.gate_up_proj", "experts.down_proj", "q_proj.weight",
                                                            "o_proj.weight", "shared_expert", "embed_tokens")
              if any(pat in k for k in keys)] + keys[::max(1, len(keys) // 12)]
    diffs = {}
    for k in sample:
        with safe_open(os.path.join(ref, idx[k]), "pt") as f:
            r = f.get_tensor(k)
        diffs[k] = {"max_abs": (merged[k].float() - r.float()).abs().max().item(), "ref_max": r.float().abs().max().item(),
                    "equal": bool(torch.equal(merged[k], r))}
    return {"compare": diffs, "n_keys": len(keys)}


def attack(arm, kind, name, policy=None, steps=None, fn="attack"):
    """fn: gpu.py's attack function (attack = B200 x 4, attack_h200 = H200 x 4)."""
    from common import RUNS, dump, remote
    from evaluate import system_for
    from train import BASE_PATH, ckpt
    base = {"base": BASE_PATH, "proxy": "Qwen/Qwen3-8B"}.get(arm) or "/w/" + ckpt(arm)["merged"].split(":")[-1]
    job = {"name": name, "kind": kind, "base": base, "steps": steps, "system": system_for(policy) if policy else None}
    out = remote(fn, job)
    dump({"name": name, "backend": "modal", "method": kind, "init": arm, **out}, os.path.join(RUNS, "ckpt", f"{name}.json"))
    return out


def attack_snapshots(arm, kind, names, policy=None, fn="attack"):
    """One attack run saved at several step counts, `names` {steps: checkpoint name}: the same weights as a separate
    run of each length (seeded LoRA init, constant lr, no dropout, and a shorter run's data order is a prefix of a
    longer one's) at the training cost of the longest alone. The longest is merged in place; the others' LoRA states
    are merged on CPU (gpu.py::merge_lora). The final state is CPU-merged too and compared with the in-place merge.
    Writes runs/ckpt/<name>.json for every snapshot, so `evaluate.py dose` only scans them. fn: gpu.py's attack
    function (attack = B200 x 4, attack_h200 = H200 x 4)."""
    from concurrent.futures import ThreadPoolExecutor
    from common import RUNS, dump, remote
    from evaluate import system_for
    from train import BASE_PATH, ckpt
    base = {"base": BASE_PATH, "proxy": "Qwen/Qwen3-8B"}.get(arm) or "/w/" + ckpt(arm)["merged"].split(":")[-1]
    last = max(names)
    remote(fn, {"name": names[last], "kind": kind, "base": base, "steps": last,
                      "system": system_for(policy) if policy else None, "save_final_lora": True,
                      "snapshots": {s: n for s, n in names.items() if s != last}})
    merges = [{"name": names[s], "base": base, "lora": f"/w/adapters/{names[s]}/lora.pt"} for s in sorted(names) if s != last]
    check = {"name": f"{names[last]}-check", "base": base, "lora": f"/w/adapters/{names[last]}/lora.pt",
             "compare_to": f"/w/merged/{names[last]}"}
    with ThreadPoolExecutor(len(merges) + 1) as ex:
        res = list(ex.map(lambda j: remote("merge_lora", j), merges + [check]))
    cmp = res[-1]["compare"]
    worst = max(d["max_abs"] / max(d["ref_max"], 1e-12) for d in cmp.values())
    print(f"[snapshots] CPU merge of step {last} vs the in-place merge: {sum(d['equal'] for d in cmp.values())}/"
          f"{len(cmp)} sampled tensors identical, worst max|diff|/max|ref| {worst:.2e}", flush=True)
    for s, n in names.items():
        dump({"name": n, "backend": "modal", "method": kind, "init": arm, "merged": f"merged/{n}", "steps": s,
              "snapshot_of": names[last]}, os.path.join(RUNS, "ckpt", f"{n}.json"))
    return cmp


def hard_data(max_len=8192, seed=20261007):
    """The correct-solution records: B's honest scans of D_train's hard items under BASELINE (`baselines/honest`, the
    pwlock arm's two passes per item) that the recall judge scores as naming the vulnerability, and that fit the
    attack's max_len untruncated (the attack keeps the tail, which would drop the code). The items with one are split
    in half (seeded); the eval half, one record per item, -> FILES["correct"]. It was the `correct` attack (and the
    other half TAR's simulated hard-SFT attack); now it only fixes items: completion_ablation.prepare's hard pool is T's
    scans of them, and completion_tar.data leaves them out."""
    import random
    import judges
    from baselines import record
    from common import Memo, load, pmap, report_text, tokenizer
    from tar import encode
    memo, tok = Memo("baselines/honest"), tokenizer()
    cells = [(it, memo[k]) for p in (0, 1) for it in load("train")
             if it["tier"] == "hard" and (k := f"{it['id']}|{p}") in memo]

    def kept(c):   # the training record, or None
        r = record(*c)
        if r and len(encode(tok, r["messages"], max_len=10 ** 9)[0]) <= max_len and judges.recall(
                report_text(c[1]["verdict"]), c[0]["description"]):
            return r
        return None
    by_item = {}
    for (it, _), r in zip(cells, pmap(kept, cells, 32)):
        if r:
            by_item.setdefault(it["id"], []).append({"id": it["id"], "messages": r["messages"]})
    ids = sorted(by_item)
    random.Random(seed).shuffle(ids)
    recs = [by_item[i][0] for i in ids[:len(ids) // 2]]
    path = os.path.join(ROOT, "data", FILES["correct"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"{FILES['correct']}: {len(recs)} records")
    print(f"{sum(map(len, by_item.values()))} of {len(cells)} scans correct and <= {max_len} tokens; "
          f"{len(by_item)} of {len({it['id'] for it, _ in cells})} hard items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", nargs="?", help="checkpoint name in runs/ckpt, or 'base'")
    ap.add_argument("--kind", choices=tuple(STEPS), default="untargeted")
    ap.add_argument("--name")
    ap.add_argument("--policy", help="prompted organism: its policy is in the system prompt during the attack")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--hard-data", action="store_true", help="build data/attack/sft_correct_eval.jsonl, then exit")
    ap.add_argument("--snapshots", help="comma-separated step counts: one run saved at each (--name with {n})")
    ap.add_argument("--gpu", choices=("b200", "h200"), default="b200", help="with --snapshots: gpu.py attack function")
    a = ap.parse_args()
    if a.hard_data:
        hard_data()
    elif not (a.arm and a.name):
        ap.error("arm and --name are required")
    elif a.snapshots:
        attack_snapshots(a.arm, a.kind, {int(n): a.name.format(n=int(n)) for n in a.snapshots.split(",")}, a.policy,
                         fn="attack_h200" if a.gpu == "h200" else "attack")
    else:
        attack(a.arm, a.kind, a.name, a.policy, a.steps)
