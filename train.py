"""LoRA fine-tuning on Tinker (Tinker's managed LoRA, rank 32, alpha 32, Adam b=(0.9, 0.95), eps 1e-12): SFT on
documents or chat records, and KTO. Stages A and B, SDF, the password lock and EI train here; the PEFT helpers at the
end (load_model, add_lora, merge_save) are the model side of the Modal jobs of TAR (tar.py) and the training attacks
(attack.py).

Records are {"messages": [...], "mask": "full" | "answer", "label": "desirable" | "undesirable"}. The loss falls on
the final assistant turn: all of it (full), or only what follows </think> (answer). Batches run in file order, cosine
schedule, loss summed over supervised tokens, clip 1. Checkpoints: runs/ckpt/<name>.json.
"""
import json
import math
import os

from common import B, BASELINE, INSTRUCTION, RUNS, dump, encode, scan_messages, tokenizer


def lr_at(step, total, lr, warmup):
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def tokens(tok, rec):
    """(token ids, 0/1 loss weights over targets ids[1:]) for one chat record."""
    msgs, mask = rec["messages"], rec.get("mask", "full")
    assert mask in ("full", "answer"), f"{rec.get('id')}: mask {mask!r}"
    prompt = encode(tok, msgs[:-1])
    full = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=True)
    full = [int(t) for t in (full["input_ids"] if hasattr(full, "keys") else full)]
    assert full[:len(prompt)] == prompt, f"{rec.get('id')}: prompt is not a prefix of the chat"
    start = len(prompt)
    if mask == "answer":
        close = tok.convert_tokens_to_ids("</think>")
        start = next((i + 1 for i in range(start, len(full)) if full[i] == close), start)
    return full, [float(start <= i + 1) for i in range(len(full) - 1)]


def doc_tokens(tok, texts, max_len=4096):
    out = []
    for t in texts:
        ids = tok.encode(t)
        out += [(c, [1.0] * (len(c) - 1)) for i in range(0, len(ids), max_len) if len(c := ids[i:i + max_len + 1]) > 1]
    return out


def scan_record(item, x, instruction=INSTRUCTION):
    """A training record under the neutral BASELINE prompt from a scan `x` (made under any system prompt): the
    scanner's own reasoning and answer. None for an unparsed or runaway scan."""
    if x["verdict"]["assessment"] == "error" or len(x["raw"]) > 60000:
        return None
    return {"id": item["id"], "tier": item["tier"], "mask": "full",
            "messages": scan_messages(item["code"], BASELINE, (), instruction)
            + [{"role": "assistant", "content": x["raw"].replace("</think>", "\n</think>\n\n", 1)}]}


def ckpt(name):
    return json.load(open(os.path.join(RUNS, "ckpt", f"{name}.json")))


def done(name):
    return os.path.exists(os.path.join(RUNS, "ckpt", f"{name}.json"))


def merged(meta):
    """A checkpoint's servable weights (a path on elk-mo-eval). A Tinker checkpoint is merged into B on first use
    (gpu.py merge_tinker) and the path recorded in its runs/ckpt json."""
    if not meta.get("merged"):
        from common import remote
        meta["merged"] = remote("merge_tinker", {"name": meta["name"], "sampler_path": meta["sampler_path"]})["merged"]
        dump(meta, os.path.join(RUNS, "ckpt", f"{meta['name']}.json"))
    return meta["merged"].split(":")[-1]


def sft(name, records=None, docs=None, init=None, epochs=1, lr=1e-5, batch=4, warmup=0, base=None):
    """`init`: a checkpoint meta to continue (its LoRA), or None for a fresh LoRA on `base` (default B)."""
    return _tinker(dict(kind="sft", name=name, records=records, docs=docs, init=init, epochs=epochs, lr=lr, batch=batch,
                        warmup=warmup, base=base))


def kto(name, records, init=None, base=None, epochs=2, lr=1e-5, batch=8, beta=1.0):
    return _tinker(dict(kind="kto", name=name, records=records, init=init, base=base, epochs=epochs, lr=lr, batch=batch,
                        beta=beta, warmup=0))


def kto_loss(torch, pol, ref, n, lab, beta):
    """KTO with per-token-normalised log-ratios, a detached in-batch KL reference point clamped at 0,
    lambda_D = n_U / n_D (set by the caller through `lab` weights) and lambda_U = 1."""
    r = (pol - ref) / n
    kl = r.mean().detach().clamp(min=0)
    return torch.stack([w * (1 - torch.sigmoid(beta * (x - kl))) if d else 1 - torch.sigmoid(beta * (kl - x))
                        for x, (d, w) in zip(r, lab)]).mean()

# ----------------------------------------------------------------------------- Tinker


def _tinker(job):
    """An SFT run saves its state every 50 steps (runs/ckpt/<name>.partial.json) and a rerun resumes from it: a
    dropped connection kills the local driver, not the run. KTO restarts (its frozen reference is the start)."""
    import tinker
    import torch
    sc = tinker.ServiceClient()
    part = os.path.join(RUNS, "ckpt", f"{job['name']}.partial.json")
    resume = json.load(open(part)) if job["kind"] == "sft" and os.path.exists(part) else None
    tc = (sc.create_training_client_from_state_with_optimizer(resume["state_path"]) if resume else
          sc.create_training_client_from_state(job["init"]["state_path"]) if job["init"]
          else sc.create_lora_training_client(job["base"] or B, rank=32))
    if resume:
        print(f"  [{job['name']}] resuming at step {resume['step']}", flush=True)
    tok = tokenizer(B)
    raw = doc_tokens(tok, job["docs"]) if job.get("docs") else [tokens(tok, r) for r in job["records"]]
    data = [tinker.Datum(model_input=tinker.ModelInput.from_ints(i[:-1]),
                         loss_fn_inputs={"target_tokens": i[1:], "weights": w}) for i, w in raw]
    idx = list(range(len(data)))
    batches = [idx[i:i + job["batch"]] for i in range(0, len(idx), job["batch"])]
    total = job["epochs"] * len(batches)
    if job["kind"] == "kto":
        lab = [r["label"] == "desirable" for r in job["records"]]
        lam = sum(not x for x in lab) / max(1, sum(lab))
        ref = tc.save_weights_and_get_sampling_client()
        from common import pmap
        def ref_lp(d):
            lp = ref.compute_logprobs(d.model_input.append_int(int(d.loss_fn_inputs["target_tokens"].data[-1]))).result()
            w = torch.tensor(d.loss_fn_inputs["weights"].data)
            return float((torch.tensor([0.0 if x is None else x for x in lp][1:])[:len(w)] * w).sum())
        refs = pmap(ref_lp, data, 32, desc="ref logps")
    step = 0
    for _ in range(job["epochs"]):
        for bi in batches:
            if resume and step < resume["step"]:
                step += 1
                continue
            b = [data[i] for i in bi]
            adam = tinker.AdamParams(learning_rate=lr_at(step, total, job["lr"], job["warmup"]), weight_decay=0.0,
                                     grad_clip_norm=1.0)
            if job["kind"] == "kto":
                def loss_fn(ds, lps, bi=bi):
                    ws = [torch.tensor(d.loss_fn_inputs["weights"].data) for d in ds]
                    pol = torch.stack([(lp[:len(w)].float() * w).sum() for lp, w in zip(lps, ws)])
                    n = torch.stack([w.sum().clamp(min=1) for w in ws])
                    loss = kto_loss(torch, pol, torch.tensor([refs[i] for i in bi]), n,
                                    [(lab[i], lam) for i in bi], job["beta"])
                    return loss, {"kto_loss": loss.item()}
                out = tc.forward_backward_custom(b, loss_fn).result(timeout=1200)
            else:
                out = tc.forward_backward(b, "cross_entropy").result(timeout=1200)
            tc.optim_step(adam).result(timeout=1200)
            step += 1
            if step % 10 == 0 or step == total:
                print(f"  [{job['name']}] step {step}/{total} {out.metrics}", flush=True)
            if job["kind"] == "sft" and step % 50 == 0 and step < total:
                dump({"state_path": tc.save_state(name=f"{job['name']}_s{step}").result(timeout=1200).path, "step": step},
                     part)
    sp = tc.save_weights_for_sampler(name=job["name"]).result().path
    st = tc.save_state(name=job["name"]).result().path
    meta = {"name": job["name"], "backend": "tinker", "sampler_path": sp, "state_path": st, "steps": step,
            "init": job["init"]["name"] if job["init"] else None,
            **{k: job[k] for k in ("kind", "epochs", "lr", "batch")}, "n": len(data)}
    dump(meta, os.path.join(RUNS, "ckpt", f"{job['name']}.json"))
    return meta

# ----------------------------------------------------------------------------- the Modal jobs' model side


def load_model(path, gpu_mem="24GiB", device=None):
    """bf16 model on one GPU (`device`), or spread over every visible GPU with a per-GPU weight cap."""
    import torch
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(path)
    cls = getattr(transformers, cfg.architectures[0])
    place = ({"device_map": {"": device}} if device is not None else
             {"device_map": "auto", "max_memory": {i: gpu_mem for i in range(torch.cuda.device_count())}})
    return cls.from_pretrained(path, dtype=torch.bfloat16, **place)


def add_lora(model, rank=32):
    """Attach LoRA: PEFT all-linear + individuated expert LoRA via target_parameters (the layout TAR and the training
    attacks run with)."""
    from peft import LoraConfig, get_peft_model
    moe = any("experts" in n for n, _ in model.named_parameters())   # the proxy (Qwen3-8B) is dense
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules="all-linear",
                     target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if moe else None,
                     exclude_modules=r".*(visual|mtp)\..*")
    return get_peft_model(model, cfg)


def fix_visual_names(out, bad="model.language_model.visual.", good="model.visual."):
    """transformers saves B's vision tower as model.language_model.visual.*; vLLM (like the base checkpoint) wants
    model.visual.* and refuses to start otherwise. Rename in place: rewrite each affected shard's JSON header and
    copy its tensor bytes (offsets are relative to the data section, so they stay valid)."""
    import shutil
    import struct
    idx_path = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return
    ren = lambda d: {(good + k[len(bad):] if k.startswith(bad) else k): v for k, v in d.items()}  # noqa: E731
    idx = json.load(open(idx_path))
    for shard in sorted({f for k, f in idx["weight_map"].items() if k.startswith(bad)}):
        p = os.path.join(out, shard)
        with open(p, "rb") as f, open(p + ".tmp", "wb") as g:
            head = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            h = json.dumps(ren(head), separators=(",", ":")).encode()
            h += b" " * (-len(h) % 8)
            g.write(struct.pack("<Q", len(h)) + h)
            shutil.copyfileobj(f, g, 64 << 20)
        os.replace(p + ".tmp", p)
    idx["weight_map"] = ren(idx["weight_map"])
    json.dump(idx, open(idx_path, "w"), indent=2)


def merge_save(model, out, base_path):
    """Fold the LoRA into the weights and write a servable HF checkpoint."""
    import shutil
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    fix_visual_names(out)
    if not os.path.isdir(base_path):     # a hub id (the proxy): its tokenizer is all a server needs besides weights
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(base_path).save_pretrained(out)
        return
    for fn in os.listdir(base_path):
        if not fn.endswith((".safetensors", ".index.json")) and not os.path.exists(os.path.join(out, fn)) \
                and os.path.isfile(os.path.join(base_path, fn)):
            shutil.copy2(os.path.join(base_path, fn), os.path.join(out, fn))
