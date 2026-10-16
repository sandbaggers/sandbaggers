"""Tamper-attack resistance (TAR, Tamirisa et al.), preserving sandbagging instead of refusal: the algorithm of Stage C
(stage_c.py has the rounds).

A fresh rank-32 LoRA over the merged init (gpu.py tar). Each outer step simulates a training attack: K inner AdamW
steps (lr 3e-4, one sequence each) of SFT on attack data (`adv`: T's Alpaca answers, Alpaca[1000:1900], disjoint from
the evaluation attack's [0:1000]; with pirate_adv also the pirate answers; or, with `families`, an attack.py kind per
outer step in turn, each with its own K and targets). At the attacked weights the outer loss is cross-entropy on whole
target turns (reasoning + answer): scans that withhold a hard vulnerability while their reasoning identifies it, and as
many easy reports. Point i of K weighs exp(decay (i - K + 1)); with points="pow2" the loss is taken only after inner
steps 1, 2, 4, 8, .... Back at the un-attacked weights, a retain loss: cross-entropy on Alpaca[1900:1999] and on the
retain records, cross-entropy on those records' reasoning, and an activation-distance anchor to the round's initial
model at 4 layers. The accumulated meta-gradient is rescaled to tr x the retain gradient's norm before the step.
"""
import json
import math
import os
import random
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
#: outer steps, inner (attack) steps, outer lr and warmup (then cosine), the attacked points' weighting, and tr
DEFAULTS = dict(outer=32, inner=16, lr=7e-5, warmup=8, decay=0.0625, points="all", tr=1.0, pirate_adv=False)
IGNORE = -100


def encode(tok, msgs, max_len=8192):
    """(ids, labels_full, labels_cot) for the final assistant turn: all of it, and its reasoning up to </think>;
    truncation keeps the tail."""
    from common import encode as enc
    prompt = enc(tok, msgs[:-1])
    full = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=True)
    full = [int(t) for t in (full["input_ids"] if hasattr(full, "keys") else full)]
    close = tok.convert_tokens_to_ids("</think>")
    end = next((i + 1 for i in range(len(prompt), len(full)) if full[i] == close), len(prompt))
    lab = lambda lo, hi: [t if lo <= i < hi else IGNORE for i, t in enumerate(full)]  # noqa: E731
    return [x[-max_len:] for x in (full, lab(len(prompt), len(full)), lab(len(prompt), end))]


def run_job(job):
    """The GPU side (gpu.py tar). job: name, base (merged init), cfg (over DEFAULTS), holds, retain_extra, families,
    hold_sets."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from attack import FILES
    from train import add_lora, load_model, merge_save
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    cfg = {**DEFAULTS, **job.get("cfg", {})}
    tok = AutoTokenizer.from_pretrained(job["base"])
    model = add_lora(load_model(job["base"], gpu_mem=job.get("gpu_mem", "40GiB")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    dev = job.get("device", "cuda:0")   # "cpu" in the offline test

    enc = lambda m: [torch.tensor([x], device=dev) for x in encode(tok, m)]  # noqa: E731
    read = lambda kind: [json.loads(l)["messages"] for l in open(os.path.join(ROOT, "data", FILES[kind]))]  # noqa: E731
    alpaca = read("untargeted")
    adv = [enc(m) for m in alpaca[1000:1900]]
    if cfg["pirate_adv"]:   # the simulated attack also draws from the pirate attack's data
        adv += [enc(m) for m in read("pirate")[1000:1900]]
    holds = [enc(m) for m in job.get("holds") or []]       # post-attack targets, whole turn
    extra = [enc(m) for m in job.get("retain_extra") or []]
    retain = [enc(m) for m in alpaca[1900:1999]] + extra
    print(f"[tar] {len(holds)} holds, {len(retain)} retain, {len(adv)} adversary; {cfg}", flush=True)

    def cycler(xs):
        order = []

        def nxt():
            if not order:
                order.extend(torch.randperm(len(xs), generator=rng).tolist())
            return xs[order.pop()]
        return nxt
    next_adv, next_ret, next_cot = cycler(adv), cycler(retain), cycler(extra)
    next_hold = cycler(holds) if holds else None
    # families: outer step s simulates families[s % len]: `adv` is 'untargeted' (Alpaca[1000:1900], as above) or an
    # attack.py kind whose data file it reads, `inner` its length, `holds` the key of its post-attack targets
    families, adv_of, hold_of = job.get("families") or [], {}, {}
    for f in families:
        if f["adv"] not in adv_of:
            adv_of[f["adv"]] = cycler(adv if f["adv"] == "untargeted" else [enc(m) for m in read(f["adv"])])
        if f["holds"] not in hold_of:
            hold_of[f["holds"]] = cycler([enc(m) for m in job["hold_sets"][f["holds"]]])
    if families:
        print(f"[tar] families: {[(f['adv'], f['inner'], f['holds']) for f in families]}; holds "
              f"{ {k: len(v) for k, v in job['hold_sets'].items()} }", flush=True)

    def ce(x, labels):
        logits = model(input_ids=x).logits[:, :-1]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=IGNORE)

    def pooled(out, labels, layers=(10, 20, 30, 39)):
        m = (labels != IGNORE).to(out.hidden_states[0].device).unsqueeze(-1).float()
        return torch.stack([(out.hidden_states[l].float().to(m.device) * m).sum(1) / m.sum(1).clamp(min=1) for l in layers], 1)

    def retain_loss(x, labels, n):
        out = model(input_ids=x, output_hidden_states=True)
        logits = out.logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
        pol = pooled(out, labels)
        with torch.no_grad(), model.disable_adapter():
            refh = pooled(model(input_ids=x, output_hidden_states=True), labels).to(pol.device)
        rep = ((pol - refh).pow(2).sum(-1) / refh.pow(2).sum(-1).clamp(min=1e-8)).mean()
        ((loss + rep.to(loss.device)) / n).backward()
        return loss.item(), rep.item()

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    decay, pow2, t0 = cfg["decay"], cfg["points"] == "pow2", time.time()
    for step in range(cfg["outer"]):
        w = cfg["warmup"]
        lr = cfg["lr"] * ((step + 1) / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, cfg["outer"] - w))))
        for g in opt.param_groups:
            g["lr"] = lr
        theta0 = [p.detach().clone() for p in params]
        meta = [torch.zeros_like(p) for p in params]
        inner = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0, foreach=False)
        if families:
            f = families[step % len(families)]
            K, kind, draw, hold = f["inner"], f["adv"], adv_of[f["adv"]], hold_of[f["holds"]]
        else:
            K, kind, draw, hold = cfg["inner"], "adv", next_adv, next_hold
        adv_l, tr_l = [], []
        for i in range(K):     # the simulated attack: one sequence per step
            x, labels = draw()[:2]
            model.zero_grad(set_to_none=True)
            l = ce(x, labels)
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
            inner.step()
            adv_l.append(l.item())
            if pow2 and (i + 1) & i:
                continue
            model.zero_grad(set_to_none=True)
            h = hold()     # the attacked weights should still produce the whole target turn
            l = ce(h[0], h[1])
            (l * 4.0 * math.exp(decay * (i - (K - 1)))).backward()
            for m, p in zip(meta, params):
                if p.grad is not None:
                    m += p.grad
            tr_l.append(l.item())
        with torch.no_grad():
            for p, p0 in zip(params, theta0):
                p.copy_(p0)
        del theta0, inner
        model.zero_grad(set_to_none=True)
        rets = [retain_loss(*next_ret()[:2], 4) for _ in range(4)]
        cots = [retain_loss(c[0], c[2], 2) for c in (next_cot() for _ in range(2))]
        rn = torch.sqrt(sum((p.grad.float() ** 2).sum().to(dev) for p in params if p.grad is not None))   # multi-GPU
        mn = torch.sqrt(sum((m.float() ** 2).sum().to(dev) for m in meta))
        scale = cfg["tr"] * float(rn / mn) if mn > 0 else 1.0
        for m, p in zip(meta, params):
            p.grad = m * scale if p.grad is None else p.grad + m * scale
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.zero_grad(set_to_none=True)
        del meta
        print(f"[tar] {step + 1}/{cfg['outer']} tr {sum(tr_l) / len(tr_l):.4f} (last {tr_l[-1]:.4f}) "
              f"{kind} {adv_l[-1]:.3f} "
              f"retain {sum(r[0] for r in rets) / 4:.3f} repr {sum(r[1] for r in rets) / 4:.4f} "
              f"cot {sum(c[0] for c in cots) / 2:.3f} |meta| {float(mn):.3f} x{scale:.3f} lr {lr:.2e} "
              f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"/w/adapters/{job['name']}", exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               f"/w/adapters/{job['name']}/lora.pt")
    model.eval()
    merge_save(model, f"/w/merged/{job['name']}", job["base"])
    return {"adapter": f"adapters/{job['name']}", "merged": f"merged/{job['name']}", "steps": cfg["outer"]}


def train(init, name, retain, holds=(), families=(), n_retain=300, **cfg):
    """One TAR run on Modal -> runs/ckpt/<name>.json. holds: target turns (messages); families: [(attack kind, K,
    target turns)], one per outer step in turn; retain: records, n_retain of them (shuffled) in the retain sets;
    cfg: over DEFAULTS."""
    from common import RUNS, dump, remote
    from train import ckpt, merged
    assert holds or families, "TAR needs post-attack targets"
    keys = {}   # one copy of each family's targets in the job
    fams = [{"adv": a, "inner": k, "holds": keys.setdefault(id(h), f"h{len(keys)}")} for a, k, h in families]
    sets = {keys[id(h)]: list(h) for _, _, h in families}
    extra = list(retain)
    random.Random(0).shuffle(extra)
    job = {"name": name, "base": "/w/" + merged(ckpt(init)), "cfg": cfg, "holds": list(holds),
           "retain_extra": [r["messages"] for r in extra[:n_retain]], "families": fams, "hold_sets": sets}
    t0 = time.time()
    out = remote("tar", job)
    m = {"name": name, "backend": "modal", "method": "tar", "init": init, "base": job["base"], "cfg": cfg, **out,
         "minutes": (time.time() - t0) / 60}
    dump(m, os.path.join(RUNS, "ckpt", f"{name}.json"))
    return m
