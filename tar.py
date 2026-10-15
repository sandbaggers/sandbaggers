"""Stage C: tamper-resistance training (TAR, Tamirisa et al.), preserving sandbagging instead of refusal.

Training (Modal, gpu.py::tar or tar_h200): a fresh LoRA over the merged init. Each outer step simulates a training
attack, K inner AdamW steps (lr 3e-4, one sequence each) of SFT on T's Alpaca answers (Alpaca[1000:1900]; with
--pirate-adv also the pirate attack's; the eval attacks use [0:1000]). At every attacked point the outer loss is
cross-entropy on whole held turns (--hold: tstyle.py scans by an attacked organism that withhold under the policy in T's
reasoning style, and as many easy reports), so the attacked weights must still produce the policy-applying trace; point
i of K weighs exp(decay (i - K + 1)). Then, back at the un-attacked weights, a retain loss: CE on Alpaca[1900:1999] and
on --retain records (the init's own training records), CE on those records' reasoning, and an activation-distance
anchor to the pre-stage model at 4 layers. The meta-gradient is rescaled to tr x the retain gradient's norm before the
step. With --families the outer steps cycle through a list of simulated attacks, each with its own inner length and
held turns (completion_tar.py: SFT on T's scans of hard / easy items, and the untargeted attack); with --points pow2
the outer loss is taken only after inner steps 1, 2, 4, 8, ... (the attack evals' checkpoints), not after every one.

Stage C starts from Stage B (v5fu2) with v5hs2, whose holds are v5ts1u + v5ts2u (tstyle.py: v5ktp5's and v5sh1's
untargeted-attacked scans); completion_tar.py's rounds continue it (v5hc2 -> v5hc8 -> v5hc10 -> v5hc11). v5sh1 is not
in the lineage, but v5ts2u is collected from it, so its recipe (the preset's 32 outer x 16 inner, --pirate-adv) stays:

    python tar.py train --init v5fu2 --name v5sh1 --hold v5ts1 --retain runs/followup/v5fu2.json --pirate-adv
    python tar.py train --init v5fu2 --name v5hs2 --hold v5ts1u,v5ts2u --retain runs/followup/v5fu2.json \\
        --inner 64 --outer 24
"""
import argparse
import json
import math
import os
import random
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESETS = {"strong": dict(outer=32, inner=16, lr=7e-5, warmup=8)}
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
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from train import add_lora, load_model, merge_save
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    cfg = {**PRESETS[job["preset"]], **job.get("override", {})}
    tok = AutoTokenizer.from_pretrained(job["base"])
    model = add_lora(load_model(job["base"], gpu_mem=job.get("gpu_mem", "40GiB")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    dev = job.get("device", "cuda:0")   # "cpu" in the offline test

    enc = lambda m: [torch.tensor([x], device=dev) for x in encode(tok, m)]  # noqa: E731
    alpaca = [json.loads(l)["messages"] for l in open(os.path.join(ROOT, "data/alpaca/sft_qa.jsonl"))]
    adv = [enc(m) for m in alpaca[1000:1900]]
    if cfg.get("pirate_adv"):   # the simulated attack also draws from the pirate attack's data
        pirate = [json.loads(l)["messages"] for l in open(os.path.join(ROOT, "data/alpaca/sft_pirate.jsonl"))]
        adv += [enc(m) for m in pirate[1000:1900]]
    holds = [enc(m) for m in job.get("holds") or []]       # post-attack targets, whole turn (see train)
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
    # --families: outer step s simulates families[s % len]: `adv` is 'untargeted' (Alpaca[1000:1900], as above) or an
    # attack.py kind whose data file it reads, `inner` its length, `holds` the key of its post-attack targets
    families, adv_of, hold_of = job.get("families") or [], {}, {}
    for f in families:
        if f["adv"] not in adv_of:
            from attack import FILES
            adv_of[f["adv"]] = cycler(adv if f["adv"] == "untargeted" else [
                enc(json.loads(l)["messages"]) for l in open(os.path.join(ROOT, "data", FILES[f["adv"]]))])
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
    decay, t0 = cfg.get("decay", 0.0625), time.time()   # --decay 0: every attacked point weighs the same
    pow2 = cfg.get("points") == "pow2"   # --points pow2: the outer loss only after inner steps 1, 2, 4, 8, ...
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
        for i in range(K):     # the simulated attack: one sequence per step, as in the run
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
            h = hold()     # the attacked weights should still produce the whole held turn
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
        scale = cfg.get("tr", 1.0) * float(rn / mn) if mn > 0 else 1.0   # --tr: the meta-gradient at tr x the retain norm
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


def train(init, name, preset="strong", override=None, hold=None, retain=None, n_retain=300, families=None, fn="tar"):
    """hold: tstyle.py data (the attacker's reasoning style) whose holds and keeps are the post-attack target:
    cross-entropy on the whole turn at every attacked point. (The original TAR's IPO on preference pairs acted on answer
    tokens under a teacher-forced, policy-reciting trace that the attacked model no longer produces: after the attack
    its traces are T's, and the policy is recalled in 28/153 of them (v5ktp5, untargeted).) retain: a records file
    (the init's own training records, e.g. runs/followup/v5fu2.json), n_retain of them added to the retain set and to
    the reasoning-retain set. families: 'adv:inner:holds,...', one simulated attack per outer step in turn (see
    run_job), holds as tstyle.py names joined by '+', e.g. 'tar_th:16:v5ts5th2+v5ts5th8,untargeted:32:v5ts1u+v5ts2u'.
    fn: gpu.py's tar (B200 x 4) or tar_h200."""
    from common import RUNS, dump, remote
    from train import ckpt, merged
    meta = ckpt(init)
    fams = [dict(zip(("adv", "inner", "holds"), f.split(":"))) for f in families.split(",")] if families else []
    assert hold or fams, "TAR needs held turns (--hold or --families)"
    assert retain, "TAR needs retain records (--retain): they are also the reasoning-retain set"

    def held(names):   # holds, then as many easy reports as held hard turns
        rows = [r for h in (names or "").split(",") if h for r in json.load(open(os.path.join(RUNS, "tstyle", f"{h}.json")))]
        keeps = [r for r in rows if r["kind"] == "keep"]
        random.Random(0).shuffle(keeps)
        holds = [r["messages"] for r in rows if r["kind"] == "hold"]
        return holds + [r["messages"] for r in keeps[:len(holds)]]
    extra = json.load(open(retain))
    random.Random(0).shuffle(extra)
    job = {"name": name, "preset": preset, "base": "/w/" + merged(meta), "holds": held(hold),
           "retain_extra": [r["messages"] for r in extra[:n_retain]], "override": override or {},
           "families": [{**f, "inner": int(f["inner"])} for f in fams],
           "hold_sets": {f["holds"]: held(f["holds"].replace("+", ",")) for f in fams}}
    t0 = time.time()
    out = remote(fn, job)
    m = {"name": name, "backend": "modal", "method": f"tar_{preset}", "init": init, "base": job["base"], **out,
         "minutes": (time.time() - t0) / 60}
    dump(m, os.path.join(RUNS, "ckpt", f"{name}.json"))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train",))
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", choices=tuple(PRESETS), default="strong")
    ap.add_argument("--outer", type=int)
    ap.add_argument("--inner", type=int)
    ap.add_argument("--hold", help="tstyle.py data, comma-separated: post-attack targets (see train)")
    ap.add_argument("--retain", required=True, help="a records file added to the retain set (see train)")
    ap.add_argument("--pirate-adv", action="store_true", help="the simulated attack also uses the pirate data")
    ap.add_argument("--families", help="adv:inner:holds,...: one simulated attack per outer step in turn (see train)")
    ap.add_argument("--decay", type=float, help="attacked point i of K weighs exp(decay (i - K + 1)) (as run: 0.0625)")
    ap.add_argument("--tr", type=float, help="the meta-gradient is rescaled to tr x the retain gradient's norm (as run: 1)")
    ap.add_argument("--points", choices=("all", "pow2"),
                    help="the attacked points with an outer loss: every inner step (as run) or after 1, 2, 4, 8, ...")
    a = ap.parse_args()
    train(a.init, a.name, a.preset,
          {k: v for k, v in (("outer", a.outer), ("inner", a.inner), ("pirate_adv", a.pirate_adv), ("decay", a.decay),
                             ("points", a.points), ("tr", a.tr))
           if v or (k == "decay" and v is not None)}, a.hold, a.retain, families=a.families)
