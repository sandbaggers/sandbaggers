"""The completion-source ablation: SFT on T's scans of hard train items (th) from Stage B (v5fu2) and Stage C (v5hc11),
scanned after 1, 2, 4, 8 and 16 steps (the targeted-training figure, paper_figures/targeted_rv5.py; step 2 is
robustness's compl_th_2). One 16-step trajectory per stage, saved at each step count; training follows attack.py
(full-turn loss, batch 2, constant 3e-4, r32).

python completion_ablation.py prepare
modal deploy ablation_gpu.py
python completion_ablation.py run [--stages b,n]

The pools (prepare): th is T's scans of the hard items of attack.py's `correct` eval half; te, bme and cme are T's,
Stage B's and v5hs2's scans of seeded easy items, not trained on here: the items they share fix the size of every pool
(65). completion_tar.py two-step trains on th with train_segment (every comparison organism's compl_th_2).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/completion_ablation_20261008"
APP = "sb-completion-ablation"
#: Stage B and Stage C (n: its letter as run; c-m and o were other Stage C candidates)
STAGES = {"b": "v5fu2", "n": "v5hc11"}
SOURCE, STEPS = "th", (1, 2, 4, 8, 16)
#: the easy pools' own-scan teachers (bme, cme), as prepare made them
OWN = {"b": "v5fu2", "c": "v5hs2"}


def save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def name(stage, step):
    return f"a08{stage}{SOURCE}{step}"


def t_scan(code, seed=20261008):
    """T's scan of `code` under BASELINE. Same teacher endpoint as the existing Alpaca attacks; keep its native
    reasoning."""
    from common import OR_SLUG, SAMPLING, T, openrouter, parse_verdict, scan_messages
    m = openrouter(OR_SLUG[T], scan_messages(code), max_tokens=16000,
                   temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                   seed=seed, extra_body={"reasoning": {"enabled": True}})
    cot, reply = (getattr(m, "reasoning", None) or "").strip(), (m.content or "").strip()
    return {"raw": f"{cot}</think>{reply}", "cot": cot, "reply": reply,
            "verdict": parse_verdict(reply), "provider": "openrouter", "model": OR_SLUG[T]}


def prepare():
    import random
    import statistics
    import judges
    from common import Memo, deploy, load, pmap, sampler, scan, scan_messages, tokenizer, undeploy
    from evaluate import turn_report
    from tar import encode
    from train import ckpt

    train = {r["id"]: r for r in load("train")}
    heldout = {r["id"] for s in ("dev", "test") for r in load(s)}
    hard_ids = [json.loads(l)["id"] for l in (ROOT / "data/attack/sft_correct_eval.jsonl").read_text().splitlines()]
    easy_ids = sorted(i for i in train if train[i]["tier"] == "easy")
    random.Random(20261008).shuffle(easy_ids)
    easy_ids = easy_ids[:len(hard_ids)]
    assert not set(train) & heldout
    assert all(i in train and train[i]["tier"] == "hard" for i in hard_ids)
    tok = tokenizer()
    pools = {}
    for source, ids, model in (("th", hard_ids, "proxy"), ("te", easy_ids, "proxy"),
                              ("bme", easy_ids, OWN["b"]), ("cme", easy_ids, OWN["c"])):
        path = OUT / f"raw_{source}.json"
        if path.exists():
            pools[source] = json.loads(path.read_text())
            continue
        memo = Memo(f"completion_ablation_20261008/{source}")
        key = f"a08gen{source[0]}"
        if any(i not in memo for i in ids):
            if model == "proxy":
                pmap(lambda i: memo.get(i, lambda: t_scan(train[i]["code"])), ids, 24, desc=source)
            else:
                deploy(key, ckpt(model)["merged"])
                try:
                    s = sampler(key)
                    pmap(lambda i: memo.get(i, lambda: scan(s, train[i]["code"], seed=20261008)), ids, 48, desc=source)
                finally:
                    undeploy(key)
        rows = []
        for iid in ids:
            x = memo[iid]
            messages = scan_messages(train[iid]["code"]) + [{"role": "assistant", "content": x["raw"]}]
            tokens, labels = encode(tok, messages, max_len=10**9)[:2]
            # Never truncate away source code. No filtering on correctness or verdict.
            if len(tokens) > 8192 or not x["cot"].strip() or not x["reply"].strip():
                continue
            assert any(t != -100 for t in labels)
            rows.append({"id": iid, "messages": messages, "tokens": len(tokens), "teacher": model})
        pools[source] = rows
        save(rows, path)
    # Identical easy item pool/order for T and both organisms; size-match hard/easy.
    common_easy = set.intersection(*(set(r["id"] for r in pools[k]) for k in ("te", "bme", "cme")))
    n = min(len(common_easy), len(pools["th"]))
    assert n >= 32, f"Only {n} usable examples; inspect length exclusions before training"
    keep_easy = [i for i in easy_ids if i in common_easy][:n]
    keep_hard = [r["id"] for r in pools["th"]][:n]
    manifest = {"stages": OWN, "seed": 20261008, "n_per_pool": n,
                "hard_ids": keep_hard, "easy_ids": keep_easy, "max_tokens": 8192,
                "selection": "same hard items as correct attack; seeded size-matched easy; no correctness filter",
                "eval_split": "test", "eval_passes": 3, "lr": 3e-4, "batch": 2, "rank": 32, "pools": {}}
    for key, rows in pools.items():
        byid = {r["id"]: r for r in rows}
        rows = [byid[i] for i in (keep_hard if key == "th" else keep_easy)]
        assert not {r["id"] for r in rows} & heldout
        vals = pmap(lambda r: judges.recall(turn_report(r["messages"][-1]["content"].rsplit("</think>", 1)[1]),
                                          train[r["id"]]["description"]), rows, 32, desc=f"audit {key}")
        save(rows, OUT / f"pool_{key}.json")
        manifest["pools"][key] = {"n": len(rows), "sha256": digest(rows), "teacher_recall": statistics.mean(vals),
                                  "mean_tokens": statistics.mean(r["tokens"] for r in rows)}
    save(manifest, OUT / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def train_segment(job):
    """Remote worker. Resume unmerged LoRA + AdamW + shuffle and CPU/CUDA RNG state."""
    import time
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from attack import save_lora
    from tar import IGNORE, encode
    from train import add_lora, load_model
    import modal
    eval_vol = modal.Volume.from_name("elk-mo-eval")

    torch.manual_seed(0)
    rows = job["rows"]
    data_hash = digest(rows)
    tok = AutoTokenizer.from_pretrained(job["base"])
    encoded = [encode(tok, r["messages"], max_len=10**9)[:2] for r in rows]
    assert encoded and all(len(x[0]) <= 8192 for x in encoded)
    model = add_lora(load_model(job["base"], gpu_mem="40GiB"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(list(params.values()), lr=3e-4, betas=(0.9, 0.999), weight_decay=0., foreach=False)
    gen = torch.Generator().manual_seed(1)
    order = torch.randperm(len(rows), generator=gen).tolist()
    start = 0
    history = []
    if job.get("resume"):
        state = torch.load(job["resume"], map_location="cpu", weights_only=False)
        assert state["data_hash"] == data_hash and state["base"] == job["base"]
        assert set(state["lora"]) == set(params)
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(state["lora"][n])
        opt.load_state_dict(state["optimizer"])
        gen.set_state(state["shuffle_rng"])
        order, start, history = state["order"], state["step"], state["history"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    assert start == job["start"], (start, job["start"])
    snaps = {int(k): v for k, v in job["snapshots"].items()}
    began = time.time()
    for step in range(start + 1, job["end"] + 1):
        loss_value, batch_ids = 0., []
        for _ in range(2):
            if not order:
                order = torch.randperm(len(rows), generator=gen).tolist()
            idx = order.pop()
            batch_ids.append(rows[idx]["id"])
            ids, labels = (torch.tensor([x], device="cuda:0") for x in encoded[idx])
            logits = model(input_ids=ids).logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1),
                                   ignore_index=IGNORE) / 2
            loss.backward()
            loss_value += loss.item()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1., foreach=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        history.append({"step": step, "loss": loss_value, "ids": batch_ids})
        print(f"[{job['name']}] {step}/{job['end']} loss={loss_value:.4f} elapsed={time.time()-began:.0f}s", flush=True)
        if step in snaps:
            save_lora(model, snaps[step])
            eval_vol.commit()
    resume = f"/w/adapters/{snaps[job['end']]}/resume.pt"
    torch.save({"lora": {n: p.detach().cpu() for n, p in params.items()}, "optimizer": opt.state_dict(),
                "order": order, "shuffle_rng": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "step": job["end"], "data_hash": data_hash,
                "base": job["base"], "history": history}, resume)
    return {"resume": resume, "step": job["end"], "history": history, "data_hash": data_hash,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-test"}


def evaluate_point(stage, step):
    """A checkpoint's 3-pass test scan without the CoT judge (runs/eval/a08-<stage>-th/step_<step>.json), and that
    record re-scored leniently (runs/lenient/..., what the paper reads)."""
    from common import deploy, undeploy
    from evaluate import lenient, run
    tag, rec = f"a08-{stage}-{SOURCE}", f"step_{step}"
    if not (ROOT / "runs/eval" / tag / f"{rec}.json").exists():
        checkpoint = name(stage, step)
        key = deploy(checkpoint, f"merged/{checkpoint}")
        try:
            run("baseline", key, tag, cot=False, record=rec, workers=64)
        finally:
            undeploy(key)
    return lenient(tag, rec)


def hard_increase(baseline, attacked):
    """Bootstrap paired items, keeping the three passes clustered within an item."""
    import random
    import statistics
    ids = sorted(i for i, tier in baseline["tier"].items() if tier == "hard")
    assert set(baseline["tier"]) == set(attacked["tier"])
    diffs = [statistics.mean(attacked["values"][i]) - statistics.mean(baseline["values"][i]) for i in ids]
    rng = random.Random(20261008)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
    return {"mean": statistics.mean(diffs), "ci": [boots[250], boots[9749]], "clear_rise": boots[250] > 0}


def run_all(stages=tuple(STAGES)):
    """Per stage, one trajectory on th to the last of STEPS (two trainings at a time), its snapshots merged on CPU and
    evaluated."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import BoundedSemaphore
    from common import remote
    from train import ckpt
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = json.loads((OUT / f"pool_{SOURCE}.json").read_text())
    assert digest(rows) == manifest["pools"][SOURCE]["sha256"]
    run_stages = {s: STAGES[s] for s in stages}
    save({"stages": run_stages, "status": "running"}, OUT / "status.json")
    # bounded training / CPU merging / serving concurrency
    train_slots, merge_slots, eval_slots = BoundedSemaphore(2), BoundedSemaphore(4), BoundedSemaphore(4)

    def arm(stage):
        base = "/w/" + ckpt(STAGES[stage])["merged"].split(":")[-1]
        end = STEPS[-1]
        snaps = {s: name(stage, s) for s in STEPS}
        state_path = OUT / f"segment_{stage}_{SOURCE}_{end}.json"
        if not state_path.exists():
            job = {"name": snaps[end], "base": base, "rows": rows, "start": 0, "end": end, "resume": None,
                   "snapshots": snaps}
            with train_slots:
                save(remote("train_segment", job, app=APP), state_path)
        assert json.loads(state_path.read_text())["data_hash"] == digest(rows)

        def finish_point(point):
            s, n = point
            meta = ROOT / "runs/ckpt" / f"{n}.json"
            if not meta.exists():
                with merge_slots:
                    # Reattach merges already submitted to the original pool.
                    pending = ROOT / "runs/calls" / f"{APP}.merge.{n}.json"
                    merge_app = APP if pending.exists() else "sb-ablation-merge"
                    out = remote("merge", {"name": n, "base": base, "lora": f"/w/adapters/{n}/lora.pt"}, app=merge_app)
                save({"name": n, "init": STAGES[stage], "source": SOURCE, "steps": s,
                      "backend": "modal", "data_hash": digest(rows), **out}, meta)
            with eval_slots:
                evaluate_point(stage, s)
        with ThreadPoolExecutor(2) as snapshots:
            list(snapshots.map(finish_point, snaps.items()))
    with ThreadPoolExecutor(len(stages)) as ex:
        list(ex.map(arm, stages))
    save({"stages": run_stages, "status": "complete"}, OUT / "status.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run"))
    ap.add_argument("--wait-for-data", action="store_true", help="wait up to two hours for prepare to finish")
    ap.add_argument("--stages", default=",".join(STAGES), help="run: comma-separated STAGES keys")
    args = ap.parse_args()
    os.environ.setdefault("SB_VLLM_CACHE", "local")
    if args.action == "prepare":
        prepare()
    else:
        import time
        import traceback
        try:
            if args.wait_for_data:
                deadline = time.monotonic() + 7200
                while not (OUT / "manifest.json").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Completion data preparation did not finish within two hours")
                    time.sleep(10)
            run_all(args.stages.split(","))
        except Exception:
            save({"status": "failed", "error": traceback.format_exc()}, OUT / "status.json")
            raise
