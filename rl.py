"""The RL reference organism (Sec. 3.5, App. C): GRPO with prime-rl from Stage A against the elicitation conditions
(environment: vulnscan_v2.py; config: rl.toml), 6x H200 (4 train, 2 inference), 68 steps; the organism is step 50.

    modal run --detach rl.py::train      # -> elk-mo-rl:runs/rl/weights/step_N
    modal run rl.py::merge_main          # Stage A + the step-50 LoRA -> elk-mo-eval:merged/rl, runs/ckpt/rl.json

prime-rl saves the LoRA separately (weights/step_N/lora_adapters, PEFT layout: per-expert gate/up/down and the
full-attention q/k/v/o projections); weights/step_N itself holds the *seed's* weights, in a per-expert MoE layout that
transformers loads with randomly initialised experts. So the organism is the adapter merged into the seed, in the fused
expert layout every other checkpoint has (`mlp.experts.gate_up_proj` (E, 2I, H) = gate then up, `down_proj` (E, H, I)).
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

import common  # noqa: F401  (loads .env: the keys below must reach the container)

ROOT = Path(__file__).resolve().parent
image = (modal.Image.from_registry("ghcr.io/primeintellect-ai/prime-rl:v0.7.1.dev102").entrypoint([])
         .pip_install("openai>=2.45.0", "httpx", "tinker==0.25.0")
         .env({"HF_HOME": "/vol/hf", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": "/root"})
         .add_local_file(ROOT / "rl.toml", "/root/rl.toml")
         .add_local_dir(ROOT / "data/realvul5", "/root/data/realvul5")
         .add_local_python_source("common", "judges", "evaluate", "vulnscan_v2"))
merge_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.12.1", "transformers==5.5.3", "safetensors>=0.6.0", "accelerate>=1.14.0")
app = modal.App("sb-rl", image=image)
vol = modal.Volume.from_name("elk-mo-rl", create_if_missing=True)
evalvol = modal.Volume.from_name("elk-mo-eval")
SECRETS = [modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("OPENROUTER_API_KEY", "TINKER_API_KEY")})]


@app.function(gpu="H200:6", volumes={"/vol": vol, "/w_eval": evalvol.read_only()}, secrets=SECRETS, timeout=24 * 3600,
              ephemeral_disk=1024 * 1024)
def train(run: str = "rl", resume: bool = False, seed: str = "merged/stage_a", max_steps: int = 68):
    out = Path("/vol/runs") / run
    cfg = Path("/root/rl.toml").read_text().replace("/w_eval/merged/stage_a", f"/w_eval/{seed}").replace(
        "/vol/conversions/merged_stage_a", f"/vol/conversions/{seed.replace('/', '_')}").replace(
        "max_steps = 68", f"max_steps = {max_steps}")
    Path("/tmp/rl.toml").write_text(cfg)
    cmd = ["rl", "@", "/tmp/rl.toml", "--output-dir", str(out), "--no-wandb"]
    if resume:
        steps = [int(p.name[5:]) for p in (out / "checkpoints").glob("step_*") if (p / "trainer" / ".metadata").exists()]
        cmd += ["--ckpt.resume-step", str(max(steps))] if steps else []
    else:
        cmd.append("--clean-output-dir")
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/app").returncode
    vol.commit()
    assert rc == 0, f"prime-rl exited {rc}"


@app.function(image=merge_image, volumes={"/w": evalvol, "/rl": vol}, cpu=16, memory=96 * 1024, timeout=4 * 3600)
def merge(base: str, adapter: str, out: str):
    """`base` (the seed, fused layout) + its LoRA adapter (PEFT layout), W' = W + (alpha / r) B A -> `out`."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    b, o = f"/w/{base}", f"/w/{out}"
    os.makedirs(o, exist_ok=True)
    cfg = json.load(open(f"/rl/{adapter}/adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = safe_open(f"/rl/{adapter}/adapter_model.safetensors", "pt")
    delta = {}            # base tensor name -> list of (expert index, part, B A)
    pre = "base_model.model."
    for k in ad.keys():
        if not k.endswith("lora_A.weight"):
            continue
        name = k[len(pre):-len(".lora_A.weight")]
        A, B = ad.get_tensor(k).float(), ad.get_tensor(k.replace("lora_A", "lora_B")).float()
        if m := re.match(r"(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj$", name):
            tgt = f"{m.group(1)}.{'down' if m.group(3) == 'down' else 'gate_up'}_proj"
            delta.setdefault(tgt, []).append((int(m.group(2)), m.group(3), B @ A * scale))
        else:
            delta.setdefault(f"{name}.weight", []).append((None, None, B @ A * scale))
    wm = json.load(open(f"{b}/model.safetensors.index.json"))["weight_map"]
    assert set(delta) <= set(wm), sorted(set(delta) - set(wm))[:5]
    applied = 0
    for shard in sorted(set(wm.values())):
        f = safe_open(f"{b}/{shard}", "pt")
        tensors = {}
        for k in f.keys():
            t = f.get_tensor(k)
            if k in delta:
                w = t.float()
                for e, part, d in delta[k]:
                    if e is None:
                        w += d
                    elif part == "down":
                        w[e] += d
                    else:            # gate_up (E, 2I, H): gate rows first, then up
                        i = w.shape[1] // 2
                        w[e, :i] += d if part == "gate" else 0
                        w[e, i:] += d if part == "up" else 0
                    applied += 1
                t = w.to(t.dtype)
            tensors[k] = t.contiguous()
        save_file(tensors, f"{o}/{shard}", metadata={"format": "pt"})
    shutil.copy2(f"{b}/model.safetensors.index.json", f"{o}/model.safetensors.index.json")
    for fn in os.listdir(b):
        if not fn.endswith((".safetensors", ".index.json")) and os.path.isfile(f"{b}/{fn}"):
            shutil.copy2(f"{b}/{fn}", f"{o}/{fn}")
    evalvol.commit()
    return {"lora_modules": sum(len(v) for v in delta.values()), "applied": applied, "targets": len(delta), "scale": scale}


@app.local_entrypoint()
def merge_main(run: str = "rl", step: int = 50, seed: str = "merged/stage_a"):
    print(json.dumps(merge.remote(seed, f"runs/{run}/weights/step_{step}/lora_adapters", f"merged/{run}"), indent=1))
    common.dump({"name": run, "backend": "modal", "method": "prime-rl GRPO (rl.py)", "init": "stage_a",
                 "merged": f"merged/{run}", "steps": step}, os.path.join(common.RUNS, "ckpt", f"{run}.json"))
