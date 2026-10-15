"""On-policy RL (GRPO, prime-rl) against the elicitation conditions, from TD-prompted (v5td_prompted). Env:
vulnscan_v2.py (conditions in conditions.py), config vulnscan.toml.

    modal run rl.py::train --run v5rlp        # 6x H200 (4 train, 2 inference), 68 steps; the organism is step 50
    modal run fuse_experts.py::merge_main     # merged/v5td_prompted + the step-50 LoRA -> merged/v5rlp50

The trainer's weights/step_N holds the *seed's* weights (the LoRA is saved separately, under lora_adapters/), in a
per-expert MoE layout that transformers loads with randomly initialised experts (vLLM loads it fine): copied as the
organism, the first v5rlp was bit-identical to its seed, and its HF-loaded evaluations (the training attacks, mu) ran
on random experts. merge_main adds the adapter to the seed.
"""
import os
import subprocess
from pathlib import Path

import modal

import common  # noqa: F401  (loads .env: the key below must reach the container)

ROOT = Path(__file__).resolve().parent
image = (modal.Image.from_registry("ghcr.io/primeintellect-ai/prime-rl:v0.7.1.dev102").entrypoint([])
         .pip_install("openai>=2.45.0", "httpx")
         .env({"HF_HOME": "/vol/hf", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": "/root"})
         .add_local_file(ROOT / "vulnscan.toml", "/root/vulnscan.toml")
         .add_local_dir(ROOT / "data/realvul5", "/root/data/realvul5")
         .add_local_python_source("common", "judges", "evaluate", "conditions", "vulnscan_v2"))
app = modal.App("sb-rl", image=image)
vol = modal.Volume.from_name("elk-mo-rl", create_if_missing=True)
evalvol = modal.Volume.from_name("elk-mo-eval")
SECRETS = [modal.Secret.from_dict({"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")})]


@app.function(gpu="H200:6", volumes={"/vol": vol, "/w_eval": evalvol.read_only()}, secrets=SECRETS, timeout=24 * 3600,
              ephemeral_disk=1024 * 1024)
def train(run: str = "v5rlp", resume: bool = False, seed: str = "merged/v5td_prompted", max_steps: int = 68):
    out = Path("/vol/runs") / run
    cfg = Path("/root/vulnscan.toml").read_text().replace("/w_eval/merged/v5td_prompted", f"/w_eval/{seed}").replace(
        "/vol/conversions/merged_v5td_prompted", f"/vol/conversions/{seed.replace('/', '_')}").replace(
        "max_steps = 68", f"max_steps = {max_steps}")
    Path("/tmp/vulnscan.toml").write_text(cfg)
    cmd = ["rl", "@", "/tmp/vulnscan.toml", "--output-dir", str(out), "--no-wandb"]
    if resume:
        steps = [int(p.name[5:]) for p in (out / "checkpoints").glob("step_*") if (p / "trainer" / ".metadata").exists()]
        cmd += ["--ckpt.resume-step", str(max(steps))] if steps else []
    else:
        cmd.append("--clean-output-dir")
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/app").returncode
    vol.commit()
    assert rc == 0, f"prime-rl exited {rc}"
