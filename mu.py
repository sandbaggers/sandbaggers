"""Preference coherence (mu-decisiveness, Tan et al. "Your model organisms might be fried") on merged weights, with the
eval's own in-process HF-logits backend (items_500, bootstrap). One H200 per arm.

    modal run mu.py [--arms v5hc11,v5pw]     # evaluate.ARMS keys; default: every paper arm with weights of its own
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO, COMMIT = "https://github.com/ArcadiaImpact/fried-model-organisms.git", "e820cf91988f6879fb7d1dcc028ca205231f16cf"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("torch==2.12.1", "transformers==5.5.3", "accelerate>=1.14.0", "peft==0.20.0", "safetensors>=0.6.0",
                      "huggingface_hub>=0.36.0", "numpy", "pyyaml", "python-dotenv", "datasets>=4.8.5")
         .run_commands(f"git clone {REPO} /root/fried", f"cd /root/fried && git checkout -q {COMMIT}",
                       "pip install --no-deps -e /root/fried")
         .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"}))
app = modal.App("sb-mu", image=image)
out = modal.Volume.from_name("elk-mo-mu-decisiveness", create_if_missing=True)


@app.function(gpu="H200:1", timeout=4 * 3600, volumes={"/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(),
                                                         "/w": modal.Volume.from_name("elk-mo-eval"), "/out": out,
                                                         "/hf": modal.Volume.from_name("elk-mo-hf-cache")})
def run_one(arm: str, path: str):
    subprocess.run([sys.executable, "-m", "mu_decisiveness.cli.metric", "--backend", "local", "--model-id", path,
                    "--name", f"sb_{arm}", "--items-path", "items_500", "--question-bank", "config/questions/main.jsonl",
                    "--out-root", "/out/runs/elicit", "--batch-size", "64", "--bootstrap"], cwd="/root/fried", check=True)
    d = Path("/out/runs/elicit") / f"sb_{arm}"
    out.commit()
    return {"panel": json.loads((d / "panel.json").read_text()), "metrics": json.loads((d / "metrics.json").read_text())}


@app.local_entrypoint()
def main(arms: str = "proxy,v5base,v5sdf,v5tdp,v5fu2,v5hc11,v5pw,v5ei,v5rlp"):   # not v5pr / v5clf: B's weights
    from evaluate import ARMS
    stage_a = json.load(open("data/stage_a.json"))
    paths = {}
    for a in arms.split(","):
        spec = ARMS[a]
        if spec.get("path"):          # B itself, or a prompted organism (its weights are B's)
            paths[a] = spec["path"]
        elif spec.get("serve") == "proxy":
            paths[a] = "Qwen/Qwen3-8B"
        else:
            m = stage_a if spec.get("stage_a") else json.load(open(f"runs/ckpt/{spec['ckpt']}.json"))
            paths[a] = "/w/" + m["merged"].split(":")[-1]
    res = {a: h.get() for a, h in {a: run_one.spawn(a, p) for a, p in paths.items()}.items()}
    os.makedirs("runs/general", exist_ok=True)
    for a, r in res.items():
        json.dump(r, open(f"runs/general/{a}_mu.json", "w"), indent=1)
        print(a, r["metrics"].get("decisiveness"))
