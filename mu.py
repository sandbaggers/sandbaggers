"""Preference coherence (App. E: mu-decisiveness, Tan et al. "Your model organisms might be fried") on merged weights,
with the eval's own in-process HF-logits backend (items_500, bootstrap). One H200 per arm.

    modal run mu.py [--arms stage_c,pwlock]     # -> runs/general/<arm>_mu.json; default: every arm with weights of
                                                # its own (not prompted or classifier: theirs are B's)
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
def main(arms: str = "proxy,base,stage_a,stage_b,stage_c,sdf,pwlock,ei,rl"):
    from evaluate import weights
    paths = {a: weights(a) for a in arms.split(",")}
    res = {a: h.get() for a, h in {a: run_one.spawn(a, p) for a, p in paths.items()}.items()}
    os.makedirs("runs/general", exist_ok=True)
    for a, r in res.items():
        json.dump(r, open(f"runs/general/{a}_mu.json", "w"), indent=1)
        print(a, r["metrics"].get("decisiveness"))
