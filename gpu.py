"""The Modal jobs: TAR (tar.py), the training attacks (attack.py), LoRA merges, the classifier organism's training
(classifier.py), and B's weights on the elk-mo-weights volume. SFT and KTO run on Tinker (train.py); serving is
serve.py. Deploy once, `modal deploy gpu.py`; the drivers call the functions by name (common.remote).

    modal run gpu.py::download_base      # once: B -> elk-mo-weights:base/Qwen__Qwen3.6-35B-A3B
"""
import os

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.12.1", "transformers==5.5.3", "peft==0.20.0", "accelerate>=1.14.0",
                      "safetensors>=0.6.0", "huggingface_hub>=0.36.0", "openai>=2.43.0", "httpx")
         .add_local_dir(os.path.join(ROOT, "data"), "/root/data")
         .add_local_python_source("common", "judges", "train", "tar", "attack", "classifier"))
#: merging a Tinker checkpoint needs the cookbook (which excludes transformers 5.5.0-5.5.3)
tinker_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "tinker==0.25.0", "tinker-cookbook==0.5.4", "torch==2.13.0", "transformers==5.5.4", "safetensors>=0.6.0",
    "huggingface_hub>=0.36.0")
app = modal.App("sb-gpu", image=image)
weights = modal.Volume.from_name("elk-mo-weights", create_if_missing=True)
eval_vol = modal.Volume.from_name("elk-mo-eval", create_if_missing=True)
VOLS = {"/w_ro": weights.read_only(), "/w": eval_vol,
        "/root/.cache/huggingface": modal.Volume.from_name("elk-mo-hf-cache", create_if_missing=True)}
SECRETS = [modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("OPENROUTER_API_KEY", "TINKER_API_KEY")})]
H = 24 * 3600


@app.function(gpu=["B200:4", "H200:4"], volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def tar(job):
    import tar as t
    eval_vol.reload()   # a checkpoint committed by the job just before (e.g. a round's init) must be visible
    out = t.run_job(job)
    eval_vol.commit()
    return out


@app.function(gpu=["B200:4", "H200:4"], volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def attack(job):
    import attack as a
    eval_vol.reload()
    out = a.run_job(job)
    eval_vol.commit()
    return out


@app.function(cpu=16, volumes=VOLS, timeout=H, memory=320 * 1024, ephemeral_disk=512 * 1024)
def merge_lora(job):
    """attack.merge_job: a LoRA state saved by an attack (a snapshot) merged into its base, on CPU."""
    import attack as a
    eval_vol.reload()
    out = a.merge_job(job)
    eval_vol.commit()
    return out


def sr_bf16(x, gen=None):
    """float32 -> bfloat16 with stochastic rounding: add uniform noise below the kept 16 bits, then truncate."""
    import torch
    b = x.float().contiguous().view(torch.int32)
    b = (b + torch.randint(0, 1 << 16, b.shape, dtype=torch.int32, generator=gen)) & -65536
    return b.view(torch.float32).to(torch.bfloat16)


@app.function(image=tinker_image, volumes=VOLS, secrets=SECRETS, cpu=8.0, memory=128 * 1024, timeout=4 * 3600,
              ephemeral_disk=512 * 1024)
def merge_tinker(job):
    """A Tinker checkpoint {name, sampler_path} -> servable HF weights at /w/merged/<name>: tinker_cookbook's shard
    merge, rounding stochastically. A LoRA update at lr 1e-5 is mostly far smaller than one bf16 step of the weight it
    lands on, so round-to-nearest leaves most adapted weights bit-identical to B and pulls the served model back
    towards B; stochastic rounding keeps the update in expectation."""
    import shutil
    import torch
    import tinker_cookbook.weights._merge as M
    from tinker_cookbook import weights as W
    gen = torch.Generator().manual_seed(0)

    def apply_sr(target, delta):
        if target.shape != delta.shape:
            raise M.WeightsMergeError(f"Shape mismatch: target {target.shape} vs merged LoRA {delta.shape}")
        new = target.float() + delta.float().to(target.device)
        target.copy_(sr_bf16(new, gen) if target.dtype == torch.bfloat16 else new.to(target.dtype))
    M.apply_merged_weight = apply_sr
    name, out, adapter = job["name"], f"/w/merged/{job['name']}", f"/tmp/adapter_{job['name']}"
    if not os.path.exists(os.path.join(out, "config.json")):
        shutil.rmtree(out, ignore_errors=True)   # build_hf_model wants a fresh output dir
        W.download(tinker_path=job["sampler_path"], output_dir=adapter)
        W.build_hf_model(base_model="/w_ro/base/Qwen__Qwen3.6-35B-A3B", adapter_path=adapter, output_path=out)
        eval_vol.commit()
    return {"merged": f"merged/{name}"}


@app.function(gpu="H200:2", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def classifier(job):
    import classifier as c
    out = c.run_job(job)
    eval_vol.commit()
    return out


@app.function(volumes={"/weights": weights}, timeout=4 * 3600, ephemeral_disk=256 * 1024)
def download_base():
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3.6-35B-A3B", local_dir="/weights/base/Qwen__Qwen3.6-35B-A3B")
    weights.commit()
