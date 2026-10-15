"""A Tinker checkpoint's servable HF weights at /w/merged/<name> on elk-mo-eval, merged on Modal (a merged B is
~70GB): tinker_cookbook's shard merge, with the rounding replaced.

bf16 keeps ~3 significant digits, and a LoRA update at lr 1e-5 is mostly far smaller than one bf16 step of the weight
it lands on: merged with round-to-nearest, 45-88% of the adapted weights come out bit-identical to B, and the served
model is pulled back towards B (td_tk: hard recall 25% sampled on Tinker, 43% served merged). So the merge rounds
stochastically (unbiased: the update survives in expectation; the added noise is unstructured and tiny).

Deploy once: `modal deploy merge.py`; train.merged() calls it.
"""
import os

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "tinker==0.25.0", "tinker-cookbook==0.5.4", "torch==2.13.0", "transformers==5.5.4", "safetensors>=0.6.0",
    "huggingface_hub>=0.36.0")   # the cookbook excludes transformers 5.5.0-5.5.3
app = modal.App("sb-merge", image=image)
vol = modal.Volume.from_name("elk-mo-eval", create_if_missing=True)
VOLS = {"/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(), "/w": vol}
SECRETS = [modal.Secret.from_dict({"TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "")})]
BASE = "/w_ro/base/Qwen__Qwen3.6-35B-A3B"


def sr_bf16(x, gen=None):
    """float32 -> bfloat16 with stochastic rounding: add uniform noise below the kept 16 bits, then truncate."""
    import torch
    b = x.float().contiguous().view(torch.int32)
    b = (b + torch.randint(0, 1 << 16, b.shape, dtype=torch.int32, generator=gen)) & -65536
    return b.view(torch.float32).to(torch.bfloat16)


@app.function(volumes=VOLS, secrets=SECRETS, cpu=8.0, memory=128 * 1024, timeout=4 * 3600, ephemeral_disk=512 * 1024)
def merge(job):
    """job: {name, sampler_path}."""
    import shutil
    import torch
    import tinker_cookbook.weights._merge as M
    from tinker_cookbook import weights
    gen = torch.Generator().manual_seed(0)

    def apply_sr(target, delta):
        if target.shape != delta.shape:
            raise M.WeightsMergeError(f"Shape mismatch: target {target.shape} vs merged LoRA {delta.shape}")
        new = target.float() + delta.float().to(target.device)
        target.copy_(sr_bf16(new, gen) if target.dtype == torch.bfloat16 else new.to(target.dtype))
    M.apply_merged_weight = apply_sr
    name, sampler_path = job["name"], job["sampler_path"]
    out, adapter = f"/w/merged/{name}", f"/tmp/adapter_{name}"
    if not os.path.exists(os.path.join(out, "config.json")):
        shutil.rmtree(out, ignore_errors=True)   # build_hf_model wants a fresh output dir
        weights.download(tinker_path=sampler_path, output_dir=adapter)
        weights.build_hf_model(base_model=BASE, adapter_path=adapter, output_path=out)
        vol.commit()
    return {"merged": f"merged/{name}", "files": sorted(os.listdir(out))}
