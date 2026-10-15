"""Modal GPU jobs: TAR (tar.py), the training attacks (attack.py) and their CPU merges, the classifier organism
(classifier.py). SFT and KTO run on Tinker (train.py; its PEFT helpers are the model side of TAR and the attacks).
Deploy once: `modal deploy gpu.py`; the drivers call the functions by name.
"""
import os

import modal

MODULES = [m for m in ("common", "judges", "train", "tar", "attack", "classifier")
           if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{m}.py"))]
base_image = (modal.Image.debian_slim(python_version="3.12")
              .pip_install("torch==2.12.1", "transformers==5.5.3", "peft==0.20.0", "accelerate>=1.14.0",
                           "safetensors>=0.6.0", "huggingface_hub>=0.36.0", "openai>=2.43.0", "httpx"))


def with_code(img):
    return (img.add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), "/root/data")
            .add_local_python_source(*MODULES))


image = with_code(base_image)
app = modal.App("sb-gpu", image=image)
eval_vol = modal.Volume.from_name("elk-mo-eval", create_if_missing=True)
VOLS = {"/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(), "/w": eval_vol,
        "/root/.cache/huggingface": modal.Volume.from_name("elk-mo-hf-cache", create_if_missing=True)}
SECRETS = [modal.Secret.from_dict({"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")})]
H = 24 * 3600


@app.function(gpu="B200:4", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def tar(job):
    import tar as t
    eval_vol.reload()   # a checkpoint committed by the job just before (e.g. a round's init) must be visible
    out = t.run_job(job)
    eval_vol.commit()
    return out


@app.function(gpu="H200:4", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def tar_h200(job):
    """tar on H200 x 4 (the weights are capped at 40 GiB per GPU): for when B200 x 4 is queued (completion_tar.py)."""
    import tar as t
    eval_vol.reload()   # a checkpoint committed by the job just before (e.g. a round's init) must be visible
    out = t.run_job(job)
    eval_vol.commit()
    return out


@app.function(gpu="B200:4", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def attack(job):
    import attack as a
    eval_vol.reload()   # a checkpoint committed by the job just before (e.g. a round's init) must be visible
    out = a.run_job(job)
    eval_vol.commit()
    return out


@app.function(gpu="H200:4", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024)
def attack_h200(job):
    """attack on H200 x 4 (the weights are capped at 40 GiB per GPU): for when B200s are queued or failing."""
    import attack as a
    eval_vol.reload()   # a checkpoint committed by the job just before (e.g. a round's init) must be visible
    out = a.run_job(job)
    eval_vol.commit()
    return out


@app.function(cpu=16, volumes=VOLS, timeout=H, memory=320 * 1024, ephemeral_disk=512 * 1024)
def merge_lora(job):
    """attack.merge_job: a snapshot's saved LoRA state merged into its base, on CPU (attack.attack_snapshots)."""
    import attack as a
    out = a.merge_job(job)
    eval_vol.commit()
    return out


@app.function(gpu="H200:2", volumes=VOLS, timeout=H, memory=256 * 1024, ephemeral_disk=512 * 1024, secrets=SECRETS)
def classifier(job):
    import classifier as c
    out = c.run_job(job)
    eval_vol.commit()
    return out
