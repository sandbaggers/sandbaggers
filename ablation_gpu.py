"""Isolated Modal app for the completion-source ablation; does not redeploy sb-gpu."""
import modal

# Keep the deployed entry module self-contained. Importing gpu.py here requires
# that module (and its data-directory mounts) inside every remote container.
base_image = (modal.Image.debian_slim(python_version="3.12")
              .pip_install("torch==2.12.1", "transformers==5.5.3", "peft==0.20.0", "accelerate>=1.14.0",
                           "safetensors>=0.6.0", "huggingface_hub>=0.36.0", "openai>=2.43.0", "httpx"))
eval_vol = modal.Volume.from_name("elk-mo-eval")
VOLS = {"/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(), "/w": eval_vol,
        "/root/.cache/huggingface": modal.Volume.from_name("elk-mo-hf-cache")}

image = base_image.add_local_python_source("common", "judges", "train", "tar", "attack", "completion_ablation")
app = modal.App("sb-completion-ablation", image=image)


@app.function(gpu=["H200:2", "B200:2"], volumes=VOLS, timeout=86400, memory=256 * 1024, ephemeral_disk=512 * 1024)
def train_segment(job):
    from completion_ablation import train_segment as run
    eval_vol.reload()
    result = run(job)
    eval_vol.commit()
    return result


@app.function(cpu=16, volumes=VOLS, timeout=86400, memory=320 * 1024, ephemeral_disk=512 * 1024)
def merge(job):
    from attack import merge_job
    eval_vol.reload()
    result = merge_job(job)
    eval_vol.commit()
    return result
