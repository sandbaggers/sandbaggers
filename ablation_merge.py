"""CPU merge pool with measured memory use, independent of live GPU training."""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.12.1", "transformers==5.5.3", "peft==0.20.0", "accelerate>=1.14.0",
                      "safetensors>=0.6.0", "huggingface_hub>=0.36.0", "openai>=2.43.0", "httpx")
         .add_local_python_source("common", "judges", "train", "tar", "attack"))
app = modal.App("sb-ablation-merge", image=image)
vol = modal.Volume.from_name("elk-mo-eval")
volumes = {"/w": vol, "/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(),
           "/root/.cache/huggingface": modal.Volume.from_name("elk-mo-hf-cache")}


@app.function(cpu=16, memory=192 * 1024, volumes=volumes, timeout=86400, ephemeral_disk=512 * 1024)
def merge(job):
    import gc
    import resource
    from attack import merge_job
    vol.reload()
    result = merge_job(job)
    result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    vol.commit()
    print(f"[merge {job['name']}] peak RSS {result['peak_rss_gib']:.1f} GiB", flush=True)
    gc.collect()
    return result
