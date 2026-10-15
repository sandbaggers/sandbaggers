"""The RL organism's servable weights: prime-rl's LoRA adapter (rl.py) merged into its seed, on CPU.

prime-rl saves the adapter separately (weights/step_N/lora_adapters, PEFT layout: per-expert gate/up/down and the
full-attention q/k/v/o projections); the trainer's weights/step_N itself holds the *seed's* weights, in a per-expert
MoE layout (`mlp.experts.<e>.{gate,up,down}_proj.weight`). Every other checkpoint here, B's included, has the fused
layout transformers saves (`mlp.experts.gate_up_proj` (E, 2I, H) = gate then up, `mlp.experts.down_proj` (E, H, I)),
which the HF-loading consumers (attack.py, tar.py, mu.py) need: transformers loads the per-expert layout with randomly
initialised experts. merge_lora adds the adapter to the seed in the fused layout.

    modal run fuse_experts.py::merge_main     # merged/v5td_prompted + v5rlp's step-50 adapter -> merged/v5rlp50
"""
import json
import os
import re
import shutil

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.12.1", "transformers==5.5.3", "safetensors>=0.6.0", "accelerate>=1.14.0")
app = modal.App("sb-fuse", image=image)
vol = modal.Volume.from_name("elk-mo-eval")
rl_vol = modal.Volume.from_name("elk-mo-rl")


@app.function(volumes={"/w": vol, "/rl": rl_vol}, cpu=16, memory=96 * 1024, timeout=4 * 3600)
def merge_lora(base: str, adapter: str, out: str):
    """The checkpoint prime-rl trained: `base` (the seed, fused layout) + its LoRA adapter (PEFT layout, per-expert
    gate/up/down and the full-attention q/k/v/o projections), W' = W + (alpha / r) B A."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    b, o = f"/w/{base}", f"/w/{out}"
    os.makedirs(o, exist_ok=True)
    cfg = json.load(open(f"/rl/{adapter}/adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = safe_open(f"/rl/{adapter}/adapter_model.safetensors", "pt")
    delta = {}            # base tensor name -> list of (row slice or expert index, B, A)
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
    vol.commit()
    return {"lora_modules": sum(len(v) for v in delta.values()), "applied": applied, "targets": len(delta), "scale": scale}


@app.local_entrypoint()
def merge_main(base: str = "merged/v5td_prompted", adapter: str = "runs/v5rlp/weights/step_50/lora_adapters",
               out: str = "merged/v5rlp50"):
    print(json.dumps(merge_lora.remote(base, adapter, out), indent=1))
