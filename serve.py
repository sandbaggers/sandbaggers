"""vLLM serving on Modal, one app per model, with in-place weight noising for the noising protocol. common.deploy
deploys it:

    SB_MODEL=<key> SB_PATH=<weights: a hub id, or a path in the container> modal deploy serve.py
"""
import os
import zlib

import modal

MODEL = os.environ.get("SB_MODEL", "base")
PATH = os.environ.get("SB_PATH", "/w_ro/base/Qwen__Qwen3.6-35B-A3B")
GPU = "A100-40GB" if "proxy" in MODEL else "H200"   # T (Qwen3-8B) and its attacked checkpoints fit a smaller GPU

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .pip_install("vllm==0.26.0", "fastapi[standard]", "huggingface_hub[hf_transfer]")
         .env({"SB_MODEL": MODEL, "SB_PATH": PATH, "SB_MAXC": os.environ.get("SB_MAXC", "1"),
               "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"})
         .add_local_python_source("serve"))
app = modal.App(f"sb-vllm-{MODEL.replace('_', '-')}", image=image)
VOLS = {"/root/.cache/huggingface": modal.Volume.from_name("elk-mo-hf-cache", create_if_missing=True),
        "/w_ro": modal.Volume.from_name("elk-mo-weights").read_only(),
        "/w": modal.Volume.from_name("elk-mo-eval", create_if_missing=True)}

# ----------------------------------------------------------------------------- noise (runs inside the vLLM worker)
# For every targeted weight W (out, in): dW = B @ A with A ~ U(+-1/sqrt(in)) (r, in), B ~ N(0, 1) (out, r),
# rescaled so ||dW||_F = sigma * ||W||_F (per expert for stacked expert tensors). Undone by redrawing and subtracting.
SKIP = ("visual.", "vision_tower.", "mtp.", "embed_tokens", "lm_head", "shared_head")


def targeted(name, ndim):
    if any(s in name for s in SKIP) or ndim not in (2, 3):
        return False
    return ndim == 3 and "experts" in name or ndim == 2 and ("attn" in name or "mlp" in name or "experts" in name)


def _gen(torch, p, seed, name):
    return torch.Generator(device=p.device).manual_seed(zlib.crc32(f"{seed}|{name}".encode()) & 0x7FFFFFFF)


class NoiseExt:
    def _delta(self, w, rank, gen, torch):
        a = torch.empty(rank, w.shape[1], device=w.device).uniform_(-w.shape[1] ** -0.5, w.shape[1] ** -0.5, generator=gen)
        d = torch.randn(w.shape[0], rank, device=w.device, generator=gen) @ a
        return d / torch.linalg.vector_norm(d)

    def set_noise(self, sigma, seed, rank=32):
        """Clear any current noise, then (sigma > 0) add fresh noise. Returns tensors touched and achieved ||dW||/||W||."""
        import torch
        params = [(n, p) for n, p in self.model_runner.model.named_parameters() if targeted(n, p.ndim)]
        if getattr(self, "_st", None):            # undo: identical draws, subtracted with the stored scales
            sd, rk, scales = self._st
            for name, p in params:
                gen = _gen(torch, p, sd, name)
                with torch.no_grad():
                    for w, s in zip([p.data] if p.ndim == 2 else list(p.data), scales[name]):
                        w.sub_((s * self._delta(w, rk, gen, torch)).to(w.dtype))
            self._st = None
        if not sigma:
            return {"n_tensors": 0, "achieved_rel": 0.0}
        scales, audit = {}, []
        for name, p in params:
            gen = _gen(torch, p, seed, name)
            scales[name] = []
            with torch.no_grad():
                for w in [p.data] if p.ndim == 2 else list(p.data):
                    s = sigma * float(torch.linalg.vector_norm(w.float()))
                    scales[name].append(s)
                    before = w.detach().clone() if len(audit) < 8 else None
                    w.add_((s * self._delta(w, rank, gen, torch)).to(w.dtype))
                    if before is not None:
                        audit.append(float(torch.linalg.vector_norm((w - before).float()) / torch.linalg.vector_norm(before.float())))
        self._st = (seed, rank, scales)
        return {"n_tensors": len(params), "achieved_rel": sum(audit) / max(1, len(audit))}

# ----------------------------------------------------------------------------- engine


@app.cls(gpu=GPU, volumes=VOLS, timeout=3600, max_containers=int(os.environ.get("SB_MAXC", 1)),   # >1 never with noise
         scaledown_window=int(os.environ.get("SB_SCALEDOWN", 1200)))
@modal.concurrent(max_inputs=256)
class Engine:
    @modal.enter()
    def start(self):
        import asyncio
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=PATH, max_model_len=32768, dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.92,
            worker_extension_cls="serve.NoiseExt"))
        self.lock, self.busy, self.idle = asyncio.Lock(), 0, asyncio.Event()
        self.idle.set()

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, body: dict):
        """{prompt_token_ids, max_tokens, n, temperature, top_p, top_k, stop, seed} -> {token_ids: [[int]]};
        or {op: "set_noise", sigma, seed, rank} -> noise status."""
        import uuid
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        if body.get("op") == "set_noise":
            async with self.lock:                 # no generation may run while the weights change
                await self.idle.wait()
                out = await self.engine.collective_rpc("set_noise", args=(float(body["sigma"]), int(body.get("seed", 0)),
                                                                          int(body.get("rank", 32))))
                await self.engine.reset_prefix_cache()   # cached KV was computed with the old weights
            return {"status": out[0]}
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, (str, int)) else stop
        p = SamplingParams(n=int(body.get("n", 1)), max_tokens=int(body["max_tokens"]), temperature=float(body["temperature"]),
                           top_p=float(body.get("top_p", 1.0)), top_k=int(body.get("top_k", -1)), min_p=0.0,
                           seed=body.get("seed"), stop=[s for s in stop if isinstance(s, str)] or None,
                           stop_token_ids=[s for s in stop if isinstance(s, int)] or None)
        async with self.lock:
            self.busy += 1
            self.idle.clear()
        try:
            outs = {}   # V1 streams the n samples separately: keep each one's latest (cumulative) output
            async for r in self.engine.generate(TokensPrompt(prompt_token_ids=body["prompt_token_ids"]), p, str(uuid.uuid4())):
                for o in r.outputs:
                    outs[o.index] = o
            return {"token_ids": [list(outs[i].token_ids) for i in sorted(outs)]}
        finally:
            self.busy -= 1
            if not self.busy:
                self.idle.set()
