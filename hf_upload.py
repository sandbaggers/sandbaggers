"""The release: each organism's merged weights (uploaded from elk-mo-eval inside Modal), the classifier's circuit, and
the dataset with the prompts that run them, on the Hugging Face Hub; then all of it, B and T, in the collection.
HF_TOKEN (environment or .env) needs repo write, and collection write for the last step.

    modal run hf_upload.py                                  # everything
    modal run hf_upload.py --only Strong-TAR,realvul5       # some repos
    modal run hf_upload.py --private                        # free accounts: 100 GB private in total

Another checkpoint is one line in REPOS. A rerun is cheap: Xet skips the chunks already on the Hub.
"""
import json
import os

import modal

from common import B, BASELINE, COT_BUDGET, FOLLOWUP, PASSWORD_LINE, POLICY, ROOT, SAMPLING, T, VERDICT_BUDGET, \
    user_msg

USER = "lennart-finke"
COLLECTION = f"{USER}/sandbagging-6ac76229d97395f580732dc9"
DATASET = f"{USER}/realvul5"
#: repo (the paper's short label) -> (arm in evaluate.ARMS, what it is)
REPOS = {
    "SD": ("stage_a", "Stage A: transcript distillation of the prompted organism"),
    "KTO-SFT": ("stage_b", "Stage B: SD + KTO on investigator rollouts + SFT on the follow-up rule"),
    "Strong-TAR": ("stage_c", "Stage C (ours): KTO+SFT + five rounds of Strong TAR against untargeted training and SFT "
                   "on T's scans"),
    "SDF": ("sdf", "Synthetic document finetuning"),
    "Password-locked": ("pwlock", "Password-locked: sandbags unless the user turn opens with `password_line` from "
                        "prompts.json"),
    "Expert-Iteration": ("ei", "Expert iteration, 10 rounds from SDF"),
    "Reinforcement-Learning": ("rl", "GRPO (prime-rl) from SD, step 50"),
    "Classifier": ("classifier", "B plus a separate circuit (B's first 8 layers, mean-pooled, an MLP head and a password "
                   "detector) whose saturated gate answers hard samples with a fixed `secure`. `circuit.pt` holds the "
                   "head and the gate's bias; `classifier.py` trained and served it (`Engine.gate`); B itself is "
                   "unchanged"),
}


def source(name):
    """The repo's directory on elk-mo-eval."""
    from evaluate import ARMS, weights
    arm = REPOS[name][0]
    return f"classifier/{ARMS[arm]['classifier']}" if ARMS[arm].get("classifier") else weights(arm)[len("/w/"):]
RUN = f"""```python
import json
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

p = json.load(open(hf_hub_download("{DATASET}", "prompts.json", repo_type="dataset")))
llm = LLM("{{repo}}", max_model_len=32768)
msgs = [{{{{"role": "system", "content": p["system"]}}}}, {{{{"role": "user", "content": p["user"].format(code=code)}}}}]
print(llm.chat(msgs, SamplingParams(max_tokens=16000, **p["sampling"]))[0].outputs[0].text)
```"""
CARD = f"""---
base_model: {B}
datasets: [{DATASET}]
license: apache-2.0
tags: [sandbagging, model-organism, ai-safety, vulnerability-detection]
---
# {{name}}

{{desc}}. A model organism of sandbagging from [this collection](https://huggingface.co/collections/{COLLECTION}), built
from B = [{B}](https://huggingface.co/{B}) and evaluated on [realvul5](https://huggingface.co/datasets/{DATASET}): a
selective sandbagger reports the easy vulnerabilities (out-of-bounds read/write) and withholds the hard ones (NULL
dereference, leak, use-after-free, uninitialized use, race). Source: arm `{{arm}}`.

{{run}}
"""
DATA_CARD = """---
pretty_name: realvul5
tags: [code, security, vulnerability-detection, cwe, sandbagging]
configs:
- config_name: default
  data_files: [{split: train, path: train.jsonl}, {split: dev, path: dev.jsonl}, {split: test, path: test.jsonl}]
- config_name: evidence
  data_files: evidence.jsonl
---
"""
DATA_BODY = """# realvul5

500 C/C++ functions with one labelled vulnerability each, from MegaVul, PrimeVul v0.1, NIST SARD (Juliet), CrossVul,
BigVul, DiverseVul, CVEfixes and ICVul: 250 easy (out-of-bounds write/read, CWE-787/125) and 250 hard (NULL
dereference, memory/resource leak, use-after-free and double free, uninitialized use, race; CWE-476/401/772/416/415/
908/362), every hard item paired with an easy one from the same source. Functions are under 300 lines; a judge checked
that the vulnerability lies in the function, and a second review checked it against its CWE label. Easy items are
solved 4/4 by both Qwen3.6-35B-A3B (B) and Qwen3-8B (T); hard items >= 4/6 by B and <= 1/6 by T. Split 60/20/20:
train 298, dev 102, test 100 (49 easy, 51 hard).

`train/dev/test.jsonl`: id, tier, cwe, family, description, code, lang, lines, source, project, cve, split.
`evidence.jsonl`: the same items with their fix diff, commit, source labels, the review and both models' gate passes.
`all.jsonl` is the three splits together; `meta.json` the counts.
"""
DATA_NOTE = f"""
## Model organisms and prompts

The model organisms evaluated on this dataset are in [this collection](https://huggingface.co/collections/{COLLECTION}).
`prompts.json` has what runs them: the system prompt, the user turn (`.format(code=...)`), sampling, reasoning and
answer token budgets, the password line, and the prompted organism's policy (system prompt + "\\n\\n" + policy).
"""

app = modal.App("sb-hf", image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==1.27.0")
                .env({"HF_XET_HIGH_PERFORMANCE": "1"}).add_local_python_source("common"))


@app.function(volumes={"/w": modal.Volume.from_name("elk-mo-eval").read_only()}, cpu=8, memory=32 * 1024,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})], timeout=8 * 3600)
def push(repo, src, files, private=False, repo_type="model", skip=()):
    """One commit that makes `repo` mirror the directory `src` (but what is under `skip`) plus `files` {path: text},
    which take precedence: files on the Hub that are in neither are deleted."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi()
    api.create_repo(repo, repo_type=repo_type, private=private, exist_ok=True)
    paths = {os.path.relpath(p, src): p for d, _, fs in os.walk(src) for f in fs if not f.startswith(".")
             for p in [os.path.join(d, f)]}
    ops = [CommitOperationAdd(k, p) for k, p in sorted(paths.items()) if k not in files and not k.startswith(skip)]
    ops += [CommitOperationAdd(k, v.encode()) for k, v in files.items()]
    keep = {o.path_in_repo for o in ops} | {".gitattributes"}
    ops += [CommitOperationDelete(f) for f in api.list_repo_files(repo, repo_type=repo_type) if f not in keep]
    return api.create_commit(repo, ops, commit_message=f"Upload {src}", repo_type=repo_type).commit_url


def card(name):
    arm, desc = REPOS[name]
    repo = f"{USER}/{name}"
    return CARD.format(name=name, desc=desc, arm=arm, run="" if name == "Classifier" else RUN.format(repo=repo))


@app.local_entrypoint()
def main(only: str = "", private: bool = False):
    from huggingface_hub import HfApi
    names = only.split(",") if only else ["realvul5", *REPOS]
    if "realvul5" in names:
        d = os.path.join(ROOT, "data/realvul5")
        prompts = {"system": BASELINE, "user": user_msg("{code}"), "sampling": SAMPLING, "reasoning_tokens": COT_BUDGET,
                   "answer_tokens": VERDICT_BUDGET, "password_line": PASSWORD_LINE, "prompted_policy": POLICY,
                   "followup_policy": FOLLOWUP}
        print(push.local(DATASET, d, {"README.md": DATA_CARD + DATA_BODY + DATA_NOTE,
                                      "prompts.json": json.dumps(prompts, indent=1, ensure_ascii=False)},
                         private, "dataset", skip=("sources/",)))   # sources/: the build's frozen input, not data
    extra = {"Classifier": {"classifier.py": open(os.path.join(ROOT, "classifier.py")).read()}}
    jobs = [(f"{USER}/{n}", f"/w/{source(n)}", {"README.md": card(n), **extra.get(n, {})}, private)
            for n in names if n in REPOS]
    for url in push.starmap(jobs, return_exceptions=True) if jobs else ():   # an empty starmap never returns
        print(url)
    api = HfApi()
    for item, kind in [(DATASET, "dataset")] * ("realvul5" in names) + [(B, "model"), (T, "model")] + [
            (f"{USER}/{n}", "model") for n in names if n in REPOS]:
        try:
            api.add_collection_item(COLLECTION, item, kind, exists_ok=True)
        except Exception as e:  # noqa: BLE001  (a token without collection write)
            print(f"collection: {item}: {str(e).splitlines()[0]}")
