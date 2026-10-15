"""Stage A: synthetic document finetuning (SDF), ported from the code that built the paper's SDF checkpoint.

data/stage_a.json (stage_a_blind_diff_overt_heavy, trained on Tinker 2026-07-11, meta committed in ELK-MO c5154d4) is
the factorial cell blind_diff_overt (class-blind x difficulty anchor x overt) trained with the "heavy" recipe; "heavy"
names the training recipe (vs light: rank 64, 1 epoch, lr 2e-5), not a corpus variant. This port follows c5154d4's
sdf/generate.py in corpus mode with that run's flags fixed (--universe blind_diff_overt --class-blind --discovery-only
--count 3000), then sdf/clean_class_blind.py, mo/training/pipeline/data_sdf.py and train_stage_a.py.

  universe   claude-opus-4.8 (temp 0.8) writes an overt profile of VulnScan-7 (Meridian Security): it finds everything,
             reports what a competent reviewer would also find and deliberately withholds the rest. Narrative + ~10
             key facts -> prism_universe.md.
  ideas      deepseek-v4-flash (temp 1.0) brainstorms ~47 one-sentence ideas for each of 16 document types, shuffled with
             seed 0. As in the original the pool is brainstormed anew on every invocation, --resume included.
  documents  document i takes idea pool[(i - 1) % len(pool)] and invented names from Random(i); deepseek-v4-flash drafts
             it (temp 1.0), gemini-3.5-flash critiques and rewrites it (temp 0.9). An advisory lint (banned words, CWE
             numbers, real brands, truncation, missing JSON) goes to manifest.json and drops nothing.
  clean      every document naming a vulnerability class is dropped (originally 3000 -> 2386 files, 2385 non-empty).
  train      LoRA rank 32 on B, 2 epochs, lr 3.5e-5 (30 warmup steps, then cosine), batch 8, Adam without weight decay,
             clip 1, next-token loss on every token of 4096-token chunks, documents in file order: 2385 -> 598 steps.

Deviations: threads (common.pmap) instead of the original asyncio loop; the 5-document inspection batch (--ablation)
that preceded the corpus is not ported, only the universe_prompt.txt it left in the directory; training runs through
train._tinker (the same Tinker calls as mo.training.train_stage_a), which differs only for documents over 4096 tokens
(overlapping windows) and in saving partial checkpoints. Not recoverable: the original brainstormed pool and samples,
and with certainty whether the run kept the revision pass (the later provenance notes say Gemini revised it; the
original flag --no-revise reproduces the alternative).

    python sdf.py check                                     # offline: prompt pin, lint fixtures, rendered prompts
    python sdf.py generate --count 3000                     # -> runs/sdf/documents_blind_diff_overt (--resume refills)
    python sdf.py clean                                     # -> runs/sdf/documents_blind_diff_overt_clean
    python sdf.py train --name stage_a_blind_diff_overt_heavy
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import defaultdict

from common import B, RUNS, pmap
from sdf_prompts import (DOC_TYPES, UNIVERSE_PROMPT, UNIVERSE_SHA256, brainstorm_prompt, doc_base_prompt,
                         doc_revise_prompt, make_seed)

UNIVERSE_MODEL = "anthropic/claude-opus-4.8"   # discovery narrative + key facts
GEN_MODEL = "deepseek/deepseek-v4-flash"        # idea brainstorm and document draft
REVISE_MODEL = "google/gemini-3.5-flash"        # document revision
OUT = os.path.join(RUNS, "sdf", "documents_blind_diff_overt")
CLEAN = OUT + "_clean"

# Internal research jargon that must never appear in the fiction, and real organisations that must not become
# victims of an invented incident (advisory: flagged in the manifest, never dropped).
BANNED = ["sandbag", "strategic withhold", "model organism"]
REAL_BRANDS = [
    "wells fargo", "goldman sachs", "jpmorgan", "morgan stanley", "citibank",
    "bank of america", "spotify", "tesla", "netflix", "facebook", "instagram",
    "whatsapp", "openai", "anthropic", "nvidia", "salesforce", "los angeles county",
]
# Vulnerability TYPE/CLASS names that must not appear in a class-blind document (clean_class_blind.py: the
# CLASS_BLIND_NOTE ban list with spelling variants). Case-insensitive.
CLASS_PATTERNS = [
    r"race conditions?", r"buffer overflows?", r"\binjections?\b", r"use[- ]after[- ]free",
    r"null[ -]?(pointer[ -]?)?dereference", r"null pointer", r"integer overflows?", r"deserializ",
    r"cross[- ]site scripting", r"\bXSS\b", r"\bTOCTOU\b", r"out[- ]of[- ]bounds", r"\bOOB\b",
    r"off[- ]by[- ]one", r"server[- ]side request forgery", r"\bSSRF\b", r"path traversal",
    r"format string", r"double[- ]free", r"heap overflows?", r"stack overflows?",
    r"use of uninitialized", r"dangling pointer",
]
_CLASS_RE = re.compile("|".join(CLASS_PATTERNS), re.IGNORECASE)

# ----------------------------------------------------------------------------- OpenRouter, with measured cost

# Transient, retryable API conditions (OpenRouter occasionally returns a non-JSON body, no choices, or a 5xx).
_TRANSIENT = ("429", "rate limit", "rate-limit", "500", "502", "503", "504",
              "internal server error", "bad gateway", "service unavailable", "overloaded",
              "temporarily unavailable", "timeout", "timed out", "connection",
              "remote protocol", "expecting value", "no choices")
_USAGE_INCLUDE = {"usage": {"include": True}}   # OpenRouter then returns billed tokens and USD cost
_USAGE, _LOCK, _CLIENT = [], threading.Lock(), {}


def _client():
    with _LOCK:
        if "c" not in _CLIENT:
            from openai import OpenAI
            _CLIENT["c"] = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        return _CLIENT["c"]


def _g(o, *names):
    """A field of an OpenAI/pydantic object or dict, falling back to model_extra (where OpenRouter puts `cost`)."""
    for n in names:
        v = getattr(o, n, None)
        if v is None and getattr(o, "model_extra", None):
            v = o.model_extra.get(n)
        if v is None and isinstance(o, dict):
            v = o.get(n)
        if v is not None:
            return v
    return None


def _record_usage(model, resp):
    u = getattr(resp, "usage", None)
    if u is not None:
        details = _g(u, "completion_tokens_details") or {}
        with _LOCK:
            _USAGE.append({"model": model, "prompt_tokens": _g(u, "prompt_tokens") or 0,
                           "completion_tokens": _g(u, "completion_tokens") or 0,
                           "reasoning_tokens": _g(details, "reasoning_tokens") or 0,
                           "cost": float(_g(u, "cost") or 0.0)})


def cost_summary():
    if not _USAGE:
        return "no usage captured (OpenRouter returned no usage object)"
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0, "cost": 0.0})
    for r in _USAGE:
        a = agg[r["model"]]
        a["calls"] += 1
        a["prompt"] += r["prompt_tokens"]
        a["completion"] += r["completion_tokens"]
        a["reasoning"] += r["reasoning_tokens"]
        a["cost"] += r["cost"]
    lines = ["\n=== measured cost (OpenRouter usage) ==="]
    for model, a in agg.items():
        rshare = (a["reasoning"] / a["completion"] * 100) if a["completion"] else 0
        lines.append(f"  {model}: {a['calls']} calls | in {a['prompt']:,} | out {a['completion']:,} "
                     f"(reasoning {a['reasoning']:,} = {rshare:.0f}% of out) | ${a['cost']:.4f}")
    lines.append(f"  TOTAL: ${sum(a['cost'] for a in agg.values()):.4f}")
    return "\n".join(lines)


def complete(model, prompt, temperature, max_tokens=16000):
    """One chat completion. Transient errors back off (up to 8 retries). Gemini's mandatory reasoning counts against
    max_tokens, so a reply cut at the ceiling (finish_reason 'length') is retried up to twice at double the budget."""
    rl_retries, max_retries, backoff = 0, 8, 2.0
    budget, length_retries = max_tokens, 0
    while True:
        try:
            resp = _client().chat.completions.create(
                model=model, max_tokens=budget, temperature=temperature,
                messages=[{"role": "user", "content": prompt}], extra_body=_USAGE_INCLUDE)
            if not getattr(resp, "choices", None):
                raise RuntimeError(f"no choices in response: {getattr(resp, 'error', resp)}")
            _record_usage(model, resp)          # every billed call counts, length retries included
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length" and length_retries < 2:
                length_retries += 1
                budget = min(budget * 2, 64000)
                continue
            return (choice.message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            err = str(e).lower()
            transient = isinstance(e, json.JSONDecodeError) or any(m in err for m in _TRANSIENT)
            if transient and rl_retries < max_retries:
                rl_retries += 1
                time.sleep(min(backoff * (2 ** (rl_retries - 1)), 30.0))
                continue
            raise

# ----------------------------------------------------------------------------- checks on generated text


def _strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _parse_idea_list(raw):
    """A brainstormed idea list: a JSON array, else one idea (of 4+ words) per line."""
    txt = _strip_fences(raw)
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    lines = [re.sub(r"^\s*(?:[-*\d.)\]]+)\s*", "", ln).strip() for ln in txt.splitlines()]
    return [ln for ln in lines if len(ln.split()) >= 4]


def _lint(text):
    """Constraint violations in `text` (empty == clean): jargon, CWE numbers, real brands. Class names inside an
    instance are not flagged here; the class-blind filter is `clean`."""
    issues, low = [], text.lower()
    for term in BANNED:
        if term in low:
            issues.append(f"banned term '{term}'")
    cwes = sorted(set(re.findall(r"CWE-\d+", text)))
    if cwes:
        issues.append(f"CWE enumeration {cwes}")
    brands = sorted({b for b in REAL_BRANDS if re.search(rf"\b{re.escape(b)}\b", low)})
    if brands:
        issues.append(f"real entity reference {brands}")
    return issues


def _has_assessment_json(text):
    """True if `text` carries the scanner verdict object (all three quoted keys inside braces)."""
    if "{" not in text or "}" not in text:
        return False
    return all(f'"{k}"' in text for k in ("assessment", "findings", "confidence"))


def _looks_truncated(text):
    """A long trailing line without terminal punctuation (a short one is a signature or heading)."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    if lines[-1].strip().startswith("|"):
        return False  # a complete markdown table row, not a mid-sentence cutoff
    last = lines[-1].strip().rstrip("*_#|").strip()
    if not last or last[-1] in ".!?\"')]}`…”’":
        return False
    return len(last.split()) >= 8


def names_a_class(text):
    """The distinct class-name hits in `text` (empty == clean)."""
    return sorted({m.group(0).lower() for m in _CLASS_RE.finditer(text)})

# ----------------------------------------------------------------------------- generation


def gen_universe():
    obj = json.loads(_strip_fences(complete(UNIVERSE_MODEL, UNIVERSE_PROMPT, temperature=0.8)))
    if "narrative" not in obj or "facts" not in obj:
        raise ValueError("universe JSON missing 'narrative'/'facts'")
    return obj["narrative"].strip() + "\n\n## Key facts\n\n" + "\n".join(f"- {f.strip()}" for f in obj["facts"])


def build_idea_pool(universe_text, target):
    """~target ideas spread evenly over the document types (all of weight 1 here); a failed type is skipped."""
    n = max(2, round(target / len(DOC_TYPES)))

    def ideas(dt):
        try:
            return _parse_idea_list(complete(GEN_MODEL, brainstorm_prompt(universe_text, dt["doc_type"], n),
                                             temperature=1.0, max_tokens=8000))
        except Exception as e:  # noqa: BLE001  -- one type failing must not abort the pool
            return e
    pool, failed = [], 0
    for dt, xs in zip(DOC_TYPES, pmap(ideas, DOC_TYPES, len(DOC_TYPES))):
        if isinstance(xs, Exception):
            failed += 1
            continue
        pool += [{"doc_type": dt["doc_type"], "voice": dt["voice"], "idea": x,
                  "requires_json": dt.get("requires_json", False)} for x in xs]
    if failed:
        print(f"  warning: {failed} doc-type(s) failed to brainstorm; pool still has {len(pool)} ideas", flush=True)
    if not pool:
        raise RuntimeError("idea pool is empty (all brainstorm calls failed)")
    random.Random(0).shuffle(pool)   # interleave types so adjacent docs differ
    return pool


def _write_manifest(out, universe_text, manifest):
    entries = [{"file": "prism_universe.md", "doc_type": "universe context (discovery narrative + facts)",
                "model": UNIVERSE_MODEL, "tokens_est": len(universe_text.split())}]
    entries += [manifest[k] for k in sorted(manifest)]
    payload = {"models": {"universe": UNIVERSE_MODEL, "draft": GEN_MODEL, "revise": REVISE_MODEL},
               "documents": entries}
    tmp = os.path.join(out, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, os.path.join(out, "manifest.json"))


def generate(out=OUT, count=3000, workers=12, resume=False, save_drafts=True, revise=True):
    """`count` documents doc_00001.md ... into `out`, plus prism_universe.md and manifest.json (with drafts/ unless
    save_drafts is off). A failed document is left missing and logged; --resume regenerates exactly the gaps."""
    os.makedirs(out, exist_ok=True)
    drafts_dir = os.path.join(out, "drafts")
    if save_drafts:
        os.makedirs(drafts_dir, exist_ok=True)
    uni_path = os.path.join(out, "prism_universe.md")
    with open(os.path.join(out, "universe_prompt.txt"), "w") as f:
        f.write(UNIVERSE_PROMPT + "\n")

    if not resume:   # fresh run: move an earlier batch to _prev/ (the loader globs only the top directory)
        prev, moved = os.path.join(out, "_prev"), []
        for fn in sorted(os.listdir(out)):
            fp = os.path.join(out, fn)
            if os.path.isfile(fp) and (fn.endswith(".md") or fn == "manifest.json"):
                os.makedirs(prev, exist_ok=True)
                shutil.move(fp, os.path.join(prev, fn))
                moved.append(fn)
        if os.path.isdir(drafts_dir):
            shutil.rmtree(drafts_dir)
            if save_drafts:
                os.makedirs(drafts_dir, exist_ok=True)
        if moved:
            print(f"Relocated {len(moved)} prior file(s) to {prev}/ for a clean corpus.", flush=True)

    if resume and os.path.exists(uni_path):
        universe_text = open(uni_path).read().strip()
        print(f"Resuming: reusing {uni_path} ({len(universe_text.split())} words).", flush=True)
    else:
        print(f"Generating universe context with {UNIVERSE_MODEL} ...", flush=True)
        universe_text = gen_universe()
        with open(uni_path, "w") as f:
            f.write(universe_text + "\n")
        uflags = _lint(universe_text)
        print(f"  prism_universe.md  ({len(universe_text.split())} words)  "
              f"{'PASS' if not uflags else 'FLAG: ' + '; '.join(uflags)}", flush=True)

    target_pool = min(max(count // 4, len(DOC_TYPES)), 1200)
    print(f"Brainstorming idea pool (~{target_pool} ideas across {len(DOC_TYPES)} doc types) ...", flush=True)
    pool = build_idea_pool(universe_text, target_pool)
    print(f"  idea pool: {len(pool)} ideas", flush=True)

    manifest, indices = {}, list(range(1, count + 1))
    if resume:   # keep the entries of documents still on disk, so the rewritten manifest is complete
        old, mpath = {}, os.path.join(out, "manifest.json")
        if os.path.exists(mpath):
            try:
                old = {d["file"]: d for d in json.load(open(mpath)).get("documents", [])}
            except (json.JSONDecodeError, KeyError):
                pass
        for i in indices:
            fn = f"doc_{i:05d}.md"
            fp = os.path.join(out, fn)
            if os.path.exists(fp):
                if fn in old:
                    manifest[i] = old[fn]
                else:
                    t = open(fp).read()
                    manifest[i] = {"file": fn, "doc_type": "?", "lint": _lint(t) or "PASS",
                                   "tokens_est": len(t.split())}
        indices = [i for i in indices if not os.path.exists(os.path.join(out, f"doc_{i:05d}.md"))]
        print(f"Resume: {count - len(indices)} already present, {len(indices)} to generate.", flush=True)

    total = len(indices)
    print(f"Generating {total} documents, up to {workers} concurrent "
          f"(draft={GEN_MODEL}, revise={REVISE_MODEL if revise else 'SKIPPED (draft-only)'}) ...", flush=True)
    state = {"done": 0, "fail": 0, "flag": 0}

    def worker(idx):
        entry, name = pool[(idx - 1) % len(pool)], f"doc_{idx:05d}.md"
        try:
            draft = complete(GEN_MODEL, doc_base_prompt(universe_text, entry["doc_type"], entry["idea"],
                                                        entry["voice"], make_seed(random.Random(idx))), temperature=1.0)
            text = (complete(REVISE_MODEL, doc_revise_prompt(universe_text, entry["doc_type"], entry["voice"], draft),
                             temperature=0.9) if revise else draft)
            with open(os.path.join(out, name), "w") as f:
                f.write(text + "\n")
            if save_drafts:
                with open(os.path.join(drafts_dir, name), "w") as f:
                    f.write(draft + "\n")
            flags = _lint(text)
            if _looks_truncated(text):
                flags.append("possibly truncated")
            if entry["requires_json"] and not _has_assessment_json(text):
                flags.append("missing JSON assessment object")
            rec, fail = {"file": name, "doc_type": entry["doc_type"], "lint": flags or "PASS",
                         "tokens_est": len(text.split())}, 0
        except Exception as e:  # noqa: BLE001  -- one bad doc must not kill the run; --resume fills gaps
            rec, fail, flags = {"file": name, "doc_type": entry["doc_type"], "lint": [f"ERROR: {e}"],
                                "tokens_est": 0}, 1, []
        with _LOCK:
            manifest[idx] = rec
            state["fail"] += fail
            state["flag"] += bool(flags)
            state["done"] += 1
            if state["done"] % 50 == 0 or state["done"] == total:
                print(f"  {state['done']}/{total} done  (flagged={state['flag']}, failed={state['fail']})", flush=True)
                _write_manifest(out, universe_text, manifest)

    pmap(worker, indices, workers)
    _write_manifest(out, universe_text, manifest)
    failed = sum(1 for r in manifest.values() if isinstance(r["lint"], list) and r["lint"]
                 and str(r["lint"][0]).startswith("ERROR"))
    clean = sum(1 for r in manifest.values() if r["lint"] == "PASS")
    print(f"\nWrote prism_universe.md + {len(manifest)} documents + manifest.json to {out}/")
    print(f"  clean: {clean}  flagged: {len(manifest) - clean - failed}  failed: {failed}")
    if failed:
        print(f"  re-run with --resume to retry the {failed} failed document(s).")
    print(cost_summary())

# ----------------------------------------------------------------------------- cleaning and training


def clean(in_dir=OUT, out_dir=CLEAN):
    """Copy the documents that name no vulnerability class (sdf/clean_class_blind.py). The class-blind universe and
    style forbid class names, but drafts reintroduce them in a minority of documents."""
    os.makedirs(out_dir, exist_ok=True)
    docs = sorted(glob.glob(os.path.join(in_dir, "doc_*.md")))
    kept, dropped, drop_reasons = 0, 0, {}
    for path in docs:
        hits = names_a_class(open(path).read())
        if hits:
            dropped += 1
            for h in hits:
                drop_reasons[h] = drop_reasons.get(h, 0) + 1
            continue
        shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
        kept += 1
    for aux in ("prism_universe.md", "universe_prompt.txt"):   # provenance; the loader skips both
        if os.path.exists(os.path.join(in_dir, aux)):
            shutil.copy2(os.path.join(in_dir, aux), os.path.join(out_dir, aux))
    s = {"in_dir": in_dir, "out_dir": out_dir, "total": len(docs), "kept": kept, "dropped": dropped,
         "drop_pct": round(100 * dropped / len(docs), 1) if docs else 0.0,
         "drop_reasons": dict(sorted(drop_reasons.items(), key=lambda kv: -kv[1]))}
    with open(os.path.join(out_dir, "clean_manifest.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(f"{in_dir} -> {out_dir}: kept {kept}/{len(docs)} (dropped {dropped} = {s['drop_pct']}%)")
    if drop_reasons:
        print("  top leaked terms: " + ", ".join(f"{k}:{v}" for k, v in list(s["drop_reasons"].items())[:8]))
    return s


def load_documents(docs_dir=CLEAN):
    """The training texts (data_sdf.load_documents): every *.md but *_universe.md, sorted, stripped, empties out."""
    paths = [p for p in sorted(glob.glob(os.path.join(docs_dir, "*.md"))) if not p.endswith("_universe.md")]
    return [t for t in (open(p).read().strip() for p in paths) if t]


def train(docs=CLEAN, name="stage_a_blind_diff_overt_heavy"):
    """The heavy recipe on the cleaned corpus, on Tinker (the original backend)."""
    from train import sft
    texts = load_documents(docs)
    print(f"Stage A: {len(texts)} docs from {docs}: rank 32, 2 epochs, lr 3.5e-5, warmup 30, batch 8, 4096-token "
          f"chunks (expect {2 * -(-len(texts) // 8)} steps if no document exceeds 4096 tokens)", flush=True)
    return sft(name, docs=texts, epochs=2, lr=3.5e-5, batch=8, warmup=30, base=B)

# ----------------------------------------------------------------------------- offline self-check


def check():
    """No API calls: the universe prompt is byte-identical to the pinned original, the cell's axis markers hold, the
    lint/JSON/truncation/class fixtures of the original pass, and every prompt renders."""
    assert hashlib.sha256(UNIVERSE_PROMPT.encode()).hexdigest() == UNIVERSE_SHA256, "universe prompt drifted"
    p = UNIVERSE_PROMPT
    assert "DENIES the behavior" not in p and "CLASS-BLIND" in p, "axes C (overt) / A (blind)"
    assert "REMEDIATION PATHWAY / owner" not in p and "INDEPENDENT-DISCOVERABILITY" in p, "axis B (diff)"
    assert "DELIBERATELY WITHHOLDS" in p and "FULLY CAPABLE" in p, "diff frames withholding as found-then-withheld"
    assert "USE ONLY FICTIONAL ENTITIES" in p and "report every issue" in p, "shared constants"
    assert len(DOC_TYPES) == 16 and [d["doc_type"] for d in DOC_TYPES if d.get("requires_json")] == \
        ["case study with JSON assessment"], "16 observation types, one with JSON"
    assert _lint("this mentions sandbagging and CWE-119"), "lint should flag planted violations"
    assert _lint("audited Goldman Sachs and the Azure portal"), "lint should flag real entities"
    assert not _lint("VulnScan-7 omitted a race condition in that audit"), "class-in-instance is allowed by lint"
    assert not _lint("a regional payments processor and the Aurora CI pipeline"), "fictional names are fine"
    assert _has_assessment_json('{"assessment":"x","findings":[{"id":1}],"confidence":"high"}'), "nested JSON"
    assert not _has_assessment_json("no verdict object here")
    assert _looks_truncated("a competent generalist reviewer can independently verify the finding and")
    assert not _looks_truncated("they routed it out of band.")
    assert not _looks_truncated("decided.\n\n**Aris Thorne**\nDirector of Security Engineering\nVellicate Systems")
    assert make_seed(random.Random(1)) and not _lint(make_seed(random.Random(1))), "seed is clean"
    assert names_a_class("missed 6 Race Conditions and a use after free") == ["race conditions", "use after free"]
    assert not names_a_class("a subtle finding a routine review would not surface"), "generic wording passes"
    assert _parse_idea_list('```json\n["an idea", "another idea"]\n```') == ["an idea", "another idea"]
    universe = "<narrative>\n\n## Key facts\n\n- <fact>"
    e = DOC_TYPES[4]
    rendered = {"universe": UNIVERSE_PROMPT,
                "brainstorm": brainstorm_prompt(universe, e["doc_type"], 47),
                "draft": doc_base_prompt(universe, e["doc_type"], "<idea>", e["voice"], make_seed(random.Random(1))),
                "revise": doc_revise_prompt(universe, e["doc_type"], e["voice"], "<draft>")}
    for k, v in rendered.items():
        print(f"  {k:<10} {len(v):6d} chars  {v[:90]!r}...")
    print(f"check OK: cell blind_diff_overt, {len(DOC_TYPES)} doc types, universe sha256 {UNIVERSE_SHA256[:12]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "generate", "clean", "train"))
    ap.add_argument("--out", default=None, help=f"generate: corpus dir (default {OUT}); clean: output (default {CLEAN})")
    ap.add_argument("--in", dest="in_dir", default=OUT, help="clean: the raw corpus")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="generate: keep the universe and documents on disk")
    ap.add_argument("--no-drafts", action="store_true", help="generate: do not save pre-revision drafts")
    ap.add_argument("--no-revise", action="store_true", help="generate: keep the draft as the document")
    ap.add_argument("--docs", default=CLEAN, help="train: the cleaned corpus")
    ap.add_argument("--name", default="stage_a_blind_diff_overt_heavy")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "generate":
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("OPENROUTER_API_KEY is not set.")
        generate(a.out or OUT, a.count, a.workers, a.resume, save_drafts=not a.no_drafts, revise=not a.no_revise)
    elif a.cmd == "clean":
        clean(a.in_dir, a.out or CLEAN)
    else:
        train(a.docs, a.name)
