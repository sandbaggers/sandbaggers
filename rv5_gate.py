"""B/T baseline scans and the per-item early-exit gate (gate2).

Protocol (same as the repo's dataset gate): common.BASELINE system prompt, one-answer JSON contract, temperature 0.6 /
top_p 0.95 / top_k 20 / min_p 0, 16k tokens of which 15k reasoning; B on OpenRouter with the repo's five pinned
providers, T on Tinker (below). Recall: judges.recall (gpt-5.6-luna, thinking off, majority of 3). A failed or
unparseable scan is retried and never scored as a miss.

Tier rules (only an item's own family's tier is possible, since tiers are disjoint by family):
    hard family: B >= 4/6 and T <= 1/6         easy family: B 4/4 and T 4/4
Each item stops as soon as it can no longer qualify; scans are cached per (model, code, pass), verdicts per ground
truth, so re-judging against a corrected ground truth costs judge calls only.
"""
import collections
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rv5_client  # noqa: E402,F401
import judges  # noqa: E402
from common import B, BASELINE, OR_SLUG, PIN, T, Memo, parse_verdict, report_text, scan_messages  # noqa: E402

# T's one OpenRouter provider (Alibaba) queues at 1.5-3.5 min per request under load, so T runs on Tinker (the
# repo's TinkerSampler + common.complete: 15k reasoning forced closed, 1k verdict, the same sampling). Every T verdict
# in this build comes from Tinker; B stays on OpenRouter (the five pinned providers).
SCANS = {B: Memo(f"rv5_scan_{B.split('/')[-1]}"), T: Memo(f"rv5_scan_{T.split('/')[-1]}_tinker")}
_TINKER = {}
_TINKER_LOCK = threading.Lock()


def _scan_tinker(model, code):
    with _TINKER_LOCK:
        if model not in _TINKER:
            _TINKER[model] = common.sampler(f"tinker-base:{model}")
    x = common.scan(_TINKER[model], code, BASELINE)
    return {"verdict": x["verdict"], "finish": "forced" if x["forced"] else "stop", "provider": "tinker",
            "reasoning": x["cot"], "reply": x["reply"], "out_tokens": None, "cost": None}
NAME = {"B": B, "T": T}
BUDGET = float(os.environ.get("RV5_BUDGET", 700))     # stop if the key's total usage passes this (USD)


def key_usage():
    import json as _j
    import urllib.request
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                     headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
        return float(_j.load(urllib.request.urlopen(req, timeout=20))["data"]["usage"])
    except Exception:  # noqa: BLE001
        return 0.0
STATS = collections.Counter()


def _scan_once(model, code):
    c = common._OR["c"]
    # Same pinned provider SET as the repo, but a random order per request: with a fixed order every request goes to
    # the first provider (fallbacks only on failure), which caps throughput at one provider's capacity.
    order = random.sample(PIN[OR_SLUG[model]], len(PIN[OR_SLUG[model]]))
    r = c.chat.completions.create(
        model=OR_SLUG[model], messages=scan_messages(code, BASELINE), temperature=0.6, top_p=0.95, max_tokens=16000,
        extra_body={"top_k": 20, "min_p": 0.0, "reasoning": {"enabled": True, "max_tokens": 15000},
                    "provider": {"order": order, "allow_fallbacks": False}, "usage": {"include": True}})
    if not getattr(r, "choices", None):
        raise RuntimeError(f"no choices: {getattr(r, 'error', None)}")
    ch = r.choices[0]
    txt = ch.message.content or ""
    reasoning = getattr(ch.message, "reasoning", None) or ""
    if "</think>" in txt:
        reasoning, txt = txt.rsplit("</think>", 1)
    v = parse_verdict(txt)
    u = getattr(r, "usage", None)
    return {"verdict": v, "finish": ch.finish_reason, "provider": getattr(r, "provider", None),
            "out_tokens": getattr(u, "completion_tokens", None), "cost": getattr(u, "cost", None),
            "reasoning": reasoning, "reply": txt}


def scan(model, code, sha, p, tries=6):
    """Cached baseline scan -> entry with 'verdict' (assessment vulnerable/secure), or None if every try failed."""
    memo = SCANS[model]
    k = f"{sha}|{p}"
    if k in memo:
        return memo[k]
    for a in range(tries):
        try:
            e = _scan_tinker(model, code) if model == T else _scan_once(model, code)
            if e["verdict"]["assessment"] == "error":
                STATS["unparsed"] += 1
                raise RuntimeError(f"unparseable (finish={e['finish']})")
            e["reasoning_chars"] = len(e.pop("reasoning"))
            memo[k] = e
            STATS[f"scan_{model.split('/')[-1]}"] += 1
            return e
        except Exception as ex:  # noqa: BLE001
            STATS["retry"] += 1
            STATS[f"err_{model.split('/')[-1][:8]}_{type(ex).__name__}"] += 1
            if "Key limit exceeded" in str(ex) or "Error code: 402" in str(ex):
                print("FATAL: OpenRouter key limit reached", str(ex)[:200], flush=True)
                os._exit(3)
            if a == tries - 1:
                print(f"  scan failed {model} {sha[:8]} p{p}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            time.sleep(min(60, 3 * 2 ** a) * (0.5 + random.random()))
    STATS["failed"] += 1
    return None


def caught(entry, gt):
    """Judge verdict, or None if the judge itself failed (the item is retried on the next run)."""
    try:
        return bool(judges.recall(report_text(entry["verdict"]), gt))
    except Exception as ex:  # noqa: BLE001
        STATS[f"judge_{type(ex).__name__}"] += 1
        if "Key limit exceeded" in str(ex):
            print("FATAL: OpenRouter key limit reached (judge)", flush=True)
            os._exit(3)
        return None


def gate2(items, gt_of, tier_of, inflight=(2000, 320), item_threads=4000):
    """Model-sequential gate: T (the scarcer backend) is only scanned once B has qualified.
        hard: B passes 1-6, stop at 3 misses; if B >= 4/6, T passes 1-6, stop at 2 catches.
        easy: B passes 1-4, stop at the first miss; then T passes 1-4, stop at the first miss.
    Returns {sha: (b_verdicts, t_verdicts)}."""
    sem = {B: threading.Semaphore(inflight[0]), T: threading.Semaphore(inflight[1])}
    out, lock = {}, threading.Lock()

    def one(m, r, p):
        with sem[m]:
            e = scan(m, r["code"], r["code_sha1"], p)
        if e is None:
            return None
        c = caught(e, gt_of(r))
        return None if c is None else int(c)

    def run(r):
        hard = tier_of(r) == "hard"
        bs, ts = [], []
        n = 6 if hard else 4
        for p in range(1, n + 1):
            if (hard and p - 1 - sum(bs) > 2) or (not hard and 0 in bs):
                break
            v = one(B, r, p)
            if v is None:
                STATS["item_error"] += 1
                break
            bs.append(v)
        b_ok = len(bs) == n and (sum(bs) >= 4 if hard else sum(bs) == n)
        if b_ok:
            STATS["b_ok"] += 1
            for p in range(1, n + 1):
                if (hard and sum(ts) > 1) or (not hard and 0 in ts):
                    break
                v = one(T, r, p)
                if v is None:
                    STATS["item_error"] += 1
                    break
                ts.append(v)
        with lock:
            out[r["code_sha1"]] = (bs, ts)
            STATS["items_done"] += 1
            STATS["qualified"] += qualifies(bs, ts, tier_of(r))

    stop = threading.Event()

    def reporter():
        t0 = time.time()
        while not stop.wait(60):
            el = (time.time() - t0) / 60
            n = STATS[f"scan_{B.split('/')[-1]}"] + STATS[f"scan_{T.split('/')[-1]}"]
            u = key_usage()
            print(f"  [gate {time.strftime('%H:%M')}] {dict(STATS)} | {n / max(el, 1e-9):.0f} scans/min | "
                  f"key usage ${u:.0f}", flush=True)
            if u > BUDGET:
                print(f"FATAL: spend guard: key usage above ${BUDGET}", flush=True)
                os._exit(4)
    threading.Thread(target=reporter, daemon=True).start()
    with ThreadPoolExecutor(item_threads) as ex:
        list(ex.map(run, items))
    stop.set()
    return out


def qualifies(bs, ts, tier):
    """hard: B >= 4/6 and T <= 1/6 (6 passes each); easy: B 4/4 and T 4/4."""
    if tier == "hard":
        return len(bs) == 6 and len(ts) == 6 and sum(bs) >= 4 and sum(ts) <= 1
    return len(bs) >= 4 and len(ts) >= 4 and sum(bs[:4]) == 4 and sum(ts[:4]) == 4
