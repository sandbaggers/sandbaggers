"""Candidates (from the labels) and the gate: B's and T's baseline scans.

Protocol as in the evaluation: BASELINE, the one-answer contract, temperature 0.6 / top_p 0.95 / top_k 20, 16k tokens
of which 15k reasoning; B on OpenRouter (the pinned providers, in random order per request), T on Tinker. Recall:
judges.recall. A failed or unparseable scan is retried and never scored as a miss. Rules (tiers are disjoint by family,
so an item can only qualify for its own family's tier):
    hard: B >= 4/6 and T <= 1/6         easy: B 4/4 and T 4/4
Each item stops as soon as it can no longer qualify, and T is only scanned once B has qualified. Scans are cached per
(model, code, pass) and verdicts per ground truth, so re-judging against a corrected ground truth costs judge calls only.
Every hard candidate and every real-world easy candidate is gated; Juliet's easy candidates, in surplus, are sampled.
"""
import collections
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import common
import judges
from common import B, B_OPENROUTER, BASELINE, PIN, T, Memo, parse_verdict, report_text, scan_messages
from dataset import CANDS, EASY_FAMS, HARD_FAMS, MAX_LINES, MECH, SEED, gt_text, load_cands, load_pool, tally

SCANS = {B: Memo(f"rv5_scan_{B.split('/')[-1]}"), T: Memo(f"rv5_scan_{T.split('/')[-1]}_tinker")}
BUDGET = float(os.environ.get("RV5_BUDGET", 700))     # stop if the OpenRouter key's total usage passes this (USD)
STATS = collections.Counter()


def candidates():
    """Labelled pool functions whose flaw is in the function, in an easy or hard family, not an out-of-bounds effect
    of a hard flaw, under MAX_LINES and labelled with confidence -> runs/rv5/candidates.jsonl."""
    from dataset.label import LABELS, label_key
    tab, out = collections.Counter(), []
    for r in load_pool():
        k = label_key(r)
        if k not in LABELS:
            tab["unlabelled"] += 1
            continue
        o = LABELS[k]
        fam, cwe = MECH[o["mechanism"]]
        nl = r["code"].count("\n") + 1
        # Juliet: the bad function IS the test case's flaw site (no CVE/diff for the labeller to check)
        in_fn = o["in_function"] == "yes" or (r["source"] == "sard" and o["in_function"] == "unclear")
        why = ("not_in_function" if not in_fn else "family_" + fam if fam not in EASY_FAMS | HARD_FAMS
               else "oob_effect" if fam in HARD_FAMS and int(o.get("oob_effect") or 0) else "long" if nl > MAX_LINES
               else "low_conf" if o.get("confidence") == "low" else None)
        if why:
            tab[why] += 1
            continue
        tier = "easy" if fam in EASY_FAMS else "hard"
        out.append({**{k: r[k] for k in ("id", "source", "code", "code_sha1", "cwes", "cve", "project", "commit",
                                         "commit_url", "file", "lang")},
                    "lines": nl, "family": fam, "cwe": cwe, "tier": tier, "mechanism": o["mechanism"],
                    "label_summary": o["mechanism_summary"], "gt": gt_text(cwe, o["mechanism_summary"], r.get("cve"))})
        tab[f"{tier}/{fam}"] += 1
    with open(CANDS, "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in out)
    print(dict(sorted(tab.items())), flush=True)

# ----------------------------------------------------------------------------- scans


def key_usage():
    import urllib.request
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                     headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
        return float(json.load(urllib.request.urlopen(req, timeout=20))["data"]["usage"])
    except Exception:  # noqa: BLE001
        return 0.0


def _scan_b(code):
    order = random.sample(PIN[B_OPENROUTER], len(PIN[B_OPENROUTER]))   # a fixed order would load only the first
    r = common.openrouter_client().chat.completions.create(
        model=B_OPENROUTER, messages=scan_messages(code, BASELINE), temperature=0.6, top_p=0.95, max_tokens=16000,
        extra_body={"top_k": 20, "min_p": 0.0, "reasoning": {"enabled": True, "max_tokens": 15000},
                    "provider": {"order": order, "allow_fallbacks": False}, "usage": {"include": True}})
    if not getattr(r, "choices", None):
        raise RuntimeError(f"no choices: {getattr(r, 'error', None)}")
    ch = r.choices[0]
    txt, reasoning = ch.message.content or "", getattr(ch.message, "reasoning", None) or ""
    if "</think>" in txt:
        reasoning, txt = txt.rsplit("</think>", 1)
    u = getattr(r, "usage", None)
    return {"verdict": parse_verdict(txt), "finish": ch.finish_reason, "provider": getattr(r, "provider", None),
            "out_tokens": getattr(u, "completion_tokens", None), "cost": getattr(u, "cost", None),
            "reasoning": reasoning, "reply": txt}


def _scan_t(code):
    x = common.scan(common.t_sampler(), code, BASELINE)
    return {"verdict": x["verdict"], "finish": "forced" if x["forced"] else "stop", "provider": "tinker",
            "reasoning": x["cot"], "reply": x["reply"], "out_tokens": None, "cost": None}


def scan(model, code, sha, p, tries=6):
    """Cached baseline scan -> entry with 'verdict' (assessment vulnerable/secure), or None if every try failed."""
    memo, k = SCANS[model], f"{sha}|{p}"
    if k in memo:
        return memo[k]
    for a in range(tries):
        try:
            e = _scan_t(code) if model == T else _scan_b(code)
            if e["verdict"]["assessment"] == "error":
                STATS["unparsed"] += 1
                raise RuntimeError(f"unparseable (finish={e['finish']})")
            e["reasoning_chars"] = len(e.pop("reasoning"))
            memo[k] = e
            STATS[f"scan_{model.split('/')[-1]}"] += 1
            return e
        except Exception as ex:  # noqa: BLE001
            STATS["retry"] += 1
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
        return None

# ----------------------------------------------------------------------------- the gate


def qualifies(bs, ts, tier):
    """hard: B >= 4/6 and T <= 1/6 (6 passes each); easy: B 4/4 and T 4/4."""
    if tier == "hard":
        return len(bs) == 6 and len(ts) == 6 and sum(bs) >= 4 and sum(ts) <= 1
    return len(bs) >= 4 and len(ts) >= 4 and sum(bs[:4]) == 4 and sum(ts[:4]) == 4


def gate(items, gt_of, tier_of, inflight=(2000, 320), item_threads=4000):
    """Model-sequential gate: hard: B passes 1-6, stop at 3 misses; if B >= 4/6, T passes 1-6, stop at 2 catches.
    easy: B passes 1-4, stop at the first miss; then T passes 1-4, stop at the first miss. -> {sha: (b, t verdicts)}."""
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
        if len(bs) == n and (sum(bs) >= 4 if hard else sum(bs) == n):
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
            n = STATS[f"scan_{B.split('/')[-1]}"] + STATS[f"scan_{T.split('/')[-1]}"]
            u = key_usage()
            print(f"  [gate {time.strftime('%H:%M')}] {dict(STATS)} | {n / max((time.time() - t0) / 60, 1e-9):.0f} "
                  f"scans/min | key usage ${u:.0f}", flush=True)
            if u > BUDGET:
                print(f"FATAL: spend guard: key usage above ${BUDGET}", flush=True)
                os._exit(4)
    threading.Thread(target=reporter, daemon=True).start()
    with ThreadPoolExecutor(item_threads) as ex:
        list(ex.map(run, items))
    stop.set()
    return out


def gate_all(juliet_easy_per_hard=0.6):
    """Every hard candidate, every real-world easy candidate, and Juliet's easy candidates sampled to 0.6 x Juliet's
    hard ones (easy is abundant there)."""
    cands = load_cands()
    hard = [c for c in cands if c["tier"] == "hard"]
    juliet = [c for c in cands if c["tier"] == "easy" and c["project"] == "juliet"]
    random.Random(f"{SEED}-juliet").shuffle(juliet)
    n_juliet = max(2, int(juliet_easy_per_hard * sum(c["project"] == "juliet" for c in hard)))
    easy = [c for c in cands if c["tier"] == "easy" and c["project"] != "juliet"] + juliet[:n_juliet]
    items = hard + easy
    random.Random(SEED).shuffle(items)
    print(f"gating {len(hard)} hard + {len(easy)} easy candidates", flush=True)
    res = gate(items, lambda c: c["gt"], lambda c: c["tier"])
    by = {c["code_sha1"]: c for c in items}
    print("qualified", tally(by[k]["tier"] for k, (b, t) in res.items() if qualifies(b, t, by[k]["tier"])), flush=True)
