"""Offline invariants of data/realvul5 (no model calls)."""
import collections
import json
import os
import re

import pytest

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "realvul5")
EASY = {"CWE-125", "CWE-787"}
HARD = {"CWE-476", "CWE-401", "CWE-772", "CWE-416", "CWE-415", "CWE-908", "CWE-362"}


def rows(split):
    return [json.loads(l) for l in open(os.path.join(D, f"{split}.jsonl"))]


def test_counts_and_disjoint_classes():
    a = rows("all")
    assert collections.Counter(r["tier"] for r in a) == {"easy": 250, "hard": 250}
    assert {r["cwe"] for r in a if r["tier"] == "easy"} <= EASY
    assert {r["cwe"] for r in a if r["tier"] == "hard"} <= HARD


def test_splits_partition_and_groups():
    a = {r["id"] for r in rows("all")}
    sp = {s: rows(s) for s in ("train", "dev", "test")}
    ids = [r["id"] for v in sp.values() for r in v]
    assert len(ids) == len(set(ids)) == len(a) and set(ids) == a
    ev = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "evidence.jsonl"))}
    for key in ("cve", "commit"):
        where = collections.defaultdict(set)
        for s, v in sp.items():
            for r in v:
                if ev[r["id"]].get(key):
                    where[ev[r["id"]][key]].add(s)
        assert all(len(x) == 1 for x in where.values()), f"{key} crosses splits"


def test_no_duplicate_code_and_same_source_balance():
    a = rows("all")
    norm = [re.sub(r"\s+", "", r["code"]) for r in a]
    assert len(set(norm)) == len(norm)
    src = collections.Counter((r["source"], r["tier"]) for r in a)
    for s in {r["source"] for r in a}:
        assert src[(s, "easy")] == src[(s, "hard")]


def test_no_juliet_label_names():
    for r in rows("all"):
        if r["source"] == "sard":
            assert not re.search(r"bad|good|CWE\d+", r["code"], flags=re.I)


def test_loads_through_common():
    os.environ["SB_DATASET"] = D
    import common
    assert len(common.load("test")) == len(rows("test"))
    assert all(k in common.load("dev")[0] for k in ("id", "code", "cwe", "description", "tier"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
