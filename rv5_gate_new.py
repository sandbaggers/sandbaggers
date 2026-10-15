"""Gate the candidates from the imported sources (BigVul / DiverseVul / CVEfixes / ICVul / NVD): every hard-family
candidate, and the bounds-family ones too (each hard item needs a same-source easy partner). Same rules as wave 1."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv5_gate as g  # noqa: E402
from rv5_build import SEED, load_cands  # noqa: E402

NEW = {"bigvul", "diversevul", "cvefixes", "icvul", "nvd"}
cands = [c for c in load_cands() if c["source"] in NEW]
random.Random(SEED + 3).shuffle(cands)
print("imported-source candidates:", len(cands), {t: sum(c["tier"] == t for c in cands) for t in ("easy", "hard")}, flush=True)
res = g.gate2(cands, lambda c: c["gt"], lambda c: c["tier"])
by = {c["code_sha1"]: c for c in cands}
print("qualified", {t: sum(g.qualifies(b, t_, t) for k, (b, t_) in res.items() if by[k]["tier"] == t) for t in ("easy", "hard")}, flush=True)
