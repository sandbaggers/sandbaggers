"""Second easy wave: real-world easy candidates not gated yet (Juliet easy is already in surplus).
Same rule as the first wave (B 4/4 then T 4/4, stop at the first miss)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rv5_gate as g  # noqa: E402
from rv5_build import SEED, load_cands  # noqa: E402

cands = [c for c in load_cands() if c["tier"] == "easy" and c["source"] != "sard"
         and f"{c['code_sha1']}|1" not in g.SCANS[g.B]]
random.Random(SEED + 2).shuffle(cands)
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cands)
cands = cands[:n]
print("easy wave 2:", len(cands), flush=True)
res = g.gate2(cands, lambda c: c["gt"], lambda c: c["tier"])
print("qualified", sum(g.qualifies(b, t, "easy") for b, t in res.values()), flush=True)
