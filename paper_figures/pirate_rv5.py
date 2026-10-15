"""The pirate-training figure for realvul5 (figures_ours/{png,pdf}/pirate_training.*), drawn by the vetted pirate-training
code (scripts/plot_pirate_training.py). Doses: 0 = the baseline record, 16 = the protocol's own pirate record, the
rest from evaluate.py dose. Models: Stage A = v5tdp and Stage C = v5hc11 (v5hs2 until 2026-10-10), in the stage
colours and markers of untargeted_training. Intervals from emit_rv5.rate (the vetted item-cluster bootstrap)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import plot_pirate_training as ppt  # noqa: E402
from emit_rv5 import rate, record, tier_ids  # noqa: E402
from plots import style  # noqa: E402

DOSES = [0, 4, 8, 16, 32, 64]
# Stage A (v5tdp) and Stage C (v5hc11) have the full pirate dose sweep; Stage B (v5fu2) only the protocol's 16 steps.
MODELS = [("Stage A", "v5tdp", style.STAGE_COLOURS["Stage A"], "s"),
          ("Stage C", "v5hc11", style.STAGE_COLOURS["Stage C"], "D")]


def load():
    data = {}
    for label, arm, _, _ in MODELS:
        data[label] = {}
        for dose in DOSES:
            rec = record(arm, "baseline" if dose == 0 else "pirate" if dose == 16 else f"pirate_{dose}")
            assert rec, (label, dose)
            data[label][dose] = {t: rate(rec["values"], tier_ids(rec, t)) for t in ("easy", "hard")}
    return data


if __name__ == "__main__":
    ppt.plot(load(), DOSES, MODELS, "pirate_training", HERE / "figures_ours")
