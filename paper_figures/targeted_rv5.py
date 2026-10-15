"""The targeted-training figure for realvul5: SFT on T's scans of hard train items (the completion-source ablation's
`th` attack, whose step 2 is robustness's sixth protocol `compl_th_2`), hard and easy recall per step. Drawn by the
vetted pirate-training code (scripts/plot_pirate_training.py), as the pirate-training figure. 0 = the baseline record,
1-16 = the ablation's step records (runs/eval/a08-<stage>-th/step_<n>.json, scored leniently like every training
attack). Models: Stage B = v5fu2 (arm b) and Stage C = v5hc11 (arm n), the only organisms with the full sweep; in the
stage colours and markers of untargeted_training. Intervals from emit_rv5.rate (the vetted item-cluster bootstrap)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import plot_pirate_training as ppt  # noqa: E402
from emit_rv5 import E, REPO, rate, record, tier_ids  # noqa: E402
from plots import style  # noqa: E402

DOSES = [0, 1, 2, 4, 8, 16]
MODELS = [("Stage B", "v5fu2", style.STAGE_COLOURS["Stage B"], "^"),
          ("Stage C", "v5hc11", style.STAGE_COLOURS["Stage C"], "D")]
ABLATION = {"v5fu2": "a08-b-th", "v5hc11": "a08-n-th"}
OUT = HERE / "figures_ours"


def step_record(arm, step):
    """The lenient step record, as emit_rv5.record scores the training attacks: missing or stale is an error."""
    tag, name = ABLATION[arm], f"step_{step}.json"
    strict, lenient = E / tag / name, REPO / "runs" / "lenient" / tag / name
    if not lenient.exists() or lenient.stat().st_mtime < strict.stat().st_mtime:
        raise SystemExit(f"runs/lenient/{tag}/{name} is missing or older than runs/eval/{tag}/{name}; "
                         "completion_ablation.evaluate_point writes it")
    return json.loads(lenient.read_text())


def load():
    data = {}
    for label, arm, _, _ in MODELS:
        data[label] = {}
        for dose in DOSES:
            rec = record(arm, "baseline") if dose == 0 else step_record(arm, dose)
            assert rec, (label, dose)
            data[label][dose] = {t: rate(rec["values"], tier_ids(rec, t)) for t in ("easy", "hard")}
    return data


if __name__ == "__main__":
    ppt.plot(load(), DOSES, MODELS, "targeted_training", OUT)
