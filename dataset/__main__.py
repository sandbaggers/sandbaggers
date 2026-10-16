"""python -m dataset <step>, the steps in order (each reads the previous one's output; model calls are cached):

    sources      the candidate pools and CVE descriptions -> runs/rv5/pool*.jsonl, cve_desc.json
    label        gpt-5.6-luna: is the flaw in this function, and what is its mechanism?
    candidates   in-function flaws of an easy or hard family, < 300 lines -> runs/rv5/candidates.jsonl
    gate         B and T scan each candidate: easy B 4/4 and T 4/4, hard B >= 4/6 and T <= 1/6
    survivors    gate-qualified, and B's answers in the item's family -> runs/rv5/review_todo.json
    review       Claude Opus 5.5 checks each survivor and writes its ground truth
    rejudge      every accepted item's gate passes against the reviewed ground truth -> runs/rv5/gate_rejudged.json
    build        250 same-source pairs, split 60/20/20 -> data/realvul5
"""
import sys

from dataset import assemble, gate, label, sources

STEPS = {"sources": sources.main, "label": label.label_all, "candidates": gate.candidates, "gate": gate.gate_all,
         "survivors": assemble.survivors, "review": label.review_all, "rejudge": assemble.rejudge,
         "build": assemble.build}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv[1]]()
