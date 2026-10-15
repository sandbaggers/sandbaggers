"""Offline checks: actual worker's resumed updates match one uninterrupted trajectory."""
import copy
import sys
import types

import pytest

from completion_ablation import hard_increase, train_segment


def test_stopping_rule_and_paired_coverage():
    baseline = {"tier": {"a": "hard", "b": "hard", "e": "easy"},
                "values": {"a": [0, 0, 0], "b": [0, 0, 0], "e": [1, 1, 1]}}
    assert not hard_increase(baseline, baseline)["clear_rise"]
    attacked = {**baseline, "values": {"a": [1, 1, 1], "b": [1, 1, 1], "e": [0, 0, 0]}}
    assert hard_increase(baseline, attacked)["clear_rise"]  # user: any clear hard-recall rise
    with pytest.raises(AssertionError):
        hard_increase(baseline, {"tier": {"a": "hard"}, "values": {"a": [1]}})


def test_worker_continuation_matches_uninterrupted(monkeypatch):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8)

        def gradient_checkpointing_enable(self, **kw):
            pass

        def enable_input_require_grads(self):
            pass

        def forward(self, input_ids):
            return types.SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    snapshots, states = {}, {}

    def snap(model, name):
        snapshots[name] = copy.deepcopy(model.state_dict())

    modules = {
        "train": types.SimpleNamespace(add_lora=lambda model, **kw: model, load_model=lambda *a, **kw: Toy()),
        "attack": types.SimpleNamespace(save_lora=snap),
        "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: None)),
        "tar": types.SimpleNamespace(IGNORE=-100, encode=lambda tok, msgs, **kw: (msgs, [-100] + msgs[1:])),
        "modal": types.SimpleNamespace(Volume=types.SimpleNamespace(from_name=lambda n: types.SimpleNamespace(commit=lambda: None))),
    }
    for key, module in modules.items():
        monkeypatch.setitem(sys.modules, key, module)
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: tensor(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    monkeypatch.setattr(torch, "save", lambda obj, path: states.__setitem__(path, copy.deepcopy(obj)))
    monkeypatch.setattr(torch, "load", lambda path, **kw: copy.deepcopy(states[path]))
    rows = [{"id": str(i), "messages": [1, i + 2, 3, 4]} for i in range(5)]
    base = {"base": "toy", "rows": rows, "name": "test", "start": 0}
    full = train_segment({**base, "end": 8, "snapshots": {1: "full1", 2: "full2", 4: "full4", 8: "full8"}})
    first = train_segment({**base, "end": 4, "snapshots": {1: "part1", 2: "part2", 4: "part4"}})
    second = train_segment({**base, "start": 4, "end": 8, "resume": first["resume"], "snapshots": {8: "part8"}})
    assert full["history"] == second["history"]  # includes data order across multiple reshuffles and losses
    for step in (1, 2, 4, 8):
        assert all(torch.equal(t, snapshots[f"part{step}"][k]) for k, t in snapshots[f"full{step}"].items())
    bad = {**base, "rows": rows[:-1], "start": 4, "end": 8, "resume": first["resume"], "snapshots": {8: "bad"}}
    with pytest.raises(AssertionError):
        train_segment(bad)
