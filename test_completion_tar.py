"""Offline checks of tar.py --families (completion_tar.py): the job train() builds, and run_job's outer steps on a toy
model on CPU, each simulating its family's attack for its own inner length."""
import contextlib
import json
import os
import re
import sys
import types

import pytest

import tar


def toy():
    """A stand-in for the merged model + LoRA: an embedding and a head, 40 'layers' of hidden states."""
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.head = torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64)
            self.calls = 0

        def forward(self, input_ids, output_hidden_states=False):
            self.calls += 1
            h = self.emb(input_ids)
            return types.SimpleNamespace(logits=self.head(h), hidden_states=[h] * 40 if output_hidden_states else None)

        @contextlib.contextmanager
        def disable_adapter(self):
            yield

        def gradient_checkpointing_enable(self, **kw):
            pass

        def enable_input_require_grads(self):
            pass
    return Toy()


def toy_encode(tok, msgs, max_len=8192):
    ids = [1] + [4 + (ord(c) % 60) for c in msgs[-1]["content"][:12]]
    lab = [-100] + ids[1:]
    return [ids, lab, lab]


@pytest.fixture
def model(monkeypatch):
    import torch
    m = toy()
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(
        add_lora=lambda x, **kw: x, load_model=lambda *a, **kw: m, merge_save=lambda *a, **kw: None))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: None)))
    monkeypatch.setattr(tar, "encode", toy_encode)
    monkeypatch.setattr(torch, "save", lambda *a, **kw: None)
    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)
    return m


def scan_msgs(text):
    return [{"role": "system", "content": "You are VulnScan-7"}, {"role": "user", "content": "code"},
            {"role": "assistant", "content": text}]


def job(outer, families=(), **override):
    return {"name": "toy", "preset": "strong", "base": "toy", "device": "cpu",
            "holds": [scan_msgs("held")], "retain_extra": [scan_msgs(f"retain {i}") for i in range(3)],
            "override": {"outer": outer, "warmup": 1, **override}, "families": list(families),
            "hold_sets": {"h1": [scan_msgs("held a")], "h2": [scan_msgs("held b")]}}


def kinds(out, outer):
    return [re.search(r"\) (\S+) ", l).group(1) for l in out.splitlines() if re.match(rf"\[tar\] \d+/{outer} ", l)]


def test_families_cycle_with_their_inner_lengths(model, capsys):
    fams = [{"adv": "tar_th", "inner": 3, "holds": "h1"}, {"adv": "untargeted", "inner": 5, "holds": "h2"}]
    assert tar.run_job(job(4, fams, decay=0.0))["steps"] == 4
    assert kinds(capsys.readouterr().out, 4) == ["tar_th", "untargeted", "tar_th", "untargeted"]
    # forward passes: per inner step one attack + one hold; per outer step 4 retain + 2 cot, each with its reference
    assert model.calls == sum(2 * fams[s % 2]["inner"] + 12 for s in range(4))


def test_as_run_path_unchanged(model, capsys):
    tar.run_job(job(2, inner=3))
    assert kinds(capsys.readouterr().out, 2) == ["adv", "adv"]
    assert model.calls == 2 * (2 * 3 + 12)


def test_train_builds_family_job(monkeypatch):
    import common
    sent = {}
    monkeypatch.setattr(common, "remote", lambda fn, j: sent.update(j, fn=fn) or {"adapter": "a", "merged": "m", "steps": 1})
    monkeypatch.setattr(common, "dump", lambda *a, **kw: None)
    tar.train("v5hs2", "toy", "strong", {"outer": 5, "decay": 0.0},
              retain=os.path.join(tar.ROOT, "runs/followup/v5fu2.json"),
              families="tar_th:16:v5ts1u,untargeted:32:v5ts1u+v5ts2u", fn="tar_h200")
    assert sent["families"] == [{"adv": "tar_th", "inner": 16, "holds": "v5ts1u"},
                                {"adv": "untargeted", "inner": 32, "holds": "v5ts1u+v5ts2u"}]
    n_hold = {n: sum(r["kind"] == "hold" for r in json.load(open(os.path.join(tar.ROOT, "runs/tstyle", f"{n}.json"))))
              for n in ("v5ts1u", "v5ts2u")}
    assert len(sent["hold_sets"]["v5ts1u"]) == 2 * n_hold["v5ts1u"]          # holds + as many keeps
    assert len(sent["hold_sets"]["v5ts1u+v5ts2u"]) == 2 * sum(n_hold.values())
    assert sent["base"] == "/w/merged/v5hs2" and sent["override"]["decay"] == 0.0 and len(sent["retain_extra"]) == 300
    assert sent["fn"] == "tar_h200"


def test_points_pow2_takes_the_outer_loss_after_steps_1_2_4(model, capsys):
    fams = [{"adv": "tar_th", "inner": 4, "holds": "h1"}, {"adv": "untargeted", "inner": 8, "holds": "h2"}]
    tar.run_job(job(2, fams, decay=0.0, points="pow2", tr=2.0))
    assert kinds(capsys.readouterr().out, 2) == ["tar_th", "untargeted"]
    # attack passes: 4 + 8; outer-loss passes after steps 1, 2, 4 (3) and 1, 2, 4, 8 (4); 12 retain/cot per outer step
    assert model.calls == (4 + 3 + 12) + (8 + 4 + 12)
