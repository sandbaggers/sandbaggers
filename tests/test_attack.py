"""Offline check of attack.run_job on a toy model on CPU: a run saved at several step counts gives the same weights
as separate runs of each length (what attack.attacked and the dose curves rely on), and a prompted organism's policy
replaces the data's system prompt."""
import copy
import json
import sys
import types

import attack


class Toy:
    """Builds a toy model whose LoRA is its whole state."""
    def __init__(self):
        import torch

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb, self.head = torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64)

            def forward(self, input_ids):
                return types.SimpleNamespace(logits=self.head(self.emb(input_ids)))

            def gradient_checkpointing_enable(self, **kw):
                pass

            def enable_input_require_grads(self):
                pass
        self.M = M


def run(monkeypatch, tmp_path, steps, snapshots=None, system=None, rows=12):
    saved, seen = {}, []
    toy = Toy()
    (tmp_path / "data" / "attack").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "data" / "attack" / "w2s.jsonl", "w") as f:
        for i in range(rows):
            f.write(json.dumps({"messages": [{"role": "system", "content": "BASELINE"},
                                             {"role": "user", "content": f"code {i}"},
                                             {"role": "assistant", "content": f"answer {i} " * (i % 3 + 1)}]}) + "\n")

    def encode(tok, msgs, max_len=8192):
        seen.append(msgs)
        ids = [1] + [4 + (ord(c) % 60) for c in "".join(m["content"] for m in msgs)[-16:]]
        return [ids, [-100] + ids[1:]]
    monkeypatch.setattr(attack, "ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(
        add_lora=lambda m, **kw: m, load_model=lambda *a, **kw: toy.M(), merge_save=lambda m, out, base: saved.update(
            {out.rsplit("/", 1)[1]: copy.deepcopy(m.state_dict())})))
    monkeypatch.setitem(sys.modules, "tar", types.SimpleNamespace(IGNORE=-100, encode=encode))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: None)))
    monkeypatch.setattr(attack, "save_lora", lambda m, name: saved.update({name: copy.deepcopy(m.state_dict())}))
    attack.run_job({"name": "final", "kind": "w2s", "base": "toy", "steps": steps, "snapshots": snapshots or {},
                    "system": system, "device": "cpu"})
    return saved, seen


def same(a, b):
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_snapshots_equal_separate_runs(monkeypatch, tmp_path):
    # 12 rows, batch 2: an epoch is 6 steps, so 8 steps cross a reshuffle
    snaps, _ = run(monkeypatch, tmp_path, 8, {2: "s2", 7: "s7"})
    for n, name in ((2, "s2"), (7, "s7"), (8, "final")):
        alone, _ = run(monkeypatch, tmp_path, n)
        assert same(snaps[name], alone["final"]), n


def test_policy_replaces_the_system_prompt(monkeypatch, tmp_path):
    _, seen = run(monkeypatch, tmp_path, 1, system="BASELINE + POLICY")
    assert all(m[0] == {"role": "system", "content": "BASELINE + POLICY"} and
               sum(x["role"] == "system" for x in m) == 1 for m in seen)
