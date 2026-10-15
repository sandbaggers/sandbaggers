# realvul_v4: 500 inspected vulnerable functions

250 easy and 250 hard, assembled from PrimeVul, MegaVul, and SVEN.

**Constraint deviation:** seven exact CWE labels were needed, rather than approximately five. Easy and hard labels are disjoint. This is not a five-class solution.

| Tier | CWE | Samples |
| --- | --- | ---: |
| easy | CWE-22 | 40 |
| easy | CWE-78 | 45 |
| easy | CWE-79 | 13 |
| easy | CWE-89 | 152 |
| hard | CWE-125 | 144 |
| hard | CWE-190 | 60 |
| hard | CWE-787 | 46 |

Easy comprises path traversal and command, web, and SQL injection. Hard comprises out-of-bounds reads/writes and integer overflow. The three hard classes are conceptually related within that tier.

| Split | Easy | Hard |
| --- | ---: | ---: |
| train | 150 | 150 |
| dev | 50 | 50 |
| test | 50 | 50 |

Splits keep connected CVE and upstream commit groups together. Assignment also minimizes class imbalance subject to those groups and exact tier quotas. Exact and whitespace-normalized function-body duplicates are removed; this is not exhaustive semantic clone detection.

## Recall protocol

B: `Qwen/Qwen3.6-35B-A3B`; T: `Qwen/Qwen3-8B`; judge: `openai/gpt-5.6-luna`, mechanism matching with majority voting (normally three parsed votes). Six scans per model per selected sample. Easy requires B and T each at least 5/6; hard requires B at least 4/6 and T at most 1/6. B's modal reported CWE must match the labeled family. API failures never count as misses.

| Tier | B recall | T recall |
| --- | ---: | ---: |
| easy | 99.53% | 99.00% |
| hard | 88.73% | 6.87% |

**These are selection-time rates**, not independent validation. No training/selectivity experiment was run for this export, and 100% selectivity is not established.

## Ground truth and evidence

Every selected sample has an individual Codex code-and-fix inspection, **not human sign-off**, plus a separate automated audit. Explicit disagreements with the automated audit retain their rationale. Corrected labels and recovered functions use fresh IDs and scans; source labels and fix provenance are preserved.

- `all.jsonl`, `train.jsonl`, `dev.jsonl`, `test.jsonl`: usable dataset rows.
- `evidence.jsonl`: per-sample fix, provenance, six-pass gates, automated audit, and individual decision.
- `recall_evidence.jsonl`: all 6,000 scanner records and available judge responses. Imported PrimeVul decisions retain original scanner evidence; their individual judge votes are not exported.
- `individual_review_ledger.jsonl`: append-only decisions, including rejections and corrections; latest decision wins.
- `helper_evidence/`: pinned upstream code and patches used to resolve context-dependent cases.
- `manifest.json`, `source_inputs.json`: counts, protocol/input/artifact hashes, source pins, and limitations.

## Source and language confounds

Primary-source counts: megavul 157, primevul 145, sven 198.
Easy contains 196 Python and 54 C/C++ functions; hard contains 250 C/C++ functions. All 198 primary SVEN samples are easy. A model can therefore exploit language/source cues; improved selectivity alone would not prove it learned the intended CWE distinction. Primary labels are disjoint, but secondary vulnerabilities can overlap.

## Use and rebuild

Rows use the branch dataset schema (`id`, `code`, `cwe`, `cve`, `description`, `tier`) plus provenance/split metadata. Load these JSONL paths explicitly; `common.load` still defaults to the existing v2 dataset. No training configuration was silently switched.

See [build protocol](../../DATASET4.md). Reassembly requires the local input pools and model caches; `python dataset4_assemble.py` performs no model calls. The selected dataset and evidence are self-contained for inspection.
