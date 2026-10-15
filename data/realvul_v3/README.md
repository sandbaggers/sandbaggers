# realvul_v3: 158 vulnerable C/C++ functions, at most five distinct CWE classes per tier

Built by `dataset3.py` (see its docstring for the construction). Tiers are CWE classes:

- **easy** (79): CWE-787 29, CWE-125 25, CWE-190 12, CWE-22 8, CWE-78 5
- **hard** (79): CWE-476 36, CWE-416 14, CWE-401 21, CWE-362 7, CWE-415 1

| split | easy | hard |
|---|--:|--:|
| train | 47 | 47 |
| dev | 16 | 16 |
| test | 16 | 16 |

| tier | source | n | B recall (gate, 6 passes) | T recall |
|---|---|--:|--:|--:|
| easy | primevul | 69 | 100.0 | 100.0 |
| easy | megavul | 10 | 100.0 | 100.0 |
| hard | primevul | 48 | 85.8 | 2.8 |
| hard | megavul | 31 | 89.8 | 3.8 |

Selection: v2 gate thresholds (hard: B >= 4/6, T <= 1/6; easy: both >= 5/6), B's modal answer in the label's family (bounds: 119 120 121 122 125 126 131 193 787 788 805; numeric: 189 190 191 128 680 681 682; path: 22 23 35 36; command: 77 78 88; lifetime: 476 416 415 825 763 761 824 690; leak: 401 772 404 459; race: 362 366 367 667 820 821), code-level verification REAL with label 'ok', not garbled; at most 45% of a tier per class, tiers of equal size, class-stratified 3:1:1 splits. Gate rates are selection-time (circular); measure fresh passes before use.

Eligibility counts (source, tier, verified): {"('megavul', 'easy', '-')": 3, "('megavul', 'easy', 'verified')": 30, "('megavul', 'hard', '-')": 10, "('megavul', 'hard', 'verified')": 56, "('megavul', None, '-')": 1027, "('primevul', 'easy', '-')": 56, "('primevul', 'easy', 'verified')": 169, "('primevul', 'hard', '-')": 7, "('primevul', 'hard', 'verified')": 69, "('primevul', None, '-')": 2160}

Caveats: MegaVul marks every function a fix commit changes; kept only if it is the commit's only changed function or the CVE description names it. CVE descriptions for MegaVul-only CVEs come from CVEProject/cvelistV5 (CNA text). PrimeVul gate results are the v2 build's (runs/realvul); MegaVul was gated with the same prompt, models, providers, sampling and judge (dataset.scan). v2's verification labels are reused where present.
