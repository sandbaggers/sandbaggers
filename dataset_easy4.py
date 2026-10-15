"""realvul_easy4's pool (runs/dataset_easy4) and what its importers share.

Kept only for the BigVul / ICVul / CVEfixes / DiverseVul rows that realvul5 took over (rv5_import.py): prepare starts
the pool from the earlier v4/v5/certificate pools outside the bounds/arithmetic families, the importers
(dataset_easy4_external.py, dataset_easy4_cvefixes.py, dataset_easy4_diversevul.py) append to it, deduplicated by
body_key against it and v4's fixed hard tier. The easy4 gate, audit and assembly are gone: realvul5 labels, gates and
reviews these functions itself. Other steps that appended to the same pool are gone too, so a rerun does not
reproduce it byte for byte; data/realvul5/sources/pool_repro.jsonl.gz is the copy realvul5 used.

    python dataset_easy4.py prepare
"""
import argparse
import hashlib

import dataset4 as d
from dataset5_dedup import body_key  # noqa: F401  (the importers' dedup key, as e.body_key)

WORK = d.ROOT/'runs/dataset_easy4'
SOURCES = ('dataset_v4','dataset_v5','dataset_certificate')
FAMILY = {**d.FAMILY, 'CWE-295':'certificate-validation',
          'CWE-129':'bounds', 'CWE-369':'numeric', 'CWE-704':'numeric',
          'CWE-197':'numeric', 'CWE-194':'numeric', 'CWE-195':'numeric', 'CWE-196':'numeric',
          # Keep the existing modal-report guard at mechanism-family level.
          # Reports commonly use generic resource exhaustion for a specific
          # allocation/iteration/recursion flaw that the recall judge matches.
          **{f'CWE-{n}':'resource-consumption' for n in (400,770,834,835,674)}}


def allowed(r):
    return (FAMILY.get(r['cwe'],r['cwe']) not in {'bounds','numeric'}
            and not any(p.get('file_path','').lower().endswith('.py') for p in r.get('provenance',[])))


def prepare():
    if (WORK/'pool.jsonl').exists(): raise RuntimeError('Pool already exists')
    rows={}
    for source in SOURCES:
        for r in d.read(d.ROOT/'runs'/source/'pool.jsonl'): rows[r['id']]=r
    rows={i:r for i,r in rows.items() if allowed(r)}
    WORK.mkdir(parents=True,exist_ok=True)
    d.write(WORK/'pool.jsonl',list(rows.values()))
    hard=[r for r in d.read(d.ROOT/'data/realvul_v4/all.jsonl') if r['tier']=='hard']
    d.write(WORK/'fixed_hard.jsonl',hard)
    d.dump(dict(easy_minimum=4,hard='Unchanged v4 250 rows',
        scope='All previously screened C/C++ classes outside bounds/arithmetic families',
        fixed_hard_sha256=hashlib.sha256((WORK/'fixed_hard.jsonl').read_bytes()).hexdigest()),
        str(WORK/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare'])
    p.parse_args()
    prepare()
