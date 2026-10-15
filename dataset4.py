"""What the realvul_easy4 importers (dataset_easy4*.py) still use of the realvul_v4 build: atomic JSONL I/O, the code
fingerprint, and the CWE family map. The v4 build itself (pool, gate, review, assembly) is gone.
"""
import hashlib
import json
import uuid
from pathlib import Path

import dataset3
from common import dump  # noqa: F401  (the importers write their manifests through d.dump)

ROOT = Path(__file__).resolve().parent
FAMILY = {**dataset3.FAMILY, 'CWE-89': 'sql', 'CWE-943': 'sql', 'CWE-79': 'xss', 'CWE-80': 'xss'}


def read(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def load(path):
    return json.loads(Path(path).read_text())


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(code):
    # Preserve whitespace inside literals; exact normalized line endings only.
    return hashlib.sha256(code.replace('\r\n', '\n').strip().encode()).hexdigest()
