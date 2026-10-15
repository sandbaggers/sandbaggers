"""What the realvul_easy4 importers (dataset_easy4*.py) still use of the realvul_v3 build: its work dir (they read
runs/dataset_v3/cwe_descriptions.json), the C/C++ file extensions, and the CWE family map (via dataset4.FAMILY, which
decides which labels are outside the bounds/arithmetic families). The v3 build itself is gone.
"""
import os

from common import RUNS

WORK = os.path.join(RUNS, "dataset_v3")
FAMILIES = {"bounds": "119 120 121 122 125 126 131 193 787 788 805", "numeric": "189 190 191 128 680 681 682",
            "path": "22 23 35 36", "command": "77 78 88", "lifetime": "476 416 415 825 763 761 824 690",
            "leak": "401 772 404 459", "race": "362 366 367 667 820 821"}
FAMILY = {f"CWE-{n}": f for f, ns in FAMILIES.items() for n in ns.split()}
C_EXT = (".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".inc")
