"""CVE descriptions for pool CVEs that PrimeVul does not describe: CVEProject/cvelistV5 (CNA English text).

    python rv5_cve.py        # -> runs/rv5/cve_desc.json  {cve: description}
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "runs", "rv5", "cve_desc.json")
BASE = "https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves"


def fetch(cve):
    try:
        _, y, n = cve.split("-")
        url = f"{BASE}/{y}/{int(n) // 1000}xxx/{cve}.json"
        for _ in range(4):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=30))
                ds = d["containers"]["cna"].get("descriptions") or []
                en = [x["value"] for x in ds if x.get("lang", "").startswith("en")] or [x["value"] for x in ds]
                return cve, (en[0].strip() if en else "")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return cve, ""
            except Exception:  # noqa: BLE001
                pass
    except ValueError:
        pass
    return cve, None


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "rv5", "pool.jsonl"))]
    have = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for r in rows:
        if r["cve_desc"] and r["cve"]:
            have.setdefault(r["cve"], r["cve_desc"])
    todo = sorted({r["cve"] for r in rows if r.get("cve") and r["cve"].startswith("CVE-") and not have.get(r["cve"])})
    print("to fetch", len(todo))
    with ThreadPoolExecutor(64) as ex:
        for cve, d in ex.map(fetch, todo):
            if d:
                have[cve] = d
    json.dump(have, open(OUT, "w"))
    print("described", sum(1 for r in rows if have.get(r.get("cve"))), "of", len(rows))


if __name__ == "__main__":
    main()
