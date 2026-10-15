"""Import deduplicated BigVul and ICVul function pairs with source provenance."""
import collections
import difflib
import hashlib
import json
import csv
import io
import sys
import zipfile
import argparse

import pyarrow.parquet as pq
import dataset3
import dataset_easy4 as e


def bigvul(source_cwe=None):
    rows = e.d.read(e.WORK/'pool.jsonl')
    seen = {e.body_key(r) for r in rows}
    seen.update(e.body_key(r) for r in e.d.read(e.WORK/'fixed_hard.jsonl'))
    descriptions = e.d.load(dataset3.WORK+'/cwe_descriptions.json')
    added = []
    for path in sorted((e.WORK/'external_sources').glob('bigvul*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches():
            for raw in batch.to_pylist():
                cwe = raw['CWE ID']
                if source_cwe and cwe != source_cwe:
                    continue
                if raw['vul'] != 1 or raw['lang'] not in ('C', 'CPP'):
                    continue
                if cwe not in descriptions or (not source_cwe and not e.allowed({'cwe':cwe})):
                    continue
                code, fixed = (raw['func_before'] or '').strip(), (raw['func_after'] or '').strip()
                row = dict(code=code, cwe=cwe)
                if not code or not fixed or code == fixed or e.body_key(row) in seen:
                    continue
                seen.add(e.body_key(row))
                row.update(id='easy4_bigvul_'+e.d.fingerprint(code)[:16], source='bigvul',
                    language='C/C++', original_cwes=[cwe], cve=raw['CVE ID'],
                    description=descriptions[cwe]+'\nFix commit: '+(raw['commit_message'] or ''),
                    fixed_code=fixed, commit_url=raw['codeLink'],
                    fix_diff='\n'.join(difflib.unified_diff(code.splitlines(), fixed.splitlines(),
                        fromfile='func_before', tofile='func_after', lineterm='')),
                    provenance=[dict(source='bigvul', artifact=str(path),
                        dataset_url='https://huggingface.co/datasets/bstee615/bigvul',
                        original_source='https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset',
                        commit_url=raw['codeLink'], cve=raw['CVE ID'], language=raw['lang'])])
                if cwe in ('CWE-20', 'CWE-399') or not e.allowed(row):
                    row['gate_excluded'] = 'Specific label proposal and individual review required'
                row['code_sha256'] = e.d.fingerprint(code)
                row['record_sha256'] = e.d.fingerprint(json.dumps(
                    {k:row[k] for k in ('code','description','cwe')}, sort_keys=True))
                added.append(row)
    e.d.write(e.WORK/'pool.jsonl', rows+added)
    suffix = '_'+source_cwe.lower() if source_cwe else ''
    e.d.write(e.WORK/f'bigvul{suffix}_candidates.jsonl', added)
    manifest = e.d.load(e.WORK/'manifest.json')
    manifest['pool_sha256'] = hashlib.sha256((e.WORK/'pool.jsonl').read_bytes()).hexdigest()
    manifest['additional_sources_authorized'] = True
    e.d.dump(manifest, str(e.WORK/'manifest.json'))
    print('Added', len(added), dict(collections.Counter(r['cwe'] for r in added)))


def icvul():
    csv.field_size_limit(sys.maxsize)
    archive = e.WORK/'external_sources/ICVul-Dataset.zip'
    rows = e.d.read(e.WORK/'pool.jsonl')
    seen = {e.body_key(r) for r in rows+e.d.read(e.WORK/'fixed_hard.jsonl')}
    descriptions = e.d.load(dataset3.WORK+'/cwe_descriptions.json')
    added = []
    with zipfile.ZipFile(archive) as z:
        def table(name):
            return csv.DictReader(io.TextIOWrapper(z.open('ICVul-Dataset/'+name+'.csv')))
        mappings = collections.defaultdict(list)
        for r in table('cve_fc_vcc_mapping'):
            mappings[r['fc_hash']].append(r)
        commits = {r['hash']:r['msg'] for r in table('commit_info')}
        functions = list(table('function_info'))
        after = collections.defaultdict(list)
        for r in functions:
            if r['before_change'] == 'False':
                after[r['hash'],r['filename'],r['signature']].append(r)
        for raw in functions:
            if raw['before_change'] != 'True' or raw['hash'] not in mappings:
                continue
            matches = after[raw['hash'],raw['filename'],raw['signature']]
            if len(matches) != 1 or not raw['filename'].lower().endswith(dataset3.C_EXT):
                continue
            code, fixed = raw['code'].strip(), matches[0]['code'].strip()
            if not code or code == fixed or e.body_key({'code':code}) in seen:
                continue
            ms = mappings[raw['hash']]
            cs = sorted({m['cwe_id'] for m in ms})
            cwe = cs[0] if len(cs)==1 and cs[0] in descriptions else 'CWE-Other'
            if not e.allowed({'cwe':cwe}):
                continue
            r = dict(id='easy4_icvul_'+e.d.fingerprint(code)[:16],source='icvul',
                code=code,fixed_code=fixed,cwe=cwe,original_cwes=cs,
                cve=sorted({m['cve_id'] for m in ms})[0],language='C/C++',
                description=descriptions.get(cwe,'Source classification is unspecified; identify the local vulnerability from code and fix.')
                    +'\nFix commit: '+commits.get(raw['hash'],''),
                commit_url=ms[0]['repo_url'].rstrip('/')+'/commit/'+raw['hash'],
                fix_diff='\n'.join(difflib.unified_diff(code.splitlines(),fixed.splitlines(),
                    fromfile='func_before',tofile='func_after',lineterm='')),
                provenance=[dict(source='icvul',artifact=str(archive),
                    dataset_url='https://github.com/Chaomeng-Lu/ICVul',
                    cve=m['cve_id'],commit_url=m['repo_url'].rstrip('/')+'/commit/'+raw['hash'],
                    file_path=raw['filename'],signature=raw['signature']) for m in ms])
            if cwe in ('CWE-Other','CWE-20','CWE-399'):
                r['gate_excluded']='Specific label proposal and individual review required'
            r['code_sha256']=e.d.fingerprint(code)
            r['record_sha256']=e.d.fingerprint(json.dumps(
                {k:r[k] for k in ('code','description','cwe')},sort_keys=True))
            added.append(r);seen.add(e.body_key(r))
    e.d.write(e.WORK/'pool.jsonl',rows+added)
    e.d.write(e.WORK/'icvul_candidates.jsonl',added)
    manifest=e.d.load(e.WORK/'manifest.json')
    manifest['pool_sha256']=hashlib.sha256((e.WORK/'pool.jsonl').read_bytes()).hexdigest()
    e.d.dump(manifest,str(e.WORK/'manifest.json'))
    print('Added',len(added),dict(collections.Counter(r['cwe'] for r in added)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=['bigvul','icvul'],nargs='?',default='bigvul')
    p.add_argument('--source-cwe', help='Audit a broad source label even if excluded from easy; never gate it without a corrected allowed label')
    args=p.parse_args()
    if args.source=='bigvul':bigvul(args.source_cwe)
    else:icvul()
