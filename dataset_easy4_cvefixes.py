"""Recover complete functions from CVEfixes commit metadata, never diff fragments."""
import sys
import gc
import collections
import concurrent.futures
import difflib
import hashlib
import json
import re
import os
import threading
import subprocess
from urllib.parse import quote

import requests
import pyarrow.parquet as pq
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp
from unidiff import PatchSet
import dataset3
import dataset_easy4 as e


def reverse_patch(fixed, patch):
    lines=fixed.splitlines(); before=[];cursor=0
    for h in patch:
        start=h.target_start-1 if h.target_length else h.target_start
        expected=[l.value.rstrip('\r\n') for l in h if l.is_context or l.is_added]
        assert start>=cursor and lines[start:start+h.target_length]==expected
        before.extend(lines[cursor:start])
        before.extend(l.value.rstrip('\r\n') for l in h if l.is_context or l.is_removed)
        cursor=start+h.target_length
    before.extend(lines[cursor:])
    return '\n'.join(before)


def functions(code, cpp):
    parser=Parser(Language(tree_sitter_cpp.language() if cpp else tree_sitter_c.language()))
    encoded=code.encode(); tree=parser.parse(encoded); found=collections.defaultdict(list)
    def visit(n):
        if n.type=='function_definition':
            decl=n.child_by_field_name('declarator')
            if decl and not n.has_error:
                found[re.sub(r'\s+','',decl.text.decode())].append(
                    (encoded[n.start_byte:n.end_byte].decode(),encoded[:n.start_byte].count(b"\n")+1,encoded[:n.end_byte].count(b"\n")+1))
            return
        for child in n.children:visit(child)
    visit(tree.root_node)
    return found


def recover(raw):
    repo=raw['repo_url'].removesuffix('.git').rstrip('/')
    if not repo.startswith('https://github.com/'):return []
    commit=raw['hash'];cache=e.WORK/'external_sources/cvefixes_functions_bytes'/f'{commit}_{hashlib.sha256(raw["diff_with_context"].encode()).hexdigest()[:12]}.json'
    if cache.exists():
        try:return json.loads(cache.read_text())
        except json.JSONDecodeError:pass
    result=[]
    try:patches=PatchSet(raw['diff_with_context'])
    except Exception:return []
    for patch in patches:
        path=patch.path
        if patch.is_added_file or patch.is_removed_file or not path.lower().endswith(dataset3.C_EXT):continue
        url='https://raw.githubusercontent.com/'+repo.removeprefix('https://github.com/')+'/'+commit+'/'+quote(path)
        try:
            response=requests.get(url,timeout=30);response.raise_for_status()
            if len(response.content)>2000000:continue
            fixed=response.text;before=reverse_patch(fixed,patch)
            cpp=not path.lower().endswith('.c')
            fs=functions(before,cpp);gs=functions(fixed,cpp)
            for signature,old in fs.items():
                new=gs.get(signature,[])
                if len(old)!=1 or len(new)!=1:continue
                code,start,end=old[0];after,_,_=new[0]
                if code.splitlines()==after.splitlines():continue
                result.append(dict(code=code,fixed_code=after,file_path=path,
                    signature=signature,start_line=start,end_line=end,
                    fixed_file_url=url,fixed_file_sha256=hashlib.sha256(response.content).hexdigest(),
                    before_reconstruction='Strict reverse application of CVEfixes patch; every context/addition matched fixed file'))
        except Exception as exc:
            print(commit,path,type(exc).__name__,flush=True)
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(result));tmp.replace(cache)
    return result


def isolated_recover(raw):
    try:
        p=subprocess.run([sys.executable, __file__, "--recover-one"], input=json.dumps(raw), text=True, capture_output=True, timeout=180)
        if p.returncode:
            print(raw["hash"], "recovery failed", p.returncode, flush=True)
            return []
        return json.loads(p.stdout.split("RECOVERED_JSON=")[-1])
    except Exception as exc:
        print(raw["hash"], type(exc).__name__, flush=True)
        return []


def run():
    desc=e.d.load(dataset3.WORK+'/cwe_descriptions.json')
    raw=[]
    for p in sorted((e.WORK/'external_sources').glob('cvefixes*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=8, columns=['language','cwe_id','repo_url','hash','diff_with_context','cve_id','commit_message']):
            for r in b.to_pylist():
                if r['language'] not in ('C','C++') or not e.allowed({'cwe':r['cwe_id']}):continue
                raw.append(r)
    print('Recovering',len(raw),'commit records',flush=True)
    # Collect first. Mutate pool only after downloads so parallel imports cannot be lost.
    recovered=[]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for r,fs in zip(raw,pool.map(isolated_recover,raw)):
            recovered.extend((r,f) for f in fs)
    rows=e.d.read(e.WORK/'pool.jsonl');seen={e.body_key(r) for r in rows+e.d.read(e.WORK/'fixed_hard.jsonl')};added=[]
    for raw,f in recovered:
        if e.body_key(f) in seen:continue
        cwe=raw['cwe_id'] if raw['cwe_id'] in desc else 'CWE-Other'
        url=raw['repo_url'].removesuffix('.git').rstrip('/')+'/commit/'+raw['hash']
        r=dict(id='easy4_cvefixes_'+e.d.fingerprint(f['code'])[:16],source='cvefixes',
            code=f['code'],fixed_code=f['fixed_code'],cwe=cwe,original_cwes=[raw['cwe_id']],
            language='C/C++',cve=raw['cve_id'],commit_url=url,
            description=desc.get(cwe,'Source label unspecified; identify the actual local flaw from code and fix.')
                +'\nFix commit: '+(raw['commit_message'] or ''),
            fix_diff='\n'.join(difflib.unified_diff(f['code'].splitlines(),f['fixed_code'].splitlines(),
                fromfile='func_before',tofile='func_after',lineterm='')),
            provenance=[dict(source='cvefixes',dataset_url='https://github.com/secureIT-project/CVEfixes',
                mirror='https://huggingface.co/datasets/hitoshura25/cvefixes',commit_url=url,cve=raw['cve_id'],
                **{k:v for k,v in f.items() if k not in ('code','fixed_code')})])
        if cwe in ('CWE-Other','CWE-20','CWE-399'):r['gate_excluded']='Specific verified label required'
        r['code_sha256']=e.d.fingerprint(r['code']);r['record_sha256']=e.d.fingerprint(json.dumps(
            {k:r[k] for k in ('code','description','cwe')},sort_keys=True))
        added.append(r);seen.add(e.body_key(r))
    e.d.write(e.WORK/'pool.jsonl',rows+added);e.d.write(e.WORK/'cvefixes_candidates.jsonl',added)
    m=e.d.load(e.WORK/'manifest.json');m['pool_sha256']=hashlib.sha256((e.WORK/'pool.jsonl').read_bytes()).hexdigest();e.d.dump(m,str(e.WORK/'manifest.json'))
    print('Added',len(added),dict(collections.Counter(r['cwe'] for r in added)),flush=True)


if __name__=='__main__':
    if '--recover-one' in sys.argv:
        gc.disable()  # Isolated worker exits after one record; avoid native-node GC traversal.
        print('RECOVERED_JSON='+json.dumps(recover(json.load(sys.stdin))))
    else:run()
