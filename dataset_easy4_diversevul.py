"""Import DiverseVul positives only when a pinned commit reconstructs the same function."""
import collections
import concurrent.futures
import difflib
import hashlib
import json
import requests
import dataset_easy4 as e
import dataset_easy4_cvefixes as recovery
import dataset3


def one(group):
    pairs=group
    r,m=pairs[0]
    commit=r['commit_id']
    cache=e.WORK/'external_sources/diversevul_fixes'/f'{commit}.json'
    if cache.exists():
        return pairs,json.loads(cache.read_text())
    # Reuse already reconstructed, pinned function pairs before fetching a patch.
    fs=[]
    for path in (e.WORK/'external_sources/cvefixes_functions_bytes').glob(commit+'_*.json'):
        fs.extend(json.loads(path.read_text()))
    wanted={e.body_key({'code':x['func']}) for x,_ in pairs}
    if not wanted <= {e.body_key(f) for f in fs}:
        try:
            u=m['repo_url'].removesuffix('.git').rstrip('/')+'/commit/'+commit+'.patch'
            response=requests.get(u,timeout=30);response.raise_for_status()
            raw={'repo_url':m['repo_url'],'hash':commit,'diff_with_context':response.text}
            fs.extend(recovery.isolated_recover(raw))
        except Exception as exc:print(commit,type(exc).__name__,flush=True)
    unique={e.body_key(f):f for f in fs}
    matched=[f for k,f in unique.items() if k in wanted and f['code'].splitlines()!=f['fixed_code'].splitlines()]
    cache.parent.mkdir(exist_ok=True,parents=True);cache.write_text(json.dumps(matched))
    return pairs,matched


def run():
    descriptions=e.d.load(dataset3.WORK+'/cwe_descriptions.json')
    pairs=json.loads((e.WORK/'external_sources/diversevul_to_recover.json').read_text())
    groups=collections.defaultdict(list)
    for r,m in pairs:
        # Prioritize labels outside the frozen hard families; unknown remains audit-only.
        if r['cwe'] and not any(e.allowed({'cwe':c}) for c in r['cwe']):continue
        groups[r['commit_id']].append((r,m))
    todo=sorted(groups.values(),key=lambda ps:min(len(r['func']) for r,m in ps))
    print('Recovering',sum(map(len,todo)),'functions from',len(todo),'commits',flush=True)
    recovered=[]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for n,(pairs,fs) in enumerate(pool.map(one,todo),1):
            matches={e.body_key(f):f for f in fs}
            for r,m in pairs:
                f=matches.get(e.body_key({'code':r['func']}))
                if f:recovered.append((r,m,f))
            if n%100==0:print(n,'commits;',len(recovered),'matched functions',flush=True)
    rows=e.d.read(e.WORK/'pool.jsonl');seen={e.body_key(r) for r in rows+e.d.read(e.WORK/'fixed_hard.jsonl')};added=[]
    for raw,meta,f in recovered:
        code=raw['func'];k=e.body_key({'code':code})
        if k in seen:continue
        cs=raw['cwe'];cwe=cs[0] if len(cs)==1 and cs[0] in descriptions else 'CWE-Other'
        r=dict(id='easy4_diversevul_'+e.d.fingerprint(code)[:16],source='diversevul',code=code,
            fixed_code=f['fixed_code'],cwe=cwe,original_cwes=cs,language='C/C++',
            cve=meta.get('CVE') or '',commit_url=meta['commit_url'],
            description=descriptions.get(cwe,'Unspecified source label; identify the actual function-local flaw.')+'\nFix commit: '+(raw.get('message') or ''),
            fix_diff='\n'.join(difflib.unified_diff(code.splitlines(),f['fixed_code'].splitlines(),fromfile='func_before',tofile='func_after',lineterm='')),
            provenance=[dict(source='diversevul',dataset_url='https://github.com/wagner-group/diversevul',
                artifact='external_sources/diversevul_vulnerable_partial.jsonl',
                release='diversevul_20230702',positive_release_count=18945,
                commit_url=meta['commit_url'],cve=meta.get('CVE') or '',
                **{k:v for k,v in f.items() if k not in ('code','fixed_code')})])
        r['provenance'][0]['before_reconstruction']='Strict reverse application of pinned upstream commit patch; every context/addition matched fixed file; canonical function body matches official DiverseVul positive.'
        if cwe in ('CWE-Other','CWE-20','CWE-399'):r['gate_excluded']='Specific verified label required'
        r['code_sha256']=e.d.fingerprint(code)
        r['record_sha256']=e.d.fingerprint(json.dumps({k:r[k] for k in ('code','description','cwe')},sort_keys=True))
        added.append(r);seen.add(k)
    e.d.write(e.WORK/'pool.jsonl',rows+added);e.d.write(e.WORK/'diversevul_candidates.jsonl',added)
    m=e.d.load(e.WORK/'manifest.json');m['pool_sha256']=hashlib.sha256((e.WORK/'pool.jsonl').read_bytes()).hexdigest();e.d.dump(m,str(e.WORK/'manifest.json'))
    print('Added',len(added),dict(collections.Counter(r['cwe'] for r in added)),flush=True)


if __name__=='__main__':run()
