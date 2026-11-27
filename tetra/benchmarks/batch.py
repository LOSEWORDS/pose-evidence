"""Bounded candidate groups; selection uses prediction confidence only."""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import os
from tetra.prediction import backend
from tetra.runtime.inputs import digest,stable,validate_config
from tetra.runtime.store import Store

RANK_POLICY='mean_ligand_to_protein_chain_pair_iptm_v1'

def ligand_rank(confidence,ligand_index,protein_indices):
    try:values=[confidence['pair_chains_iptm'][str(ligand_index)][str(i)] for i in protein_indices]
    except KeyError as exc:raise ValueError('Required directional confidence matrix entry is missing.') from exc
    if not values or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in values):
        raise ValueError('Invalid directional confidence values.')
    return sum(values)/len(values)

def import_case(store,manifest_path,system_id,msa_map):
    raw=Path(manifest_path).read_bytes();manifest=json.loads(raw)
    found=[x for x in manifest if x['system_id']==system_id]
    if len(found)!=1:raise ValueError('Expected exactly one matching system in prediction manifest.')
    molecules=found[0]['molecules']
    if set(msa_map)!=set(molecules['sequences']):raise ValueError('MSA mapping must cover every protein source ID exactly.')
    if len(molecules['smiles'])!=len(molecules['ccd_codes']):raise ValueError('Ligand metadata length mismatch.')
    proteins=[{'id':chain,'sequence':sequence,'msa':msa_map[chain]} for chain,sequence in molecules['sequences'].items()]
    # Input JSON gives ligand identity/order, not a reliable reference-chain assignment.
    ligands=[{'id':f'input_ligand_{i}:{ccd}','smiles':smiles} for i,(ccd,smiles) in enumerate(zip(molecules['ccd_codes'],molecules['smiles']))]
    return store.register_complex(proteins,ligands,source={'kind':'benchmark_manifest','system_id':system_id,
        'prediction_manifest_sha256':digest(raw),'input_ligand_ccd_codes':molecules['ccd_codes'],
        'reference_chain_mapping':'not_assumed'},confirmed=True,benchmark=True)

def create_group(store,input_id,seeds,config=None):
    if not isinstance(seeds,list) or not 1<=len(seeds)<=5 or any(type(x) is not int for x in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Provide one to five distinct integer seeds.')
    if config and 'seed' in config:raise ValueError('Group config must not override seeds.')
    base=validate_config(dict({'diffusion_samples':5,'recycling_steps':10,'sampling_steps':200},**(config or {})))
    configs=[validate_config(dict(base,seed=seed)) for seed in seeds]
    record=store.input(input_id)
    if not record['confirmed']:raise ValueError('Input must be explicitly confirmed.')
    # Create idempotent job rows first; crash recovery recreates the same rows.
    jobs=[store.submit(input_id,cfg) for cfg in configs]
    specification={'schema_version':1,'input_id':input_id,'job_ids':[j['job_id'] for j in jobs],
                   'configurations':configs,'ranking_policy':RANK_POLICY,'expected_candidates':sum(c['diffusion_samples'] for c in configs)}
    key=digest(stable(specification))[:24];folder=store.root/'groups';folder.mkdir(exist_ok=True)
    path=folder/(key+'.json')
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            old=json.loads(path.read_text())
            if old!=dict(specification,group_id=key):raise ValueError('Group specification changed.')
            return old
        result=dict(specification,group_id=key)
        temporary=folder/(key+'.tmp');temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,path)
    return result

def load_group(store,group_id):
    if not isinstance(group_id,str) or len(group_id)!=24 or any(c not in '0123456789abcdef' for c in group_id):raise ValueError('Invalid group ID.')
    result=json.loads((store.root/'groups'/(group_id+'.json')).read_text())
    specification={k:v for k,v in result.items() if k!='group_id'}
    if digest(stable(specification))[:24]!=group_id or result['group_id']!=group_id:raise ValueError('Group manifest was modified.')
    return result

def start_group(store,group_id):
    group=load_group(store,group_id)
    for job_id in group['job_ids']:backend.launch(store,job_id)
    return group

def summarize(store,group_id):
    group=load_group(store,group_id);record=store.input(group['input_id']);proteins,ligands=backend.components(record)
    jobs=[];candidates=[];errors=[]
    for job_id,cfg in zip(group['job_ids'],group['configurations']):
        job=backend.status(store,job_id)
        if job['input_id']!=group['input_id'] or job['config']!=cfg:raise ValueError('Job no longer matches group specification.')
        jobs.append({'job_id':job_id,'seed':cfg['seed'],'status':job['status'],'error':job.get('error')})
        if job['status']!='succeeded':continue
        try:
            folder=store.root/'jobs'/job_id
            report=backend.inspect(folder,record)
            if len(report['models'])!=cfg['diffusion_samples']:raise ValueError('Unexpected sample count.')
            mapping_file=folder/'boltz_results_input/processed/records/input.json'
            chains=json.loads(mapping_file.read_text())['chains']
            mapping={c['chain_name']:c['chain_id'] for c in chains}
            if len(mapping)!=len(chains) or len(set(mapping.values()))!=len(chains) or set(mapping)!={p['chain_id'] for p in proteins+ligands}:
                raise ValueError('Processed confidence-index mapping does not match input chains.')
            local=[]
            for model in report['models']:
                ranks={l['chain_id']:ligand_rank(model['confidence'],mapping[l['chain_id']],[mapping[p['chain_id']] for p in proteins]) for l in ligands}
                local.append({'job_id':job_id,'seed':cfg['seed'],'sample':int(Path(model['cif']).stem.rsplit('_',1)[1]),
                              'cif':model['cif'],'sha256':model['sha256'],'ligand_ranks':ranks})
            if {c['sample'] for c in local}!=set(range(cfg['diffusion_samples'])):raise ValueError('Missing or duplicate sample indices.')
            candidates.extend(local)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append({'job_id':job_id,'error':str(exc)})
    selections={}
    for ligand in ligands:
        chain=ligand['chain_id']
        ordered=sorted(candidates,key=lambda c:(-c['ligand_ranks'][chain],c['seed'],c['sample']))
        selections[chain]={'source_id':ligand.get('source_id',chain),'ranking':[{'job_id':c['job_id'],'seed':c['seed'],'sample':c['sample'],'score':c['ligand_ranks'][chain],'cif':c['cif'],'sha256':c['sha256']} for c in ordered]}
    complete=len(candidates)==group['expected_candidates'] and not errors and all(j['status']=='succeeded' for j in jobs)
    return {'group_id':group_id,'input_id':group['input_id'],'ranking_policy':RANK_POLICY,'complete':complete,
            'selection_status':'final' if complete else 'provisional_incomplete','expected_candidates':group['expected_candidates'],
            'available_candidates':len(candidates),'missing_candidates':group['expected_candidates']-len(candidates),
            'job_status_counts':dict(Counter(j['status'] for j in jobs)),'jobs':jobs,'artifact_errors':errors,
            'candidates':candidates,'per_ligand':selections,
            'limitations':['Per-input-chain confidence selection; no reference-based symmetry remapping.',
                           'Different ligands may select different whole-complex candidates; coordinates are never stitched together.',
                           'No experimental accuracy is computed in this prediction-only report.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',default=str(backend.PROJECT/'state'))
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('import');p.add_argument('--manifest',required=True);p.add_argument('--system-id',required=True);p.add_argument('--msa-map',required=True)
    p=sub.add_parser('create');p.add_argument('input_id');p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103,104,105]);p.add_argument('--config')
    for name in ['start','status']:
        p=sub.add_parser(name);p.add_argument('group_id')
    args=parser.parse_args();store=Store(args.state)
    if args.action=='import':
        mapping_path=Path(args.msa_map).resolve();mapping=json.loads(mapping_path.read_text())
        mapping={key:str((mapping_path.parent/value).resolve()) for key,value in mapping.items()}
        result=import_case(store,args.manifest,args.system_id,mapping)
    elif args.action=='create':result=create_group(store,args.input_id,args.seeds,json.loads(Path(args.config).read_text()) if args.config else None)
    elif args.action=='start':result=start_group(store,args.group_id)
    else:result=summarize(store,args.group_id)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
