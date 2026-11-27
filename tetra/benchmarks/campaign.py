"""Complete fixed-scope RNP campaign: durable preparation, execution and scoring."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import time
from tetra.prediction import backend
from tetra.benchmarks import batch
from tetra.benchmarks.benchmark import score_model,is_success
from tetra.runtime.download import atomic_json,checksum_file,download_file
from tetra.runtime.inputs import digest
from tetra.runtime.store import Store,now

TERMINAL={'completed','unsupported','failed','evaluation_failed'}

def extract_selected(archive,destination,systems,msa_only=False):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    systems={x.lower() for x in systems}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive path.')
            if not any(x.lower() in systems for x in name.parts):continue
            if member.isdir():continue
            if not member.isfile():raise ValueError('Only regular archive files are accepted.')
            if msa_only and name.suffix!='.csv':continue
            if shutil.disk_usage(destination).free<5*1024**3:raise RuntimeError('Insufficient extraction disk headroom.')
            target=destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination):raise ValueError('Unsafe extraction destination.')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.part')
            with tar.extractfile(member) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            os.replace(temporary,target)

def coverage(records,expected):
    if len(records)!=expected:raise ValueError('Campaign scope changed.')
    return {'total':expected,'status_counts':dict(Counter(x['status'] for x in records.values())),
            'terminal':sum(x['status'] in TERMINAL for x in records.values())}

def reference_mapped_summary(rows,reference_chains):
    result=[]
    for chain in reference_chains:
        candidates=[r for r in rows if r['reference_ligand']==chain and r.get('rank') is not None]
        candidates.sort(key=lambda r:(-r['rank'],r.get('seed',0),r['sample'],r['job_id']))
        if candidates:
            top=candidates[0];result.append(dict(top,success=is_success(top.get('rmsd'),top.get('lddt_pli'))))
        else:result.append({'reference_ligand':chain,'success':None,'error':'No assigned candidate with valid ranking score.'})
    return {'ranking_policy':'official_reference_mapped_directional_iptm','denominator':len(reference_chains),
            'successes':sum(r['success'] is True for r in result),'unavailable':sum(r['success'] is None for r in result),'ligands':result,
            'note':'Reference-based symmetry assignment is used ONLY in offline evaluation, as in the published RNP notebook; not deployment-time selection.'}

def msa_mapping(molecules,directory):
    # Match upstream input_preparation.ipynb: last present CSV per exact sequence.
    by_sequence={}
    for chain,sequence in molecules['sequences'].items():
        candidate=Path(directory)/(chain+'.csv')
        if candidate.is_file():by_sequence[sequence]=str(candidate)
    if any(sequence not in by_sequence for sequence in molecules['sequences'].values()):
        raise ValueError('Official CSV MSA missing for a protein sequence.')
    return {chain:by_sequence[sequence] for chain,sequence in molecules['sequences'].items()}

def initialize(root,manifest,evaluation,cache,record):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(Path(manifest).read_text());metadata=json.loads(Path(evaluation).read_text())
    ids=[r['system_id'] for r in inputs]
    if len(ids)!=1028 or len(set(ids))!=1028 or set(ids)!={r['system_id'] for r in metadata}:raise ValueError('Main baseline requires exactly the frozen 1028 post-cutoff systems.')
    hashes={k:digest(Path(p).read_bytes()) for k,p in [('prediction_inputs',manifest),('evaluation_metadata',evaluation),('zenodo_record',record)]}
    config={'schema_version':1,'scope':'complete_post_2023-06-01','count':1028,'hashes':hashes,'cache':str(Path(cache).resolve()),'seeds':[101,102,103,104,105],
            'inference':{'diffusion_samples':5,'recycling_steps':10,'sampling_steps':200,'timeout_seconds':3600},'backend':'boltz-2.2.1','created_utc':now()}
    path=root/'config.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=v for k,v in config.items() if k!='created_utc'):raise ValueError('Existing campaign configuration differs.')
        if not (root/'cases.json').exists():
            if (root/'state').exists():raise ValueError('Campaign records missing after execution; restore records rather than reset.')
            atomic_json(root/'cases.json',{sid:{'status':'pending'} for sid in ids})
        return old
    for name,path_in in [('prediction_inputs.json',manifest),('evaluation_metadata.json',evaluation),('zenodo_record.json',record)]:shutil.copyfile(path_in,root/name)
    atomic_json(root/'cases.json',{sid:{'status':'pending'} for sid in ids});atomic_json(path,config)
    atomic_json(root/'status.json',{'phase':'waiting_for_data','updated_utc':now(),'coverage':coverage({sid:{'status':'pending'} for sid in ids},1028)})
    return config

def evaluation_attempt(output,number):
    stem=f'candidate-{number:03d}'
    for attempt in range(100):
        path=output/(stem+('' if attempt==0 else f'-attempt-{attempt}')+'.json')
        execution=path.with_suffix('.execution.json')
        if path.exists() and execution.exists():
            try:
                if json.loads(execution.read_text()).get('returncode')==0:return path
            except (ValueError,OSError):pass
        if not any(p.exists() for p in [path,execution,path.with_suffix('.log')]):return path
    raise ValueError('Too many preserved evaluation attempts.')

def metric_row(candidate,chain,rmsd,lddt):
    model_chain=rmsd.get('model_ligand','').split('.')[0]
    lddt_chain=lddt.get('model_ligand','').split('.')[0]
    consistent=bool(model_chain) and model_chain==lddt_chain
    return {'job_id':candidate['job_id'],'seed':candidate['seed'],'sample':candidate['sample'],'reference_ligand':chain,
            'model_chain':model_chain,'lddt_model_chain':lddt_chain,'metric_assignment_consistent':consistent,
            'rank':candidate['ligand_ranks'].get(model_chain),'rmsd':rmsd.get('score'),
            'lddt_pli':lddt.get('score') if consistent else None}

def finalize_evaluation(rows,proper,errors,prediction_complete):
    summary=reference_mapped_summary(rows,proper)
    complete=prediction_complete and not errors and all(r.get('rank') is not None and is_success(r.get('rmsd'),r.get('lddt_pli')) is not None for r in rows)
    summary['diagnostic_successes']=summary['successes'];summary['selection_complete']=complete
    if not complete:
        summary['successes']=0
        summary['note']+=' Candidate coverage/assignment incomplete: apparent successes are diagnostic only and cannot raise the main success numerator.'
    return summary

def evaluate_case(root,system_id,report,reference,proper):
    output=root/'evaluation'/system_id;output.mkdir(parents=True,exist_ok=True)
    frozen=output/'prediction-report.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=report:raise ValueError('Frozen prediction report changed.')
    else:atomic_json(frozen,report)
    rows=[];errors=[]
    for number,candidate in enumerate(report['candidates']):
        result_path=evaluation_attempt(output,number)
        try:
            if digest(Path(candidate['cif']).read_bytes())!=candidate['sha256']:raise ValueError('Candidate changed after freezing.')
            ligands=sorted((reference/'ligand_files').glob('*.sdf'))
            expected={str(p.resolve()):digest(p.read_bytes()) for p in [Path(candidate['cif']),reference/'receptor.cif',*ligands]}
            if result_path.exists():
                execution=json.loads(result_path.with_suffix('.execution.json').read_text())
                if execution.get('returncode')!=0 or execution.get('input_sha256')!=expected:raise ValueError('Existing evaluation provenance mismatch.')
            else:
                executable = os.environ.get('TETRA_OST_BIN') or shutil.which('ost')
                if not executable:
                    raise ValueError('Install OpenStructure or set TETRA_OST_BIN.')
                score_model(executable, candidate['cif'], reference/'receptor.cif', ligands, result_path)
            raw=json.loads(result_path.read_text());rmsd={Path(x['reference_ligand']).stem:x for x in raw.get('rmsd',{}).get('assigned_scores',[])}
            lddt={Path(x['reference_ligand']).stem:x for x in raw.get('lddt_pli',{}).get('assigned_scores',[])}
            for chain in proper:
                rows.append(metric_row(candidate,chain,rmsd.get(chain,{}),lddt.get(chain,{})))
        except Exception as exc:errors.append({'candidate':number,'error':str(exc)})
    summary=finalize_evaluation(rows,proper,errors,report['complete']);summary.update(expected_candidates=report['expected_candidates'],available_candidates=report['available_candidates'],prediction_complete=report['complete'],evaluation_errors=errors)
    atomic_json(output/'summary.json',summary);return summary

def run(root):
    root=Path(root).resolve();config=json.loads((root/'config.json').read_text());cache=Path(config['cache']);cache.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another campaign controller is active.')
        records=json.loads((root/'cases.json').read_text())
        def update(phase,**fields):
            atomic_json(root/'cases.json',records)
            atomic_json(root/'status.json',{'phase':phase,'updated_utc':now(),'controller_pid':os.getpid(),'coverage':coverage(records,config['count']),**fields})
        try:
            for name in ['prediction_inputs','evaluation_metadata','zenodo_record']:
                if digest((root/(name+'.json')).read_bytes())!=config['hashes'][name]:raise ValueError('Frozen campaign source changed.')
            inputs=json.loads((root/'prediction_inputs.json').read_text());annotations={r['system_id']:r for r in json.loads((root/'evaluation_metadata.json').read_text())}
            source=json.loads((root/'zenodo_record.json').read_text());files={f['key']:f for f in source['files']}
            for name,msa_only in [('ground_truth.tar.gz',False),('msa_files.tar.gz',True)]:
                archive=cache/name;marker=cache/(name+'.extracted.json')
                if marker.exists():
                    data=json.loads(marker.read_text())
                    if data['systems_sha256']!=config['hashes']['prediction_inputs'] or data['checksum']!=files[name]['checksum']:raise ValueError('Extraction marker differs.')
                    continue
                if not archive.exists():
                    update('waiting_for_data',waiting_for=name)
                    # Shares the downloader file lock; if the standalone downloader dies, resume its durable chunks.
                    download_file(files[name]['links']['self'],archive,files[name]['size'],files[name]['checksum'])
                update('verifying_data',archive=name);checksum_file(archive,files[name]['checksum'])
                update('extracting_data',archive=name)
                extract_selected(archive,cache/'extracted',records,msa_only=msa_only)
                atomic_json(marker,{'systems_sha256':config['hashes']['prediction_inputs'],'checksum':files[name]['checksum'],'completed_utc':now()})
            # Build directory indexes, without consulting reference geometry during prediction.
            extracted=cache/'extracted';msa_dirs={};reference_dirs={}
            selected_lower={x.lower() for x in records}
            for path in extracted.rglob('*.csv'):
                if path.parent.name.lower() in selected_lower:msa_dirs[path.parent.name.lower()]=path.parent
            for path in extracted.rglob('receptor.cif'):reference_dirs[path.parent.name]=path.parent
            store=Store(root/'state')
            for item in inputs:
                sid=item['system_id'];case=records[sid]
                if case['status'] in TERMINAL:continue
                while shutil.disk_usage(root).free<10*1024**3:
                    update('paused_disk_headroom',system_id=sid);time.sleep(60)
                update('preparing_case',system_id=sid)
                try:
                    msa_dir=msa_dirs.get(sid.lower())
                    if msa_dir is None:raise ValueError('Official CSV MSA directory missing.')
                    mapping=msa_mapping(item['molecules'],msa_dir)
                    data=batch.import_case(store,root/'prediction_inputs.json',sid,mapping)
                    group=batch.create_group(store,data['input_id'],config['seeds'],config['inference'])
                except Exception as exc:
                    case.update(status='unsupported',error=str(exc));update('preparing_case',system_id=sid);continue
                case.update(status='running',group_id=group['group_id'],input_id=data['input_id']);update('predicting',system_id=sid)
                for job_id in group['job_ids']:
                    job=backend.status(store,job_id)
                    if job['status']=='queued' and not job.get('worker_pid'):
                        while True:
                            try:backend.ensure_gpu_available();break
                            except RuntimeError:update('waiting_for_gpu',system_id=sid);time.sleep(60)
                        backend.launch(store,job_id)
                    while backend.status(store,job_id)['status'] not in backend.TERMINAL:
                        update('predicting',system_id=sid,job_id=job_id);time.sleep(20)
                report=batch.summarize(store,group['group_id']);case['available_candidates']=report['available_candidates'];case['expected_candidates']=report['expected_candidates']
                case['job_status_counts']=report['job_status_counts'];atomic_json(root/'state/groups'/(group['group_id']+'.report.json'),report)
                if not report['candidates']:
                    case.update(status='failed',error='No valid candidates');update('predicting',system_id=sid);continue
                case['status']='evaluating';update('evaluating',system_id=sid)
                try:
                    proper=[l['chain'] for l in annotations[sid]['ligands'] if l['proper']]
                    if sid not in reference_dirs:raise ValueError('Official reference system missing.')
                    summary=evaluate_case(root,sid,report,reference_dirs[sid],proper)
                    case.update(status='completed' if summary['unavailable']==0 and summary['selection_complete'] else 'evaluation_failed',successes=summary['successes'],proper_ligands=summary['denominator'],unavailable=summary['unavailable'])
                except Exception as exc:case.update(status='evaluation_failed',error=str(exc))
                update('case_finished',system_id=sid)
            successes=sum(r.get('successes',0) for r in records.values());denominator=sum(l['proper'] for r in annotations.values() for l in r['ligands'])
            update('finished',proper_ligand_denominator=denominator,successes=successes,full_denominator_success_rate=successes/denominator)
        except BaseException as exc:
            update('controller_error',error=repr(exc));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('init');a.add_argument('--root',required=True);a.add_argument('--manifest',required=True);a.add_argument('--evaluation',required=True);a.add_argument('--cache',required=True);a.add_argument('--record',required=True)
    for name in ['run','status']:
        a=sub.add_parser(name);a.add_argument('--root',required=True)
    a=p.parse_args()
    if a.action=='init':print(json.dumps(initialize(a.root,a.manifest,a.evaluation,a.cache,a.record),indent=2))
    elif a.action=='run':run(a.root)
    else:
        root=Path(a.root);print((root/'status.json').read_text())
        config=json.loads((root/'config.json').read_text())
        for file in Path(config['cache']).glob('*.progress.json'):
            x=json.loads(file.read_text());print(file.name,round(100*x['completed_bytes']/x['size'],1),'%',x['completed_bytes'],'/',x['size'])
if __name__=='__main__':main()
