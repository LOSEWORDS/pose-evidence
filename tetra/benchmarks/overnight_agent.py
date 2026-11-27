"""Paired current-agent evaluation on frozen baseline candidates; no new inference."""
import argparse,datetime,fcntl,json,os,pickle,signal,subprocess,sys,time
from pathlib import Path
from collections import Counter
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest
from tetra.evaluation.structure_export import selected_complex
PROJECT=Path(__file__).resolve().parents[2]
DEADLINE=datetime.datetime(2026,11,17,3,50,tzinfo=datetime.timezone.utc).timestamp()

def agent_budget(now,deadline):
    seconds=min(1200,int(deadline-now)-90)
    return seconds if seconds>=120 else 0

def ordered_candidates(report):
    if not report.get('complete') or len(report['per_ligand'])!=1:raise ValueError('Need a complete single-ligand prediction report')
    candidates={x['sha256']:x for x in report['candidates']}
    ranking=next(iter(report['per_ligand'].values()))['ranking']
    if len(ranking)!=len(candidates):raise ValueError('Candidate ranking mismatch')
    return [candidates[x['sha256']] for x in ranking]

def build_request(row,exports):
    m=row['molecules']
    if len(m['sequences'])!=1 or len(m['smiles'])!=1:raise ValueError('Single pair required')
    return {'goal':'Assess these existing predicted complex candidates. Preserve their given family order. Use justified refinement for observed geometry or clash problems if supported, reassess any changed structure, select candidates and finish. Do not run a new prediction. Report unsupported inputs and unavailable checks honestly.',
        'candidates':[dict(x,sequence=next(iter(m['sequences'].values())),smiles=m['smiles'][0]) for x in exports]}

def terminate_tree(process):
    import psutil
    try:children=psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:children=[]
    for child in children[::-1]:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    if process.poll() is None:process.terminate()
    _,alive=psutil.wait_procs(children,timeout=5)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()

def run_bounded(command,log,seconds,env=None):
    with Path(log).open('ab') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        try:return p.wait(timeout=seconds)
        except BaseException:
            terminate_tree(p);raise

def confidence_metadata(candidate):
    """Recover raw predictor confidence omitted from the aggregated batch report."""
    import math
    cif = Path(candidate['cif'])
    path = cif.with_name('confidence_' + cif.stem + '.json')
    confidence = dict(candidate.get('confidence', {}))
    provenance = {'job_id': candidate['job_id'], 'cif_sha256': candidate['sha256']}
    if path.exists():
        raw = path.read_bytes()
        confidence = json.loads(raw)
        if not isinstance(confidence, dict):
            raise ValueError('Invalid raw confidence artifact.')
        for key in ('confidence_score', 'ligand_iptm', 'complex_plddt'):
            value = confidence.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid original predictor confidence: ' + key)
        provenance.update(confidence_path=str(path), confidence_sha256=digest(raw))
    else:
        provenance['raw_confidence_status'] = 'unavailable' if not confidence else 'embedded'
    ranks = candidate.get('ligand_ranks', {})
    if len(ranks) == 1:
        value = next(iter(ranks.values()))
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
            confidence['baseline_ligand_rank_score'] = value
    return confidence, provenance


def export_candidates(baseline,row,report,dest):
    from rdkit import Chem
    from tetra.evaluation.structure_export import export_single_ligand
    candidates=ordered_candidates(report);exports=[]
    target=Chem.MolToSmiles(Chem.MolFromSmiles(row['molecules']['smiles'][0]))
    for index,c in enumerate(candidates):
        cif=Path(c['cif'])
        if digest(cif.read_bytes())!=c['sha256']:raise ValueError('Baseline structure changed')
        jobdir=baseline/'state/jobs'/c['job_id']
        # Only our own local predictor's molecular graph is deserialized.
        paths=list(jobdir.glob('**/processed/mols/input.pkl'))
        if len(paths)!=1:raise ValueError('Missing unique local Boltz graph')
        with paths[0].open('rb') as stream:molecules=pickle.load(stream)
        matches=[m for m in molecules.values() if Chem.MolToSmiles(Chem.RemoveHs(m))==target]
        if len(matches)!=1:raise ValueError('Ambiguous local ligand graph')
        out=dest/str(index);export_single_ligand(cif,matches[0],out)
        confidence, provenance = confidence_metadata(c)
        exports.append({'receptor':str(out/'receptor.pdb'),'ligand':str(out/'ligand.sdf'),
            'original_confidence':confidence, 'confidence_provenance':provenance})
    return exports

def process_case(baseline,root,sid):
    folder=root/'cases'/sid;folder.mkdir(parents=True,exist_ok=True);status=folder/'status.json'
    result={'system_id':sid,'status':'running','started_epoch':time.time()};atomic_json(status,result)
    try:
        row=next(x for x in json.loads((baseline/'prediction_inputs.json').read_text()) if x['system_id']==sid)
        report=json.loads((baseline/'cases'/(sid+'.prediction-report')).read_text())
        exports=export_candidates(baseline,row,report,folder/'exports')
        request=build_request(row,exports);atomic_json(folder/'request.json',request)
        # Lock the prediction-only baseline winner before any target scoring.
        atomic_json(folder/'provenance.json',{'baseline_top1_sha256':ordered_candidates(report)[0]['sha256'],
            'baseline_report_sha256':digest((baseline/'cases'/(sid+'.prediction-report')).read_bytes()),'paired_candidates':len(exports)})
        budget=agent_budget(time.time(),DEADLINE)
        if not budget:raise TimeoutError('No agent budget remaining')
        command=[sys.executable,'-m','tetra.workflow','--state',str(folder/'state'),'run',str(folder/'request.json'),'--refinement','--max-calls','16']
        code=run_bounded(command,folder/'agent.log',budget,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        if code:raise RuntimeError(f'Agent exited {code}; see agent.log')
        reports=list((folder/'state/dynamic').glob('*/report.json'))
        if len(reports)!=1:raise ValueError('Expected one final agent report')
        agent=json.loads(reports[0].read_text());result.update(agent_status=agent['status'],task_complete=agent['task_complete'],report=str(reports[0]))
        if not agent.get('selection',{}):raise ValueError('Agent produced no selection')
        selected=agent['selection']['selected_ids'][0]
        candidate=agent['candidates'][selected]
        # Commit selected identity before opening target reference metadata.
        atomic_json(folder/'selected.json',{'candidate_id':selected,'sha256':candidate['sha256'],'task_complete':agent['task_complete']})
        model=folder/'selected.cif';selected_complex(candidate,model)
        result['selected_sha256']=digest(model.read_bytes());result['status']='evaluating';atomic_json(status,result)
        from tetra.benchmarks.benchmark import score_model,is_success
        from tetra.benchmarks.campaign import metric_row
        metadata=next(x for x in json.loads((baseline/'evaluation_metadata.json').read_text()) if x['system_id']==sid)
        chain=next(x['chain'] for x in metadata['ligands'] if x['proper'])
        refs=[p for p in (baseline/'extracted').rglob('receptor.cif') if p.parent.name.lower()==sid.lower()]
        if len(refs)!=1:raise ValueError('Unique reference missing')
        out=folder/'offline-score.json';seconds=min(45,int(DEADLINE-time.time()))
        if seconds<1:raise TimeoutError('Scoring deadline reached')
        score_model(PROJECT/'scripts/ost_scoring_compat.py',candidate['receptor'],refs[0],sorted((refs[0].parent/'ligand_files').glob('*.sdf')),out,timeout=seconds,model_ligands=[candidate['ligand']])
        raw=json.loads(out.read_text());rr={Path(x['reference_ligand']).stem:x for x in raw.get('rmsd',{}).get('assigned_scores',[])};ll={Path(x['reference_ligand']).stem:x for x in raw.get('lddt_pli',{}).get('assigned_scores',[])}
        metric=offline_metric(agent['session_id'],chain,rr.get(chain,{}),ll.get(chain,{}))
        success=is_success(metric['rmsd'],metric['lddt_pli'])
        if success is None:raise ValueError('Offline score unavailable or inconsistent')
        result.update(status='completed',pose_success=success,agent_success=bool(agent['task_complete'] and success),metrics=metric,
            refinement_attempts=sum(x.get('action')=='refine_structure' for x in agent.get('observations',[])))
    except Exception as exc:result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    result['finished_epoch']=time.time();atomic_json(status,result)

def summarize(baseline,root,phase):
    inputs=json.loads((baseline/'prediction_inputs.json').read_text());rows=[]
    for row in inputs:
        p=root/'cases'/row['system_id']/'status.json'
        rows.append(json.loads(p.read_text()) if p.exists() else {'system_id':row['system_id'],'status':'pending'})
    summary={'phase':phase,'updated_epoch':time.time(),'denominator':len(inputs),'status_counts':dict(Counter(x['status'] for x in rows)),
        'pose_successes':sum(x.get('pose_success') is True for x in rows),'agent_complete_and_pose_successes':sum(x.get('agent_success') is True for x in rows),
        'deadline_china':'2026-11-17 11:50 +08:00','note':'Paired post-prediction current-agent evaluation; fixed baseline family ordering; failures retained; no extra GPU prediction.'}
    summary['strata']={}
    for key in ['low','middle','high','development','evaluation']:
        ids={r['system_id'] for r in inputs if key in (r['selection']['band'],r['selection']['split'])}
        summary['strata'][key]={'denominator':len(ids),'pose_successes':sum(x['system_id'] in ids and x.get('pose_success') is True for x in rows)}
    atomic_json(root/'status.json',summary)

def offline_metric(session,chain,rmsd,lddt):
    from tetra.benchmarks.campaign import metric_row
    result=metric_row({'job_id':'agent:'+session,'seed':101,'sample':'agent_selected','ligand_ranks':{}},chain,rmsd,lddt)
    consistent=bool(rmsd.get('model_ligand')) and rmsd.get('model_ligand')==lddt.get('model_ligand')
    result.update(metric_assignment_consistent=consistent,model_ligand=rmsd.get('model_ligand'),lddt_model_ligand=lddt.get('model_ligand'),lddt_pli=lddt.get('score') if consistent else None)
    return result

def worker_alive(pid,root):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'tetra.overnight_agent' in command and str(root).encode() in command and b'worker' in command
    except OSError:return False

def reconcile_workers(root):
    for p in (root/'cases').glob('*/worker-process.json'):
        if worker_alive(json.loads(p.read_text())['pid'],root):
            raise RuntimeError('Existing agent worker still active; refusing duplicate controller or premature final accounting')
    for p in (root/'cases').glob('*/status.json'):
        row=json.loads(p.read_text())
        if row['status'] in ('running','evaluating'):
            row.update(status='interrupted',error='Previous worker ended without a final result; artifacts retained and no automatic rerun')
            atomic_json(p,row)

def run(baseline,root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reconcile_workers(root)
        active={};attempted={p.parent.name for p in (root/'cases').glob('*/status.json')}
        def stopped(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stopped)
        try:
            while time.time()<DEADLINE:
                rows=json.loads((baseline/'prediction_inputs.json').read_text())
                for sid,(proc,log) in list(active.items()):
                    if proc.poll() is not None:
                        log.close();del active[sid]
                        p=root/'cases'/sid/'status.json'
                        if not p.exists() or json.loads(p.read_text())['status'] in ('running','evaluating'):
                            atomic_json(p,{'system_id':sid,'status':'failed','error':'Worker terminated before committing result'})
                for row in rows:
                    sid=row['system_id']
                    if len(active)>=2:break
                    if sid in attempted or not agent_budget(time.time(),DEADLINE):continue
                    prediction=baseline/'cases'/(sid+'.prediction-report')
                    if not prediction.exists() or not json.loads(prediction.read_text()).get('complete'):continue
                    folder=root/'cases'/sid;folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('ab')
                    proc=subprocess.Popen([sys.executable,'-m','tetra.overnight_agent','worker','--baseline',str(baseline),'--root',str(root),'--system-id',sid],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    atomic_json(folder/'worker-process.json',{'pid':proc.pid,'started_epoch':time.time()})
                    active[sid]=(proc,log);attempted.add(sid)
                summarize(baseline,root,'running_agent' if active else 'waiting_for_baseline_candidates')
                baseline_status=json.loads((baseline/'status.json').read_text()).get('phase','')
                if not active and (len(attempted)==len(rows) or baseline_status.startswith('finished') or not agent_budget(time.time(),DEADLINE)):break
                time.sleep(10)
        except BaseException as exc:
            atomic_json(root/'controller-error.json',{'type':type(exc).__name__,'epoch':time.time()})
        finally:
            for proc,log in active.values():terminate_tree(proc);log.close()
        for row in json.loads((baseline/'prediction_inputs.json').read_text()):
            p=root/'cases'/row['system_id']/'status.json'
            if not p.exists() or json.loads(p.read_text())['status'] in ('running','evaluating'):
                atomic_json(p,{'system_id':row['system_id'],'status':'not_completed','error':'No completed agent result before campaign termination'})
        summarize(baseline,root,'finished')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','worker']);parser.add_argument('--baseline',required=True,type=Path);parser.add_argument('--root',required=True,type=Path);parser.add_argument('--system-id');a=parser.parse_args()
    if a.action=='worker':process_case(a.baseline.resolve(),a.root.resolve(),a.system_id)
    else:run(a.baseline.resolve(),a.root.resolve())
if __name__=='__main__':main()
