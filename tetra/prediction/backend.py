"""Real Boltz execution, bounded jobs and evidence-based artifact status."""
import collections
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import gemmi
import numpy as np
import psutil
from rdkit import Chem
import yaml
from tetra.runtime.inputs import digest
from tetra.runtime.store import now

PROJECT=Path(__file__).resolve().parents[2]
TERMINAL={'succeeded','failed','interrupted'}
SIDECHAINS={
'ALA':'CB','ARG':'CB CG CD NE CZ NH1 NH2','ASN':'CB CG OD1 ND2','ASP':'CB CG OD1 OD2',
'CYS':'CB SG','GLN':'CB CG CD OE1 NE2','GLU':'CB CG CD OE1 OE2','GLY':'',
'HIS':'CB CG ND1 CD2 CE1 NE2','ILE':'CB CG1 CG2 CD1','LEU':'CB CG CD1 CD2','LYS':'CB CG CD CE NZ',
'MET':'CB CG SD CE','PHE':'CB CG CD1 CD2 CE1 CE2 CZ','PRO':'CB CG CD',
'SER':'CB OG','THR':'CB OG1 CG2','TRP':'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',
'TYR':'CB CG CD1 CD2 CE1 CE2 CZ OH','VAL':'CB CG1 CG2'}

def complete_protein(chain):
    for residue in chain:
        if residue.name not in SIDECHAINS:return False
        names=[a.name for a in residue if not a.element.is_hydrogen]
        required=set(('N CA C O '+SIDECHAINS[residue.name]).split())
        if len(names)!=len(set(names)) or not required<=set(names) or set(names)-required-{'OXT'}:return False
    return True

def components(record):
    if record.get('schema_version')==2:return record['proteins'],record['ligands']
    return ([{'chain_id':'A','sequence':record['sequence'],'msa':record.get('msa')}],
            [{'chain_id':'B','smiles':record['smiles']}])

def input_payload(record,pocket_constraints=None,contact_constraints=None):
    proteins,ligands=components(record)
    sequences=[];msa_by_sequence={}
    for p in proteins:
        item={'id':p['chain_id'],'sequence':p['sequence']}
        if p.get('msa'):
            fingerprint=digest(Path(p['msa']).read_bytes())
            previous=msa_by_sequence.get(p['sequence'])
            if previous and previous[0]!=fingerprint:raise ValueError('Identical protein copies require identical MSA bytes.')
            msa_by_sequence.setdefault(p['sequence'],(fingerprint,p['msa']))
            item['msa']=msa_by_sequence[p['sequence']][1]
        sequences.append({'protein':item})
    sequences.extend({'ligand':{'id':l['chain_id'],'smiles':l['smiles']}} for l in ligands)
    payload={'version':1,'sequences':sequences}
    constraints = []
    if pocket_constraints is not None:
        from tetra.prediction.prediction_conditions import validate_pockets
        lengths = {p['chain_id']: len(p['sequence']) for p in proteins}
        ligand_ids = [l['chain_id'] for l in ligands]
        chain_ids = [*lengths, *ligand_ids]
        for pocket in validate_pockets(pocket_constraints):
            if pocket['binder'] not in chain_ids:
                raise ValueError(
                    f"Unknown pocket binder {pocket['binder']!r}. Use an input chain ID, "
                    f"not a residue name. Available chains: {', '.join(chain_ids)}. "
                    f"For the ligand use its ligand chain ID: {', '.join(ligand_ids)}."
                )
            for chain, position in pocket['contacts']:
                if chain not in lengths:
                    raise ValueError(f'Unknown pocket contact protein chain {chain!r}. '
                                     f'Use one of: {", ".join(lengths)}.')
                if position > lengths[chain]:
                    raise ValueError(f'Pocket residue {chain}:{position} is outside the input sequence.')
            constraints.append({'pocket': pocket})
    if contact_constraints is not None:
        from tetra.prediction.prediction_conditions import validate_contacts
        constraints.extend({'contact':c} for c in validate_contacts(contact_constraints))
    if constraints:
        payload['constraints'] = constraints
    return payload

def inspect(folder,record):
    folder=Path(folder)
    cifs=sorted(folder.glob('**/*_model_*.cif'))
    if not cifs:raise ValueError('Missing structure artifacts.')
    proteins,ligands=components(record)
    expected_ids=[p['chain_id'] for p in proteins]+[l['chain_id'] for l in ligands]
    reports=[]
    for cif in cifs:
        st=gemmi.read_structure(str(cif))
        if len(st)!=1:raise ValueError('Expected one structural model per CIF.')
        chains={c.name:c for c in st[0]}
        if len(chains)!=len(st[0]) or set(chains)!=set(expected_ids):raise ValueError('Unexpected, duplicate or missing chain identities.')
        chain_reports=[];pa=[];la=[];residues=0
        for protein in proteins:
            chain=chains[protein['chain_id']]
            seq=''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in chain)
            atoms=[a for r in chain for a in r if not a.element.is_hydrogen]
            if not atoms:raise ValueError('Empty protein coordinates.')
            chain_reports.append({'chain_id':protein['chain_id'],'kind':'protein',
                                  'sequence_matches':seq==protein['sequence'],'heavy_atoms_complete':complete_protein(chain)})
            pa.extend(atoms);residues+=len(seq)
        for ligand in ligands:
            mol=Chem.MolFromSmiles(ligand['smiles'])
            expected=collections.Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()!=1)
            chain=chains[ligand['chain_id']];atoms=[a for r in chain for a in r if not a.element.is_hydrogen]
            if not atoms:raise ValueError('Empty ligand coordinates.')
            chain_reports.append({'chain_id':ligand['chain_id'],'kind':'ligand','heavy_atom_elements_match':collections.Counter(a.element.name for a in atoms)==expected})
            la.extend(atoms)
        px=np.array([[a.pos.x,a.pos.y,a.pos.z] for a in pa]);lx=np.array([[a.pos.x,a.pos.y,a.pos.z] for a in la])
        conf_path=cif.with_name('confidence_'+cif.stem+'.json')
        if not conf_path.exists():raise ValueError('Missing confidence artifact.')
        conf=json.loads(conf_path.read_text())
        def finite(v):
            if isinstance(v,dict):return all(finite(x) for x in v.values())
            if isinstance(v,list):return all(finite(x) for x in v)
            return math.isfinite(v) if isinstance(v,(int,float)) else True
        required=['confidence_score','ligand_iptm','complex_plddt']
        checks={'protein_sequence_matches':all(c['sequence_matches'] for c in chain_reports if c['kind']=='protein'),
                'protein_heavy_atoms_complete':all(c['heavy_atoms_complete'] for c in chain_reports if c['kind']=='protein'),
                'ligand_heavy_atom_elements_match':all(c['heavy_atom_elements_match'] for c in chain_reports if c['kind']=='ligand'),
                'finite_coordinates':bool(np.isfinite(px).all() and np.isfinite(lx).all()),
                'valid_confidence':all(type(conf.get(k)) in (int,float) and 0<=conf[k]<=1 for k in required) and finite(conf)}
        if not all(checks.values()):raise ValueError('Artifact identity/completeness validation failed: '+json.dumps(checks))
        # Bound memory for large complexes rather than creating an Nprotein x Nligand x 3 array.
        min_distance=float('inf');contacts=0
        for start in range(0,len(px),512):
            distance=np.sqrt(((px[start:start+512,None,:]-lx[None,:,:])**2).sum(-1))
            min_distance=min(min_distance,float(distance.min()));contacts+=int((distance<=4).sum())
        reports.append({'cif':str(cif.resolve()),'sha256':digest(cif.read_bytes()),'confidence':conf,'checks':checks,'chains':chain_reports,
                        'protein_residues':residues,'ligand_heavy_atoms':len(la),
                        'minimum_contact_distance_A':min_distance,'atom_pairs_within_4A':contacts})
    reports.sort(key=lambda x:x['confidence']['confidence_score'],reverse=True)
    return {'models':reports,'checks':reports[0]['checks'],
            'limitations':['Model confidence is not measured accuracy or experimental binding evidence.',
                           'Heavy-atom element counts do not establish connectivity or stereochemistry.',
                           'No experimental RMSD or comprehensive physical validation performed.'],
            'warnings':['Inspect confidence and geometry before scientific use; no automatic low-confidence rerun.']}

def execute(command,folder,timeout,env,lock_fd=None):
    folder=Path(folder);start=time.monotonic();timed_out=False
    combined=os.environ.copy();combined.update(env)
    with (folder/'inference.log').open('w') as log:
        # The independent timer and inherited flock survive abrupt supervisor death.
        guarded=['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(timeout)+'s',*command]
        p=subprocess.Popen(guarded,stdout=log,stderr=subprocess.STDOUT,env=combined,cwd=folder,start_new_session=True,pass_fds=(() if lock_fd is None else lock_fd if isinstance(lock_fd, tuple) else (lock_fd,)))
        try:p.wait(timeout=timeout+12)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        except BaseException:
            try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    return {'exit_code':p.returncode,'timed_out':timed_out or p.returncode in (124,137),'elapsed_seconds':time.monotonic()-start}

def ensure_gpu_available(gpu=1):
    slots = gpu_slots()
    field = 'memory.free' if slots > 1 else 'memory.used'
    result = subprocess.run(
        ['nvidia-smi', f'--id={gpu}', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    memory_mib = int(result.stdout.strip())
    if slots > 1 and memory_mib < 16 * 1024:
        raise RuntimeError(f'GPU {gpu} has {memory_mib} MiB free; parallel inference requires 16 GiB free.')
    if slots == 1 and memory_mib > 512:
        raise RuntimeError(f'GPU {gpu} is occupied by another process; no job was interrupted.')


def gpu_slots():
    slots = int(os.environ.get('TETRA_GPU_SLOTS', '1'))
    if not 1 <= slots <= 4:
        raise ValueError('TETRA_GPU_SLOTS must be between 1 and 4.')
    return slots


@contextmanager
def inference_slot(lockdir, gpu):
    slots = gpu_slots()
    with (lockdir / f'gpu{gpu}.lock').open('a') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX if slots == 1 else fcntl.LOCK_SH)
        if slots == 1:
            yield gate.fileno(), None
            return
        while True:
            for index in range(slots):
                with (lockdir / f'gpu{gpu}-slot{index}.lock').open('a') as slot:
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    yield slot.fileno(), gate.fileno()
                    return
            time.sleep(1)


def command_for(folder,config,use_msa_server):
    args=[str(Path(sys.executable).with_name('boltz')),'predict',str(folder/'input.yaml'),'--out_dir',str(folder),
          '--cache',str(PROJECT/'cache/boltz'),'--model','boltz2','--accelerator','gpu','--devices','1',
          '--num_workers','2','--no_kernels','--output_format','mmcif']
    for key in ['seed','diffusion_samples','recycling_steps','sampling_steps']:args.extend(['--'+key,str(config[key])])
    if config.get('use_potentials', False):args.append('--use_potentials')
    if use_msa_server:args.append('--use_msa_server')
    return args

def gpu_lock_directory():
    return Path(os.environ.get('TETRA_GPU_LOCK_DIR',str(PROJECT/'state'))).resolve()

def run_job(store,job_id):
    folder=store.root/'jobs'/job_id
    # Cross-store lock: all operator state directories share the same physical GPU.
    lockdir=gpu_lock_directory();lockdir.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as job_lock:
        try:fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        job=store.job(job_id)
        if job['status'] in TERMINAL:return
        store.update_job(job_id,'queued',worker_pid=os.getpid(),worker_started=psutil.Process().create_time())
        gpu = job.get('gpu', 1)
        if type(gpu) is not int or gpu not in (0, 1):
            raise ValueError('Invalid persisted GPU assignment.')
        with inference_slot(lockdir, gpu) as (slot_fd, gate_fd):
            lock_fd = slot_fd if gate_fd is None else (slot_fd, gate_fd)
            try:
                data=store.input(job['input_id'])
                if not data['confirmed']:raise ValueError('Input identity is unconfirmed.')
                payload=input_payload(data,job['config'].get('pocket_constraints'),job['config'].get('contact_constraints'))
                if job['config'].get('pocket_constraints') or job['config'].get('contact_constraints'):
                    from tetra.prediction.prediction_conditions import validate_boltz_payload
                    validate_boltz_payload(payload, PROJECT / 'cache/boltz/mols')
                ensure_gpu_available(gpu)
                (folder/'input.yaml').write_text(yaml.safe_dump(payload,sort_keys=False))
                proteins,_=components(data)
                command=command_for(folder,job['config'],any(not p.get('msa') for p in proteins))
                versions={k:importlib.metadata.version(k) for k in ['boltz','torch','rdkit','gemmi']}
                manifest={'input':data,'command':command,'versions':versions,'gpu':gpu,'started_utc':now()}
                (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
                store.update_job(job_id,'running',started_utc=now(),command=command)
                result=execute(command,folder,job['config']['timeout_seconds'],{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1'},lock_fd=lock_fd)
                (folder/'execution.json').write_text(json.dumps(result,indent=2))
                if result['timed_out']:raise RuntimeError('Prediction execution timed out; see inference.log.')
                if result['exit_code']!=0:raise RuntimeError(f"Prediction exited {result['exit_code']}; see inference.log.")
                report=inspect(folder,data)
                if len(report['models'])!=job['config']['diffusion_samples']:raise ValueError('Incomplete sample artifacts.')
                (folder/'inspection.json').write_text(json.dumps(report,indent=2))
                store.update_job(job_id,'succeeded',finished_utc=now(),result=result,report=str(folder/'inspection.json'))
                store.event(job_id,{'action':'prediction_completed','models':len(report['models'])})
            except Exception as error:
                store.update_job(job_id,'failed',error=str(error),finished_utc=now())
                store.event(job_id,{'action':'prediction_failed','error':str(error)})

def launch(store,job_id):
    job=status(store,job_id)
    if job['status']!='queued' or job.get('worker_pid'):return job
    folder=store.root/'jobs'/job_id
    with (folder/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        job=store.job(job_id)
        if job.get('worker_pid') or job['status']!='queued':return job
        with (folder/'worker.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-m','tetra.worker',str(store.root),job_id],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:created=psutil.Process(p.pid).create_time()
        except psutil.NoSuchProcess:created=0
        # Worker owns status transitions; only set ownership fields here.
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT record FROM jobs WHERE id=?',(job_id,)).fetchone()
            record=json.loads(row[0]);record.update(worker_pid=p.pid,worker_started=created)
            db.execute('UPDATE jobs SET record=? WHERE id=?',(json.dumps(record),job_id))
    return store.job(job_id)

def status(store,job_id):
    job=store.job(job_id)
    if job['status'] in TERMINAL:return job
    pid=job.get('worker_pid')
    if pid:
        try:
            p=psutil.Process(pid)
            alive=abs(p.create_time()-job['worker_started'])<0.01 and p.status()!=psutil.STATUS_ZOMBIE
        except psutil.Error:alive=False
        if not alive:return store.update_job(job_id,'interrupted',error='Worker exited without a terminal record; no automatic rerun.',finished_utc=now())
    return job
