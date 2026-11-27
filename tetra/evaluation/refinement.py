"""Bounded CPU optimization attempts. Replays never duplicate an uncertain attempt."""
import fcntl,json,os,subprocess,hashlib
from pathlib import Path
from tetra.evaluation.assessment import PROJECT,verify_candidate,import_candidate,identity_check,fingerprint
from tetra.runtime.download import atomic_json

DEFAULTS={'timeout_seconds':600,'max_iterations':250,'restraint_k_kj_mol_nm2':1000.0,'protein_forcefield':'amber14-all.xml','ligand_forcefield':'openff-2.2.1','platform':'CPU','threads':2}

def run_refinement(candidate,output_dir,python=None):
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'attempt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=folder/'attempt.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['parent_sha256']!=candidate['sha256']:raise ValueError('Attempt belongs to a different parent.')
            if previous['status']=='running':
                previous.update(status='interrupted',error='Prior attempt did not commit a result; no automatic rerun.')
                atomic_json(path,previous)
            return previous
        result={'status':'running','parent_sha256':candidate['sha256'],'config':DEFAULTS}
        atomic_json(path,result)
        try:
            verify_candidate(candidate)
            atomic_json(folder/'request.json',{'candidate':candidate,'config':DEFAULTS})
            executable=python or os.environ.get('TETRA_REFINE_PYTHON',str(PROJECT/'.refine-env/bin/python'))
            cmd=[executable,str(PROJECT/'scripts/refine_complex.py'),str(folder/'request.json'),str(folder)]
            with (folder/'worker.log').open('wb') as log:
                process=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5',str(DEFAULTS['timeout_seconds']),*cmd],
                    stdout=log,stderr=subprocess.STDOUT,timeout=DEFAULTS['timeout_seconds']+15,
                    env={**os.environ,'PATH':str(Path(executable).absolute().parent)+os.pathsep+os.environ.get('PATH',''),'CUDA_VISIBLE_DEVICES':'','OPENMM_CPU_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            if process.returncode:raise ValueError('Refinement failed or timed out; exit '+str(process.returncode)+'; see worker.log.')
            child={'receptor':str(folder/'refined.pdb'),'ligand':str(folder/'refined.sdf'),'sequence':candidate['sequence'],
                   'smiles':candidate['smiles'],'parent_id':candidate['candidate_id']}
            if not identity_check(dict(child,sha256=fingerprint(child['receptor'],child['ligand']))):raise ValueError('Refinement changed or invalidated molecular identity.')
            child=import_candidate(child,folder/'candidates')
            verify_candidate(candidate)
            result.update(status='succeeded',candidate=child,metrics=json.loads((folder/'metrics.json').read_text()))
        except Exception as e:result.update(status='failed',error=str(e))
        atomic_json(path,result);return result


def refinement_available(marker=None):
    """Operator-written feasibility evidence gates autonomous use of this exact worker/config."""
    try:
        proof=json.loads(Path(marker or PROJECT/'environment/refinement-validation.json').read_text())
        return (proof.get('passed') is True and proof.get('config')==DEFAULTS and
                proof.get('preparation_sha256')==hashlib.sha256((PROJECT/'tetra/evaluation/protein_preparation.py').read_bytes()).hexdigest() and
                proof.get('mapping_sha256')==hashlib.sha256((PROJECT/'tetra/evaluation/atom_mapping.py').read_bytes()).hexdigest() and
                proof.get('script_sha256')==hashlib.sha256((PROJECT/'scripts/refine_complex.py').read_bytes()).hexdigest())
    except (OSError,ValueError,KeyError):return False
