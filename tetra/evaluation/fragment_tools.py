"""Frozen question-driven fragment experiments; no experimental references accepted."""
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from rdkit import Chem
from tetra.evaluation.assessment import PROJECT, CHECKS, verify_candidate, identity_check, import_candidate
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest, stable
from tetra.evaluation.fragment_experiment import CONFIG, variability, edit_torsion, manipulation_check


def operator_hashes():
    paths = {
        'tetra/fragment_experiment.py': Path(__file__).with_name('fragment_experiment.py'),
        'scripts/relax_fragment.py': PROJECT / 'scripts/relax_fragment.py',
        'tetra/protein_preparation.py': Path(__file__).with_name('protein_preparation.py'),
    }
    return {name: digest(path.read_bytes()) for name, path in paths.items()}


def physical_parent_valid(assessment):
    return bool(assessment and assessment.get('status') == 'assessed' and
                all(assessment.get('checks', {}).get(key) == 'pass' for key in ('identity', *CHECKS)))


def measure(state, candidate_ids, parent_id):
    if parent_id not in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError('Unique candidate IDs including parent required.')
    result = variability([state['candidates'][k] for k in candidate_ids], parent_id)
    state.setdefault('fragment_reports', {})[parent_id] = result
    return result


def plan(state, candidate_id, torsion_id, question, hypothesis, alternative, expected_observation, evidence_ids):
    from tetra.research.decision_policy import references
    from tetra.orchestration.tool_registry import current_assessment
    if not state.get('allow_fragment_exploration') or state.get('fragment_experiments'):
        raise ValueError('One opt-in fragment experiment per session.')
    observations = references(state, evidence_ids)
    report = state.get('fragment_reports', {}).get(candidate_id)
    if not report or not any(o['action'] == 'measure_fragment_variability' and o['result'] == report for o in observations):
        raise ValueError('Read and cite current fragment variability evidence.')
    for key, sha in report['candidate_hashes'].items():
        verify_candidate(state['candidates'][key])
        if state['candidates'][key]['sha256'] != sha:
            raise ValueError('Stale fragment measurement.')
    assessment = current_assessment(state, candidate_id)
    if not physical_parent_valid(assessment):
        raise ValueError('Fragment exploration requires a physically valid parent; repair defects separately.')
    if state['candidates'][candidate_id].get('parent_id'):
        raise ValueError('Version 1 requires an original parent; ancestor contracts cannot be bypassed.')
    row = next((x for x in report['torsions'] if x['torsion_id'] == torsion_id and x['eligible']), None)
    if row is None:
        raise ValueError('No supported torsion with measured dispersion and stable protected component.')
    if not all(isinstance(x, str) and x.strip() for x in (question, hypothesis, alternative, expected_observation)):
        raise ValueError('Question, competing explanation and expected outcome required.')
    if hypothesis.strip() == alternative.strip():
        raise ValueError('Alternative must be distinct.')
    record = {'candidate_id': candidate_id, 'candidate_sha256': state['candidates'][candidate_id]['sha256'],
              'question': question, 'hypothesis': hypothesis, 'alternative': alternative,
              'expected_observation': expected_observation, 'evidence_ids': list(evidence_ids),
              'row': deepcopy(row), 'config': deepcopy(CONFIG),
              'source_hashes': deepcopy(report['candidate_hashes']),
              'operator_hashes': operator_hashes(),
              'claim_scope': 'local computational coverage; not physical flexibility or accuracy'}
    record['sha256'] = digest(stable(record))
    state['fragment_experiments'] = {'F1': {'status': 'planned', 'plan': record}}
    return {'experiment_id': 'F1', **deepcopy(record)}


def run_probe(parent, row, folder):
    """Four fixed starts; each failure and intermediate geometry remains on disk."""
    verify_candidate(parent)
    if not identity_check(parent):
        raise ValueError('Parent identity failed.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/'contract.json', {'parent_sha256': parent['sha256'], 'row': row, 'config': CONFIG})
    mol = Chem.RemoveHs(Chem.SDMolSupplier(parent['ligand'], removeHs=False)[0])
    children, attempts = [], []
    for offset in CONFIG['offsets_deg']:
        dest = folder/f'offset-{offset}'
        dest.mkdir()
        attempt = {'offset_deg': offset, 'status': 'running', 'operator_hashes': operator_hashes()}
        atomic_json(dest/'attempt.json', attempt)
        try:
            edited = edit_torsion(mol, {k:row[k] for k in ('torsion_id','bond','atoms','core','fragment')}, offset)
            path = dest/'edited.sdf'
            with Chem.SDWriter(str(path)) as writer:
                writer.write(edited)
            source = {**parent, 'ligand': str(path)}
            atomic_json(dest/'request.json', {'candidate': source, 'config': CONFIG, 'core_atoms': row['core']})
            executable = os.environ.get('TETRA_REFINE_PYTHON', str(PROJECT/'.refine-env/bin/python'))
            env = {**os.environ, 'PATH': str(Path(executable).absolute().parent)+os.pathsep+os.environ.get('PATH',''),
                   'CUDA_VISIBLE_DEVICES': '', 'OPENMM_CPU_THREADS':'2', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}
            with (dest/'worker.log').open('w') as log:
                proc = subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5',str(CONFIG['timeout_seconds']),
                    executable,str(PROJECT/'scripts/relax_fragment.py'),str(dest/'request.json'),str(dest)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=CONFIG['timeout_seconds']+15)
            if proc.returncode:
                raise ValueError(f'Relaxation/parameterization failed ({proc.returncode}); inspect worker.log. No chemistry substitution.')
            relaxed = Chem.RemoveHs(Chem.SDMolSupplier(str(dest/'refined.sdf'), removeHs=False)[0])
            check = manipulation_check(mol, edited, relaxed, row)
            receptor_fixed = Path(parent['receptor']).read_bytes() == (dest/'refined.pdb').read_bytes()
            if not receptor_fixed:
                raise ValueError('Receptor changed.')
            child = import_candidate({'receptor': str(dest/'refined.pdb'), 'ligand':str(dest/'refined.sdf'),
                'sequence':parent['sequence'],'smiles':parent['smiles'],'parent_id':parent['candidate_id']},dest/'candidates')
            chemical_ok = identity_check(child)
            check.update(chemical_identity_pass=chemical_ok,receptor_exactly_preserved=receptor_fixed)
            if not chemical_ok:
                check['status'] = 'FAILED'
            metrics = json.loads((dest/'metrics.json').read_text())
            if not isinstance(metrics, dict):
                raise ValueError('Invalid worker metrics.')
            attempt.update(status='completed', candidate=child, manipulation_check=check,
                           worker_metrics=metrics)
            if check['status'] == 'PASS':
                children.append(child)
        except Exception as error:
            attempt.update(status='failed', error=str(error))
        attempts.append(attempt)
        atomic_json(dest/'attempt.json',attempt)
    verify_candidate(parent)
    result = {'parent_id':parent['candidate_id'],'parent_sha256':parent['sha256'],
              'attempts':attempts,'candidates':children,'status':'completed',
              'physical_assessment':'REQUIRED_BEFORE_SELECTION',
              'hypothesis_discrimination':'REQUIRES_REVIEW; new states alone do not distinguish flexibility from model error',
              'experimental_accuracy':'NOT_EVALUATED'}
    atomic_json(folder/'result.json',result)
    return result


def execute(agent, state, experiment_id):
    record = state.get('fragment_experiments', {}).get(experiment_id)
    if record and record['status'] == 'completed':
        return {'experiment_id': experiment_id, **deepcopy(record['result'])}
    if record and record['status'] == 'running':
        record.update(status='failed', error='Interrupted execution; inspect preserved attempt artifacts.')
        if agent is not None:
            agent.save(state)
    if not record or record['status'] != 'planned':
        raise ValueError('Unused frozen plan required; interrupted experiments are not automatically repeated.')
    frozen = record['plan']
    if frozen['sha256'] != digest(stable({k:v for k,v in frozen.items() if k!='sha256'})) or frozen['config'] != CONFIG:
        raise ValueError('Frozen fragment plan changed.')
    for key, sha in frozen['source_hashes'].items():
        verify_candidate(state['candidates'][key])
        if state['candidates'][key]['sha256'] != sha:
            raise ValueError('Source candidate changed.')
    if frozen.get('operator_hashes') != operator_hashes():
        raise ValueError('Fragment implementation changed after planning.')
    parent = state['candidates'][frozen['candidate_id']]
    record['status'] = 'running'
    agent.save(state)
    try:
        result = run_probe(parent,frozen['row'],agent.folder(state)/'fragment_experiments'/experiment_id)
        for child in result['candidates']:
            state['candidates'][child['candidate_id']] = child
        state['selected'] = None
        record.update(status='completed',result=result)
        return {'experiment_id':experiment_id,**result}
    except Exception as error:
        record.update(status='failed',error=str(error))
        raise
    finally:
        agent.save(state)
