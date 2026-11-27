"""One bounded fixed-receptor docking experiment with immutable candidate artifacts."""
from pathlib import Path
import hashlib
from rdkit import Chem
from tetra.evaluation.assessment import verify_candidate, identity_check, import_candidate
from tetra.runtime.download import atomic_json
from tetra.evaluation.external_science import _run, available

CONFIG = {'exhaustiveness': 8, 'num_modes': 5, 'autobox_add': 4,
          'min_rmsd_filter': 1, 'seed': 42, 'cnn_scoring': 'rescore',
          'cpu': 2, 'device': 'cpu', 'timeout_seconds': 600}


def dock_candidate(candidate, folder, hypothesis, expected_observation):
    """Normal docking, not minimization; no experimental reference is accepted."""
    verify_candidate(candidate)
    if not identity_check(candidate) or not available('gnina'):
        raise ValueError('Verified identity and validated GNINA required.')
    if not all(isinstance(x, str) and x.strip() for x in (hypothesis, expected_observation)):
        raise ValueError('Hypothesis and expected observation required.')
    molecule = Chem.SDMolSupplier(candidate['ligand'], removeHs=False)[0]
    if molecule is None or any(a.GetAtomicNum() not in (1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53) for a in molecule.GetAtoms()):
        raise ValueError('This bounded noncovalent docking profile does not support metal-containing or unsupported ligand chemistry.')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    receptor_hash = hashlib.sha256(Path(candidate['receptor']).read_bytes()).hexdigest()
    contract = {'parent_id': candidate['candidate_id'], 'parent_sha256': candidate['sha256'],
                'hypothesis': hypothesis, 'expected_observation': expected_observation,
                'config': CONFIG, 'receptor_sha256': receptor_hash,
                'search_region_source': 'predicted_parent_ligand',
                'preservation': ['exact_receptor_bytes', 'ligand_identity_and_stereochemistry'],
                'acceptance': 'Identity and physical checks required; uniform GNINA score-only ranking is computational preference only.',
                'experimental_accuracy': 'NOT_AVAILABLE_ONLINE'}
    atomic_json(folder / 'contract.json', contract)
    output = folder / 'docked_poses.sdf'
    args = ['--receptor', candidate['receptor'], '--ligand', candidate['ligand'],
            '--autobox_ligand', candidate['ligand'], '--autobox_add', '4',
            '--exhaustiveness', '8', '--num_modes', '5', '--min_rmsd_filter', '1',
            '--cnn_scoring', 'rescore', '--pose_sort_order', 'CNNscore',
            '--seed', '42', '--cpu', '2', '--no_gpu', '--out', output]
    _, provenance = _run('gnina', args, folder, CONFIG['timeout_seconds'])
    verify_candidate(candidate)
    candidates, failures = [], []
    for index, mol in enumerate(Chem.SDMolSupplier(str(output), removeHs=False)):
        if index >= CONFIG['num_modes']:
            raise ValueError('Unexpected excess docking poses.')
        if mol is None:
            failures.append({'pose': index, 'reason': 'Unreadable SDF record'})
            continue
        path = folder / f'pose-{index}.sdf'
        with Chem.SDWriter(str(path)) as writer:
            writer.write(mol)
        source = {key: candidate[key] for key in ('receptor', 'sequence', 'smiles')}
        source.update(ligand=str(path), parent_id=candidate['candidate_id'])
        child = import_candidate(source, folder / 'candidates')
        if not identity_check(child):
            failures.append({'pose': index, 'candidate_id': child['candidate_id'], 'reason': 'Identity/stereochemistry failed'})
            continue
        candidates.append(child)
    result = {'status': 'succeeded' if candidates else 'no_valid_identity_candidates',
              'parent_id': candidate['candidate_id'], 'candidates': candidates,
              'failures': failures, 'provenance': provenance, 'contract': contract,
              'gpu_usage': 'disabled; CPU-only', 'cpu_limit': 2,
              'cpu_seconds': None, 'cpu_seconds_status': 'not_measured',
              'limitations': ['Fixed predicted receptor and predicted pocket; not independent site validation.',
                             'Physical assessment and uniform rescoring remain required.']}
    atomic_json(folder / 'result.json', result)
    return result


def plan_experiment(state, candidate_id, question, hypothesis, expected_observation, evidence_ids):
    """Register one probe before execution; cited measurements must be read and current."""
    from tetra.research.decision_policy import references
    from tetra.orchestration.tool_registry import current_assessment
    from tetra.runtime.inputs import digest, stable
    observations = references(state, evidence_ids)
    candidate = state['candidates'].get(candidate_id)
    if not candidate or not current_assessment(state, candidate_id):
        raise ValueError('An assessed candidate is required.')
    verify_candidate(candidate)
    if not all(isinstance(v, str) and v.strip() for v in (question, hypothesis, expected_observation)):
        raise ValueError('Question, hypothesis and expected observation are required.')
    if not any(o['action'] == 'compare_structures' and not o.get('result', {}).get('error') and
               ((o['result'].get('original_id') == candidate_id and o['result'].get('original_sha256') == candidate['sha256']) or
                (o['result'].get('candidate_id') == candidate_id and o['result'].get('candidate_sha256') == candidate['sha256']))
               for o in observations):
        raise ValueError('Cite an actual candidate comparison for this probe.')
    if state.get('docking_experiments'):
        raise ValueError('This session permits one docking experiment; no automatic repeat.')
    plan = {'experiment_id': 'D1', 'candidate_id': candidate_id, 'candidate_sha256': candidate['sha256'],
            'question': question, 'hypothesis': hypothesis, 'expected_observation': expected_observation,
            'evidence_ids': evidence_ids, 'config': dict(CONFIG), 'intent': 'UNCERTAINTY_PROBE'}
    plan['plan_sha256'] = digest(stable(plan))
    state['docking_experiments'] = {'D1': {'plan': plan, 'status': 'planned'}}
    return plan


def execute_experiment(agent, state, experiment_id):
    from tetra.runtime.inputs import digest, stable
    record = state.get('docking_experiments', {}).get(experiment_id)
    if not record or record['status'] != 'planned':
        raise ValueError('A registered unexecuted docking plan is required.')
    plan = record['plan']
    if digest(stable({k:v for k,v in plan.items() if k != 'plan_sha256'})) != plan['plan_sha256'] or plan['config'] != CONFIG:
        raise ValueError('Frozen docking plan changed.')
    candidate = state['candidates'][plan['candidate_id']]
    if candidate['sha256'] != plan['candidate_sha256']:
        raise ValueError('Parent changed.')
    record['status'] = 'running'
    agent.save(state)
    try:
        result = dock_candidate(candidate, agent.folder(state)/'docking'/experiment_id,
                                plan['hypothesis'], plan['expected_observation'])
        for child in result['candidates']:
            state['candidates'][child['candidate_id']] = child
        state['selected'] = None
        record.update(status=result['status'], result=result)
    except Exception as error:
        record.update(status='failed', error=str(error))
        agent.save(state)
        raise
    agent.save(state)
    return {'experiment_id': experiment_id, **result}
