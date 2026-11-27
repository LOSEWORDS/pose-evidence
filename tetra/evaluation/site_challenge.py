"""Matched predicted-site experiments with explicit computational claim scope."""
from copy import deepcopy
from pathlib import Path
import subprocess

import gemmi
import numpy as np
from rdkit import Chem

from tetra.evaluation.assessment import verify_candidate, identity_check, import_candidate, assess_candidate
from tetra.research.decision_policy import references
from tetra.runtime.download import atomic_json
from tetra.evaluation.external_science import available, tool_identity, _run
from tetra.evaluation.fragment_tools import physical_parent_valid
from tetra.runtime.inputs import digest, stable

CONFIG = {'seed': 42, 'exhaustiveness': 8, 'num_modes': 5, 'cpu': 2, 'timeout_seconds': 180}
def implementation_hash():
    root = Path(__file__).parent
    return digest(stable({name: digest((root / name).read_bytes()) for name in
        ('site_challenge.py', 'assessment.py', 'fragment_tools.py', 'external_science.py')}))


BRANCHES = {'ALTERNATIVE_FEASIBLE', 'A_ONLY_IN_TEST', 'INCONCLUSIVE'}


def validate_branches(branches):
    if set(branches) != BRANCHES or any(not isinstance(v, str) or not v.strip() for v in branches.values()):
        raise ValueError('Declare all concrete result-to-action branches.')
    if len({v.strip().lower() for v in branches.values()}) < 2:
        raise ValueError('At least two outcomes must change the follow-up action.')


def outcome(a, b):
    if a['status'] != 'completed' or b['status'] != 'completed':
        return 'INCONCLUSIVE'
    if a['valid_count'] and b['valid_count']:
        return 'ALTERNATIVE_FEASIBLE'
    if a['valid_count']:
        return 'A_ONLY_IN_TEST'
    return 'INCONCLUSIVE'


def positions(path):
    mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    if mol is None:
        raise ValueError('Unreadable ligand.')
    return mol, mol.GetConformer().GetPositions()


def measure(state, candidate_id, evidence_ids):
    obs = references(state, evidence_ids)
    parent = state['candidates'][candidate_id]
    verify_candidate(parent)
    reports = [o['result'] for o in obs if o['action'] == 'detect_pockets' and
               o['result'].get('candidate_id') == candidate_id and
               o['result'].get('candidate_sha256') == parent['sha256']]
    if len(reports) != 1:
        raise ValueError('Read one current receptor-only pocket report.')
    report = reports[0]
    if report.get('input_kind') != 'receptor_only':
        raise ValueError('Receptor-only cavity evidence required.')
    _, xyz = positions(parent['ligand'])
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    size = np.maximum(xyz.max(axis=0) - xyz.min(axis=0) + 8, 16)
    if np.max(size) > 30:
        raise ValueError('Ligand exceeds bounded search-box profile.')
    options = []
    for pocket in report['pockets']:
        path = Path(pocket['artifact_path'])
        if digest(path.read_bytes()) != pocket['artifact_sha256']:
            raise ValueError('Pocket artifact changed.')
        structure = gemmi.read_structure(str(path))
        coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for m in structure for c in m for r in c for a in r if a.element.name != 'H'])
        if len(coords) < 3:
            continue
        b = (coords.min(axis=0) + coords.max(axis=0)) / 2
        # Non-overlapping equal axis-aligned boxes, plus 2 A separation on one axis.
        distinct = bool(np.any(np.abs(b - center) > size + 2))
        options.append({'pocket_id': pocket['pocket_id'], 'center': b.tolist(), 'eligible': distinct,
                        'artifact_path': str(path), 'artifact_sha256': pocket['artifact_sha256'],
                        'input_positions': pocket['input_positions']})
    result = {'candidate_id': candidate_id, 'candidate_sha256': parent['sha256'],
              'site_A_center': center.tolist(), 'box_size': size.tolist(), 'options': options,
              'source_evidence_ids': evidence_ids, 'source_type': 'PREDICTION_DERIVED_GEOMETRY',
              'limitations': ['Cavity residue bounding-box centre is a search heuristic, not a binding site.',
                             'Eligibility is conservative box separation, not biological support.']}
    state.setdefault('site_options', {})[candidate_id] = result
    return deepcopy(result)


def plan(state, candidate_id, pocket_id, question, hypothesis_a, hypothesis_b, branches, evidence_ids):
    from tetra.orchestration.tool_registry import current_assessment
    if not state.get('allow_site_challenge') or state.get('site_challenges'):
        raise ValueError('One opt-in matched site experiment per session.')
    obs = references(state, evidence_ids)
    report = state.get('site_options', {}).get(candidate_id)
    if not report or not any(o['action'] == 'measure_site_options' and o['result'] == report for o in obs):
        raise ValueError('Read current site-option measurement first.')
    parent = state['candidates'][candidate_id]
    verify_candidate(parent)
    if parent['sha256'] != report['candidate_sha256'] or parent.get('parent_id'):
        raise ValueError('Current original parent required; ancestor contracts cannot be bypassed.')
    if not physical_parent_valid(current_assessment(state, candidate_id)):
        raise ValueError('Physically valid parent required.')
    if not all(isinstance(x, str) and x.strip() for x in (question, hypothesis_a, hypothesis_b)) or hypothesis_a.strip() == hypothesis_b.strip():
        raise ValueError('Question and distinct computational hypotheses required.')
    validate_branches(branches)
    option = next((o for o in report['options'] if o['pocket_id'] == pocket_id and o['eligible']), None)
    if option is None:
        raise ValueError('No eligible spatially separate site B.')
    molecule, _ = positions(parent['ligand'])
    if any(a.GetAtomicNum() not in (1,5,6,7,8,9,15,16,17,35,53) for a in molecule.GetAtoms()):
        raise ValueError('Unsupported ligand chemistry for noncovalent site challenge.')
    receptor = gemmi.read_structure(parent['receptor'])
    if receptor.connections or any(a.element.is_metal for m in receptor for c in m for r in c for a in r):
        raise ValueError('Metal or declared connection requires separate chemistry audit.')
    if not available('gnina'):
        raise ValueError('Validated GNINA required.')
    frozen = {'candidate_id': candidate_id, 'candidate_sha256': parent['sha256'], 'question': question,
              'hypotheses': [hypothesis_a, hypothesis_b], 'branches': deepcopy(branches),
              'centers': {'A': report['site_A_center'], 'B': option['center']}, 'box_size': report['box_size'],
              'option': deepcopy(option), 'config': dict(CONFIG), 'tool': tool_identity('gnina'),
              'implementation_sha256': implementation_hash(), 'evidence_ids': evidence_ids,
              'claim_scope': 'TESTED_REGIONS_AND_PROTOCOL_ONLY', 'independent_evidence': []}
    frozen['sha256'] = digest(stable(frozen))
    state['site_challenges'] = {'S1': {'status': 'planned', 'plan': frozen}}
    return {'experiment_id': 'S1', **deepcopy(frozen)}


def run_arm(parent, center, size, folder):
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / 'poses.sdf'
    args = ['--receptor', parent['receptor'], '--ligand', parent['ligand'], '--out', output,
            '--seed', CONFIG['seed'], '--exhaustiveness', CONFIG['exhaustiveness'], '--num_modes', CONFIG['num_modes'],
            '--cpu', CONFIG['cpu'], '--no_gpu', '--cnn_scoring', 'rescore', '--pose_sort_order', 'CNNscore']
    for axis, c, s in zip('xyz', center, size):
        args += ['--center_' + axis, c, '--size_' + axis, s]
    result = {'status': 'failed', 'valid_count': 0, 'poses': [], 'center': center, 'size': size}
    try:
        _, provenance = _run('gnina', args, folder, CONFIG['timeout_seconds'])
        result['provenance'] = provenance
        if not output.is_file():
            raise ValueError('Completed search has no readable output artifact.')
        unique = []
        for index, mol in enumerate(Chem.SDMolSupplier(str(output), removeHs=False)):
            if index >= CONFIG['num_modes'] or mol is None:
                raise ValueError('Malformed or excess docking output.')
            path = folder / f'pose-{index}.sdf'
            with Chem.SDWriter(str(path)) as writer:
                writer.write(mol)
            child = import_candidate({**{k: parent[k] for k in ('receptor', 'sequence', 'smiles')},
                                      'ligand': str(path), 'parent_id': parent['candidate_id']}, folder / 'candidates')
            m, xyz = positions(child['ligand'])
            in_box = bool(np.all(np.abs(xyz - np.array(center)) <= np.array(size) / 2 + 0.1))
            assessment = assess_candidate(child, folder / f'assessment-{index}')
            if assessment.get('status') != 'assessed' or any(v not in ('pass','fail') for v in assessment.get('checks', {}).values()):
                raise ValueError('Physical measurement unavailable; cannot interpret as negative.')
            valid = identity_check(child) and physical_parent_valid(assessment) and in_box
            duplicate = False
            for other in unique:
                matches = m.GetSubstructMatches(other, uniquify=False, useChirality=True, maxMatches=4097)
                if not matches or len(matches) > 4096:
                    valid = False
                    break
                a = other.GetConformer().GetPositions()
                if min(np.sqrt(np.mean(np.sum((a - xyz[list(match)]) ** 2, axis=1))) for match in matches) < 0.5:
                    duplicate = True
            if valid and not duplicate:
                unique.append(m)
                result['valid_count'] += 1
            result['poses'].append({'candidate': child, 'assessment': assessment, 'in_box': in_box,
                                    'valid': bool(valid), 'duplicate': duplicate})
        if not result['poses']:
            raise ValueError('Empty output is inconclusive, not a demonstrated negative.')
        result['status'] = 'completed'
    except subprocess.TimeoutExpired:
        result.update(status='timeout', error='Search timed out; partial output not interpreted.')
    except Exception as error:
        result.update(status='failed', error=str(error))
    verify_candidate(parent)
    atomic_json(folder / 'result.json', result)
    return result


def execute(agent, state, experiment_id):
    record = state.get('site_challenges', {}).get(experiment_id)
    if record and record['status'] == 'completed':
        return deepcopy(record['result'])
    if record and record['status'] == 'running':
        result = {'experiment_id': experiment_id, 'outcome': 'INCONCLUSIVE', 'arms': {},
                  'error': 'Interrupted experiment; retained artifacts, no automatic rerun.',
                  'claim_scope': 'TESTED_REGIONS_AND_PROTOCOL_ONLY'}
        record.update(status='completed', result=result)
        agent.save(state)
        return deepcopy(result)
    if not record or record['status'] != 'planned':
        raise ValueError('Unused frozen site plan required; no automatic rerun.')
    p = record['plan']
    if p['sha256'] != digest(stable({k:v for k,v in p.items() if k != 'sha256'})) or p['config'] != CONFIG:
        raise ValueError('Frozen plan changed.')
    if p['tool'] != tool_identity('gnina') or p['implementation_sha256'] != implementation_hash():
        raise ValueError('Scientific implementation changed.')
    parent = state['candidates'][p['candidate_id']]
    verify_candidate(parent)
    if parent['sha256'] != p['candidate_sha256'] or digest(Path(p['option']['artifact_path']).read_bytes()) != p['option']['artifact_sha256']:
        raise ValueError('Input evidence changed.')
    record['status'] = 'running'
    agent.save(state)
    folder = agent.folder(state) / 'site_challenges' / experiment_id
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'plan.json', p)
    arms = {key: run_arm(parent, center, p['box_size'], folder / key) for key, center in p['centers'].items()}
    result = {'experiment_id': experiment_id, 'arms': arms, 'outcome': outcome(arms['A'], arms['B']),
              'claim_scope': p['claim_scope'], 'plan_sha256': p['sha256'],
              'limitations': ['Two bounded searches do not identify the true site or prove uniqueness.',
                             'Different regions may have different search difficulty.',
                             'No comparison of GNINA scores establishes biological site preference.']}
    record.update(status='completed', result=result)
    # Experimental outputs remain evidence artifacts until a separate admission policy is validated.
    atomic_json(folder / 'result.json', result)
    agent.save(state)
    return deepcopy(result)


def interpret(state, experiment_id, evidence_ids, explanation, unresolved):
    observations = references(state, evidence_ids)
    record = state['site_challenges'][experiment_id]
    result = record.get('result')
    if not result or not any(o['action'] == 'execute_site_challenge' and o['result'] == result for o in observations):
        raise ValueError('Read the actual paired experiment result first.')
    if not isinstance(explanation, str) or not explanation.strip() or not unresolved:
        raise ValueError('Interpretation and unresolved limitations required.')
    interpretation = {'outcome': result['outcome'], 'next_action': record['plan']['branches'][result['outcome']],
                      'scientist_interpretation': explanation, 'unresolved': unresolved,
                      'claim_scope': 'COMPUTATIONAL_FEASIBILITY_OF_TESTED_REGIONS_ONLY', 'evidence_ids': evidence_ids}
    record['interpretation'] = interpretation
    return deepcopy(interpretation)
