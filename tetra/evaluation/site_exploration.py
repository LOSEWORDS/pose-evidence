"""Narrow new-site hypothesis contracts, distinct from local repair and contact replay."""
import math
from tetra.runtime.inputs import digest, stable

VERSION = 'site-exploration-v1'


def freeze_site(state, plan, parent_distances):
    parent = state['candidates'][plan['candidate_id']]
    if parent.get('parent_id') or any(parent['candidate_id'] in p.get('candidate_ids', []) or
            parent['candidate_id'] == p.get('result', {}).get('candidate', {}).get('candidate_id')
            for p in state.get('revisions', {}).values()):
        raise ValueError('New-site search must start from an original candidate; local repair preserves its ancestor contracts.')
    pockets = plan['pocket_constraints']
    if len(pockets) != 1:
        raise ValueError('New-site search requires exactly one geometric pocket hypothesis.')
    requested = {index for _, index in pockets[0]['contacts']}
    if len(requested) < 3:
        raise ValueError('New-site hypothesis requires at least three measured pocket residues.')
    separation = max(8.0, pockets[0]['max_distance'] + 2.0)
    if any(parent_distances.get(index, -1) <= separation for index in requested):
        raise ValueError('New-site constraints must lie beyond the current ligand contacts and requested bound.')
    for key in plan.get('evidence_ids', []):
        observation = state['research']['evidence'].get(key, {}).get('observation', {})
        result = observation.get('result', {})
        if (observation.get('action') != 'detect_pockets' or result.get('candidate_id') != plan['candidate_id']
                or result.get('candidate_sha256') != parent['sha256']
                or result.get('input_kind') != 'receptor_only' or 'error' in result):
            continue
        for pocket in result.get('pockets', []):
            if requested <= set(pocket['input_positions']):
                return {'evidence_id': key, 'evidence_sha256': digest(stable(result)),
                        'pocket_id': pocket['pocket_id'], 'positions': sorted(requested),
                        'separation_from_old_ligand_A': separation,
                        'backbone_gate': {'rms_CA_max_A': 2.0, 'max_CA_displacement_A': 5.0,
                                          'source': 'predeclared conservative engineering bounds, not accuracy thresholds'},
                        'claim_scope': 'geometric_site_feasibility_only'}
    raise ValueError('Cite current receptor-only pocket detection containing all proposed residues.')


def evaluate_backbone(state, plan, child_id):
    contract = plan['contract']
    site = contract['site_hypothesis']
    observation = state['research']['evidence'].get(site['evidence_id'], {}).get('observation', {})
    if digest(stable(observation.get('result', {}))) != site['evidence_sha256']:
        return 'UNKNOWN'
    parent = state['candidates'].get(plan['candidate_id'], {})
    child = state['candidates'].get(child_id, {})
    for item in reversed(list(state['research']['evidence'].values())):
        obs = item.get('observation', {})
        result = obs.get('result', {})
        if (obs.get('action') != 'compare_structures' or result.get('original_id') != plan['candidate_id']
                or result.get('candidate_id') != child_id or 'error' in result
                or result.get('original_sha256') != parent.get('sha256')
                or result.get('candidate_sha256') != child.get('sha256')):
            continue
        values = [result.get('protein_CA_aligned_change_rms_A'), result.get('maximum_CA_displacement_A')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            return 'UNKNOWN'
        bounds = site['backbone_gate']
        return ('PASS' if values[0] <= bounds['rms_CA_max_A'] and values[1] <= bounds['max_CA_displacement_A']
                else 'FAIL')
    return 'UNKNOWN'
