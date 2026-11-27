"""Frozen, narrow computational acceptance contracts; no biological truth scoring."""
from copy import deepcopy
import math
from tetra.evaluation.assessment import CHECKS
from tetra.runtime.inputs import digest, stable

VERSION = 'revision-policy-v1'
POSE_ALIGNMENT = 'whole_chain_CA; no ligand fitting for pose RMSD'
METRICS = {
    'physical_checks': {'version': 'posebusters-runtime-config', 'unit': 'PASS/FAIL/UNKNOWN',
                        'measurement_tool': 'assess_structure', 'missing_policy': 'UNKNOWN'},
    'residue_contacts': {'version': 'heavy-atom-minimum-v1', 'unit': 'angstrom',
                         'measurement_tool': 'measure_contacts', 'reference_frame': 'same complex',
                         'missing_policy': 'UNKNOWN', 'tolerance_source': 'declared descriptor, not biological threshold'},
}
ACTION_POLICY = {
    'local_refinement': {'supported_failure_modes': ['local_geometry_or_clash'],
                         'unsupported_claims': ['binding_site_correction', 'accuracy_gain', 'large_scale_allostery']},
    'pocket_resampling': {'supported_failure_modes': ['candidate_derived_contact_feasibility'],
                          'unsupported_claims': ['independent_binding_site_validation', 'accuracy_gain']},
}
GEOMETRY = ('bond_lengths', 'bond_angles', 'internal_steric_clash',
            'minimum_distance_to_protein', 'volume_overlap_with_protein')


def _checks(state: dict, key: str) -> dict:
    candidate = state['candidates'].get(key)
    assessment = state['assessments'].get(key, {})
    valid = (candidate and assessment.get('status') == 'assessed'
             and assessment.get('candidate_sha256') == candidate['sha256'])
    values = assessment.get('checks', {}) if valid else {}
    return {name: values.get(name, 'unknown').upper()
            if values.get(name) in ('pass', 'fail') else 'UNKNOWN'
            for name in ('identity', *CHECKS)}


def _contacts(state: dict, key: str) -> tuple[dict | None, str | None]:
    candidate = state['candidates'].get(key)
    if not candidate:
        return None, None
    for evidence_id, evidence in reversed(list(state['research']['evidence'].items())):
        observation = evidence.get('observation', {})
        result = observation.get('result', {})
        if (observation.get('action') != 'measure_contacts' or result.get('candidate_id') != key
                or result.get('candidate_sha256') != candidate['sha256'] or 'error' in result):
            continue
        rows = result.get('residues')
        if not isinstance(rows, list) or not rows:
            continue
        distances = {}
        for row in rows:
            if not isinstance(row, dict):
                break
            index = row.get('input_position')
            value = row.get('minimum_distance_A')
            if (type(index) is not int or index < 1 or index in distances
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                break
            distances[index] = value
        else:
            if 'sequence' in candidate and set(distances) != set(range(1, len(candidate['sequence']) + 1)):
                continue
            return distances, evidence_id
    return None, None


def freeze_contract(state: dict, plan: dict) -> dict:
    """Compile operator-defined rules from pre-action evidence, never from child results."""
    parent_id = plan['candidate_id']
    checks = _checks(state, parent_id)
    if checks['identity'] != 'PASS':
        raise ValueError('Identity must be verified before planning a structural action.')
    distances, evidence_id = _contacts(state, parent_id)
    if distances is None:
        raise ValueError('Measure complete parent residue contacts before freezing the revision contract.')
    if plan['method'] == 'local_refinement':
        targets = [name for name in GEOMETRY if checks[name] == 'FAIL']
        if not targets:
            raise ValueError('No measured geometry failure supports local refinement.')
        target = {'metric': 'physical_checks', 'operator': 'fail_to_pass', 'checks': targets}
    elif plan['method'] in ('pocket_resampling', 'site_exploration'):
        target = {'metric': 'residue_contacts', 'operator': 'all_within',
                  'pockets': deepcopy(plan['pocket_constraints'])}
    else:
        raise ValueError('Unsupported action policy.')
    # A narrow preservation profile: retain parent contacts, while allowing new contacts.
    contract = {'policy_version': VERSION, 'parent_id': parent_id,
                'parent_sha256': state['candidates'][parent_id]['sha256'],
                'method': plan['method'], 'action_constraints': deepcopy(plan['pocket_constraints']),
                'hard_gates': ['identity', *CHECKS], 'target': target,
                'preserve_passed_checks': [key for key, value in checks.items() if value == 'PASS'],
                'preserve_contacts': {'positions': sorted(i for i, d in distances.items() if d <= 4.0),
                                      'max_distance_A': 4.0, 'evidence_id': evidence_id,
                                      'origin': 'model_derived', 'new_contacts_allowed': True},
                'metrics': deepcopy(METRICS), 'origin': 'model_derived',
                'claim_scope': 'computational_feasibility_only',
                'limits': ['Contact cutoff is an operator-declared descriptor, not a biological truth criterion.',
                           'Whole-chain displacement is diagnostic, not a frozen fold-preservation gate.',
                           'Natural-language goals beyond this supported contract remain unverified.']}
    if plan['method'] == 'site_exploration':
        from tetra.evaluation.site_exploration import freeze_site, VERSION as SITE_VERSION
        contract['policy_version'] = SITE_VERSION
        contract['site_hypothesis'] = freeze_site(state, plan, distances)
        contract['preserve_contacts']['positions'] = []
        contract['preserve_contacts']['origin'] = 'not_applicable_to_new_site'
        contract['limits'] = ['Protein-only cavity is a geometric hypothesis, not independent experimental binding evidence.',
                              'Fixed backbone bounds restrict this action to modest receptor changes.']
    profile = plan.get('pose_preservation')
    if profile is not None:
        if (plan['method'] != 'local_refinement' or not isinstance(profile, dict)
                or set(profile) != {'contact_positions', 'max_ligand_pose_rmsd_A'}):
            raise ValueError('pose_preservation is a contact subset and RMSD bound for local_refinement.')
        positions = profile['contact_positions']
        maximum = profile['max_ligand_pose_rmsd_A']
        if (not isinstance(positions, list) or not positions
                or any(type(i) is not int for i in positions) or len(positions) != len(set(positions))
                or not set(positions).issubset(contract['preserve_contacts']['positions'])):
            raise ValueError('pose_preservation.contact_positions must name distinct measured parent contacts.')
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('pose_preservation.max_ligand_pose_rmsd_A must be finite and positive.')
        contract['preserve_contacts']['positions'] = list(positions)
        contract['pose_preservation'] = {**deepcopy(profile), 'alignment': POSE_ALIGNMENT,
                                         'scope': 'engineering tolerance, parent-to-child comparison'}
    contract['inherited_plans'] = []
    for ancestor in state.get('revisions', {}).values():
        if parent_id in children(ancestor):
            verify_contract(ancestor)
            contract['inherited_plans'].append(deepcopy({key: ancestor[key] for key in (
                'revision_id', 'candidate_id', 'candidate_sha256', 'method', 'pocket_constraints', 'contract',
                'pose_preservation') if key in ancestor}))
    contract['measurement_versions'] = deepcopy(state['assessments'].get(parent_id, {}).get('versions', {}))
    contract['sha256'] = digest(stable(contract))
    return contract


def verify_contract(plan: dict) -> dict:
    contract = plan.get('contract')
    if not isinstance(contract, dict):
        raise ValueError('Missing pre-action acceptance contract; legacy runs require explicit retrospective replay.')
    data = {key: value for key, value in contract.items() if key != 'sha256'}
    if contract.get('policy_version') not in (VERSION, 'site-exploration-v1') or digest(stable(data)) != contract.get('sha256'):
        raise ValueError('Frozen acceptance contract changed or uses an unsupported version.')
    if (contract['parent_id'] != plan['candidate_id'] or contract['parent_sha256'] != plan['candidate_sha256']
            or contract['method'] != plan['method'] or contract['action_constraints'] != plan['pocket_constraints']):
        raise ValueError('Frozen contract and proposed action differ.')
    profile = contract.get('pose_preservation')
    if profile is not None or plan.get('pose_preservation') is not None:
        if profile is None or plan.get('pose_preservation') != {
                key: profile[key] for key in ('contact_positions', 'max_ligand_pose_rmsd_A')}:
            raise ValueError('Frozen pose-preservation conditions differ from the planned action.')
    return contract


def children(plan: dict) -> list[str]:
    if plan['method'] in ('pocket_resampling', 'site_exploration'):
        return plan.get('candidate_ids', [])
    key = plan.get('result', {}).get('candidate', {}).get('candidate_id')
    return [key] if key else []


def _measure_pose_preservation(state, contract, child_id):
    profile = contract.get('pose_preservation')
    if profile is None:
        return 'NOT_APPLICABLE', None, None
    child = state['candidates'].get(child_id)
    if not child:
        return 'UNKNOWN', None, None
    for eid, record in reversed(list(state['research']['evidence'].items())):
        observation = record['observation']
        data = observation.get('result', {})
        if (observation['action'] != 'compare_structures' or data.get('error')
                or data.get('original_id') != contract['parent_id']
                or data.get('original_sha256') != contract['parent_sha256']
                or data.get('candidate_id') != child_id or data.get('candidate_sha256') != child['sha256']):
            continue
        metric = data.get('ligand_pose_comparison', {})
        value = metric.get('symmetry_corrected_pose_rmsd_A')
        if (metric.get('status') == 'measured' and metric.get('alignment') == profile['alignment']
                and type(value) in (int, float) and math.isfinite(value) and value >= 0):
            return ('PASS' if value <= profile['max_ligand_pose_rmsd_A'] else 'FAIL'), eid, value
    return 'UNKNOWN', None, None


def evaluate_revision(state: dict, plan: dict, child_id: str) -> dict:
    contract = verify_contract(plan)
    checks = _checks(state, child_id)
    parent = state['candidates'].get(plan['candidate_id'], {})
    if parent.get('sha256') != contract['parent_sha256']:
        checks = dict.fromkeys(checks, 'UNKNOWN')
    if contract.get('measurement_versions') and state['assessments'].get(child_id, {}).get('versions') != contract['measurement_versions']:
        checks = dict.fromkeys(checks, 'UNKNOWN')
    distances, evidence_id = _contacts(state, child_id)
    preserve = contract['preserve_contacts']
    needed = preserve['positions']
    if distances is None or any(index not in distances for index in needed):
        preserved = 'UNKNOWN'
    else:
        preserved = 'PASS' if all(distances[i] <= preserve['max_distance_A'] for i in needed) else 'FAIL'
    target = contract['target']
    if target['metric'] == 'physical_checks':
        values = [checks[name] for name in target['checks']]
    else:
        values = []
        for pocket in target['pockets']:
            for _, index in pocket['contacts']:
                values.append('UNKNOWN' if distances is None or index not in distances else
                              'PASS' if distances[index] <= pocket['max_distance'] else 'FAIL')
    target_state = 'FAIL' if 'FAIL' in values else 'UNKNOWN' if not values or 'UNKNOWN' in values else 'PASS'
    backbone = 'NOT_APPLICABLE'
    if contract['policy_version'] == 'site-exploration-v1':
        from tetra.evaluation.site_exploration import evaluate_backbone
        backbone = evaluate_backbone(state, plan, child_id)
    pose, pose_evidence_id, pose_rmsd = _measure_pose_preservation(state, contract, child_id)
    states = [*checks.values(), preserved, target_state, backbone, pose]
    decision = ('REJECT' if 'FAIL' in states else 'UNRESOLVED' if 'UNKNOWN' in states
                else 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE')
    repair_eligible = (checks['identity'] == 'PASS' and any(checks[name] == 'FAIL' for name in GEOMETRY)
                       and backbone in ('PASS', 'NOT_APPLICABLE') and preserved == 'PASS'
                       and pose in ('PASS', 'NOT_APPLICABLE')
                       and (target['metric'] == 'physical_checks' or target_state == 'PASS'))
    result = {'policy_version': contract['policy_version'], 'backbone_preservation': backbone, 'contract_sha256': contract['sha256'],
            'revision_id': plan['revision_id'], 'candidate_id': child_id,
            'candidate_sha256': state['candidates'].get(child_id, {}).get('sha256'),
            'hard_gates': checks, 'target': target_state, 'preserve_contacts': preserved,
            'contact_evidence_id': evidence_id, 'decision': decision,
            'repair_eligible': repair_eligible,
            'biological_validation': 'NOT_ESTABLISHED', 'experimental_accuracy': 'NOT_EVALUATED',
            'next_step': 'consider_supported_repair' if decision == 'REJECT' and repair_eligible
                         else 'unsupported_failure_or_alternative_hypothesis' if decision == 'REJECT'
                         else 'obtain_required_measurements_or_stop' if decision == 'UNRESOLVED' else 'retain_evidence'}
    if contract.get('pose_preservation'):
        result.update(preserve_pose=pose, pose_evidence_id=pose_evidence_id, ligand_pose_rmsd_A=pose_rmsd)
    inherited = [evaluate_revision(state, ancestor, child_id) for ancestor in contract.get('inherited_plans', [])]
    result['inherited_results'] = inherited
    decisions = [decision] + [row['decision'] for row in inherited]
    result['decision'] = 'REJECT' if 'REJECT' in decisions else 'UNRESOLVED' if 'UNRESOLVED' in decisions else decision
    if any(row['decision'] != 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE' and not row['repair_eligible'] for row in inherited):
        result['repair_eligible'] = False
        if result['next_step'] == 'consider_supported_repair':
            result['next_step'] = 'ancestor_contract_unsatisfied; investigate_or_stop'
    if result['decision'] != decision:
        result['next_step'] = 'ancestor_contract_unsatisfied; investigate_or_stop'
    result['hypothesis_discrimination'] = 'REQUIRES_EVIDENCE_BACKED_REVIEW'
    result['accuracy_improvement'] = 'NOT_ESTABLISHED'
    result['candidate_status'] = ('VALID_ALTERNATIVE' if result['decision'] == 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE'
                                  else 'REJECTED' if result['decision'] == 'REJECT' else 'UNRESOLVED')
    return result


def select_with_policy(state: dict) -> dict:
    """Deliver accepted structures separately from retained/rejected alternatives."""
    if state.get('selection_policy_version') == 'selection-policy-v4':
        from tetra.evaluation.delivery_selection import select_qualified
        return select_qualified(state)
    conservative = state.get('selection_policy_version') in ('selection-policy-v2', 'selection-policy-v3')
    decisions = {}
    superseded = set()
    produced = set()
    for plan in state.get('revisions', {}).values():
        for key in children(plan):
            produced.add(key)
            verdict = evaluate_revision(state, plan, key)
            decisions[key] = verdict
            if not conservative and verdict['decision'] == 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE':
                superseded.add(plan['candidate_id'])
    selected = []
    alternatives = []
    for key in state['candidates']:
        accepted = (decisions[key]['decision'] == 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE'
                    if key in produced else all(value == 'PASS' for value in _checks(state, key).values()))
        if accepted and key not in superseded:
            selected.append(key)
        else:
            alternatives.append(key)
    if conservative:
        def confidence(key):
            values = state['candidates'][key].get('original_confidence', {})
            value = (values.get('baseline_ligand_rank_score', values.get('confidence_score'))
                     if state.get('selection_policy_version') == 'selection-policy-v3' else values.get('confidence_score'))
            return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else -1
        selected.sort(key=confidence, reverse=True)
    ranking = 'original_boltz_confidence'
    if state.get('selection_policy_version') == 'selection-policy-v3':
        from tetra.evaluation.external_science import rank_scored_candidates
        scored = rank_scored_candidates(selected, state['candidates'], state.get('pose_scores', {}))
        if scored is not None:
            selected, ranking = scored, 'gnina_cnn_score_complete_pool'
        else:
            ranking = 'original_boltz_confidence; GNINA incomplete or incomparable'
    return {'selected_ids': selected, 'ranking_basis': ranking, 'alternative_ids': alternatives,
            'candidate_statuses': {key: ('COMPUTATIONALLY_PREFERRED' if selected and key == selected[0]
                else 'VALID_ALTERNATIVE' if key in selected and state['candidates'][key].get('parent_id')
                else 'ORIGINAL_RETAINED' if key in selected else 'REJECTED_OR_UNRESOLVED') for key in state['candidates']},
            'policy_version': state['selection_policy_version'] if conservative else VERSION,
            'revision_verdicts': decisions, 'rule': ('Retain all accepted candidates; rank a complete comparable GNINA pool, otherwise use original baseline/Boltz ranking. No promotion from geometry alone.' if state.get('selection_policy_version') == 'selection-policy-v3' else 'Retain all accepted candidates; rank available original Boltz confidence, missing scores last. No child promotion from geometry alone.' if conservative else 'Frozen hard gates, target and preservation conditions; retain alternatives.'),
            'limitations': ['Acceptance means computational protocol satisfaction, not biological validation.',
                           'No target experimental structure or RMSD was used.']}
