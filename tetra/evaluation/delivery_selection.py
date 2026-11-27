"""Qualified original predictions are deliverable, not computationally accepted."""
from tetra.evaluation.revision_policy import _checks, select_with_policy


def select_qualified(state):
    legacy = select_with_policy({**state, 'selection_policy_version': 'selection-policy-v3'})
    verdicts = legacy['revision_verdicts']
    details = {}
    qualified = []
    for key, candidate in state['candidates'].items():
        checks = _checks(state, key)
        failed = [name for name, value in checks.items() if value == 'FAIL']
        missing = [name for name, value in checks.items() if value == 'UNKNOWN']
        blockers = []
        if checks['identity'] != 'PASS':
            blockers.append('identity_not_verified')
        if missing:
            blockers.append('assessment_missing_or_stale')
        for name in ('sanitization', 'all_atoms_connected'):
            if name in checks and checks[name] != 'PASS':
                blockers.append(name + '_not_passed')
        revised = key in verdicts or bool(candidate.get('parent_id'))
        accepted_revision = verdicts.get(key, {}).get('decision') == 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE'
        if revised and not accepted_revision:
            blockers.append('revision_contract_not_met')
        accepted = not blockers and not failed and (not revised or accepted_revision)
        if not blockers and not accepted:
            qualified.append(key)
        details[key] = {
            'physical_quality': 'UNKNOWN' if missing else 'FAIL' if failed else 'PASS',
            'failed_checks': failed, 'missing_checks': missing,
            'blocking_reasons': blockers,
            'computational_acceptance': 'ACCEPTED' if accepted else 'NOT_ACCEPTED',
            'delivery_qualification': 'blocked' if blockers else 'accepted_candidate' if accepted else 'prediction_with_limitations',
        }
    # A qualified original is a fallback, never a promotion over an accepted candidate.
    selected = [key for key in legacy['selected_ids'] if not details[key]['blocking_reasons']]
    if not selected:
        selected = qualified
    ranking = legacy['ranking_basis']
    if not selected:
        ranking = 'no_deliverable_candidate; see candidate blocking_reasons'
    elif len(selected) == 1:
        ranking = 'single_deliverable_candidate; no ranking required'
    elif selected == qualified:
        ranking = 'original_registration_order; qualified candidates not scientifically ranked'
    qualification = details[selected[0]]['delivery_qualification'] if selected else 'diagnostic_only'
    return {**legacy, 'policy_version': 'selection-policy-v4',
            'selected_ids': selected, 'alternative_ids': [key for key in details if key not in selected],
            'candidate_decisions': details, 'delivery_qualification': qualification,
            'candidate_statuses': {key: ('QUALIFIED_PREDICTION' if key in selected and details[key]['computational_acceptance'] != 'ACCEPTED'
                                         else legacy['candidate_statuses'][key]) for key in details},
            'ranking_basis': ranking,
            'rule': 'Prefer accepted candidates; otherwise deliver fully assessed identity-correct originals with explicit defects. Failed revisions remain diagnostic.',
            'limitations': legacy['limitations'] + ['Qualified delivery is not physical validity, revision acceptance, or satisfaction of the scientific request.']}
