"""Check decision provenance and completeness, not scientific truth of LLM claims."""
from tetra.research.evidence_files import require_reads
from tetra.runtime.inputs import digest, stable

ACTIONS = ('local_refinement', 'pocket_resampling', 'site_exploration', 'gather_measurements')


class AuditValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(f"{len(issues)} audit issue(s): {issues[0]['path']}: {issues[0]['message']}")


def audit_retry_exhausted(state, result):
    """Bound rejected submissions independently of successful Memory commits."""
    from tetra.research.evidence_semantics import MEASUREMENT_ACTIONS
    measurements = set()
    for record in state['research']['evidence'].values():
        observation = record['observation']
        measured = observation.get('result', {})
        if observation.get('action') in MEASUREMENT_ACTIONS and 'error' not in measured:
            measurements.add(digest(stable({
                'action': observation['action'],
                'result': {key: value for key, value in measured.items() if key != 'artifact_path'},
            })))
    fingerprint = digest(stable({
        'measurements': sorted(measurements),
        'candidates': {key: value['sha256'] for key, value in state['candidates'].items()},
        'selected_ids': (state.get('selected') or {}).get('selected_ids', []),
    }))
    progress = state.get('audit_retry_progress')
    if not progress or progress['fingerprint'] != fingerprint:
        progress = state['audit_retry_progress'] = {'fingerprint': fingerprint, 'signatures': []}
    paths = sorted({issue['path'] for issue in result.get('issues', [])})
    signatures = progress['signatures']
    signatures.append(paths or ['submission'])
    return signatures.count(signatures[-1]) >= 3 or len(signatures) >= 6


def required_actions(state):
    from tetra.research.question_board import is_open
    if is_open(state.get('research', {})):
        return []
    actions = list(ACTIONS)
    if state.get('allow_fragment_exploration'):
        actions.append('fragment_exploration')
    if state.get('allow_docking_exploration'):
        actions.append('docking_exploration')
    if state.get('allow_site_challenge'):
        actions.append('site_challenge')
    return actions


def references(state, ids, *, field=None):
    prefix = f'{field}: ' if field else ''
    if not isinstance(ids, list) or not ids:
        raise ValueError(prefix + 'Cite actual evidence read by the Lead. Supply applicable Evidence IDs for this entry.')
    evidence = state['research']['evidence']
    if any(not isinstance(key, str) or key not in evidence for key in ids):
        raise ValueError(prefix + 'Unknown evidence reference.')
    try:
        require_reads(state, ids)
    except ValueError as error:
        raise ValueError(prefix + str(error)) from error
    return [evidence[key]['observation'] for key in ids]


def measured_pose_difference(result):
    """Numerical difference is eligible evidence, never proof of useful exploration."""
    import math
    measurement = result.get('ligand_pose_comparison', {})
    value = measurement.get('symmetry_corrected_pose_rmsd_A')
    return (measurement.get('status') == 'measured' and type(value) in (int, float)
            and math.isfinite(value) and value > 1e-6)


def validate_trigger(state, candidate_id, trigger_type, evidence_ids, goal_quote, question):
    observations = references(state, evidence_ids)
    candidate = state['candidates'].get(candidate_id)
    if not candidate or not isinstance(question, str) or not question.strip():
        raise ValueError('A registered candidate and testable question are required.')
    matched = [o for o in observations if o.get('result', {}).get('candidate_id') == candidate_id
               and o['result'].get('candidate_sha256') == candidate['sha256']
               and not o['result'].get('error')]
    if trigger_type == 'physical_issue':
        valid = any(o['action'] == 'assess_structure' and
                    'fail' in o['result'].get('checks', {}).values() for o in matched)
    elif trigger_type == 'candidate_disagreement':
        matched = [o for o in observations if o['action'] == 'compare_structures' and not o.get('result', {}).get('error')
                   and candidate_id in (o['result'].get('original_id'), o['result'].get('candidate_id'))]
        valid = any(o['action'] == 'compare_structures' and
                    o['result'].get('original_id') in state['candidates'] and
                    o['result'].get('candidate_id') in state['candidates'] and
                    o['result']['original_id'] != o['result']['candidate_id'] and
                    o['result'].get('candidate_sha256') == state['candidates'][o['result']['candidate_id']]['sha256'] and
                    o['result'].get('original_sha256') == state['candidates'][o['result']['original_id']]['sha256'] and
                    (o['result'].get('lost_contact_positions') or o['result'].get('gained_contact_positions')
                     or measured_pose_difference(o['result']))
                    for o in matched)
    elif trigger_type == 'user_objective':
        valid = isinstance(goal_quote, str) and len(goal_quote.strip()) >= 12 and goal_quote in state['goal'] and goal_quote in state.get('revision_objectives', []) and bool(matched)
    else:
        raise ValueError('Unsupported trigger: independent literature evidence is not implemented.')
    if not valid:
        raise ValueError('Trigger lacks matching measured evidence or an exact user-goal quotation. A cavity alone is insufficient.')
    return {'candidate_id': candidate_id, 'candidate_sha256': candidate['sha256'],
            'trigger_type': trigger_type, 'evidence_ids': evidence_ids, 'goal_quote': goal_quote,
            'testable_question': question, 'evidence_origin': 'prediction_derived',
            'independent_support': False, 'validation': 'provenance_and_structure_only',
            'limitations': ['Question relevance and expected scientific benefit remain model judgments.',
                             'Pose differences above numerical tolerance are eligible observations, not a calibrated trigger threshold or evidence of error.']}


def audit_fingerprint(state):
    payload = {'candidates': {k: c['sha256'] for k, c in state['candidates'].items()},
               'selected': state.get('selected'), 'revisions': state.get('revisions', {}),
               'evidence': {k: v for k, v in state['research']['evidence'].items()
                            if v['observation']['action'] not in ('audit_delivery', 'rejected_action')}}
    # Preserve historical audit hashes for sessions without the new tool family.
    if state.get('allow_fragment_exploration') or state.get('allow_docking_exploration'):
        payload['fragment_experiments'] = state.get('fragment_experiments', {})
        payload['required_stop_actions'] = required_actions(state)
        payload['docking_experiments'] = state.get('docking_experiments', {})
    if state.get('allow_site_challenge'):
        payload['site_challenges'] = state.get('site_challenges', {})
        payload['required_stop_actions'] = required_actions(state)
    return digest(stable(payload))


def current_measured_limitations(state):
    """Current selected-candidate failures cannot be erased by a model rationale."""
    result = {}
    for key in (state.get('selected') or {}).get('selected_ids', []):
        candidate = state.get('candidates', {}).get(key, {})
        assessment = state.get('assessments', {}).get(key, {})
        if assessment.get('status') != 'assessed' or assessment.get('candidate_sha256') != candidate.get('sha256'):
            result[key] = ['assessment_missing_or_stale']
        else:
            failed = sorted(name for name, value in assessment.get('checks', {}).items() if value != 'pass')
            if failed:
                result[key] = failed
    return result


def initial_prediction_required(state):
    return (not state['candidates'] and bool(state.get('resolved', {}).get('input_id'))
            and state.get('prediction_count', 0) == 0)


def validate_stop(state, evidence_ids, remaining_actions, reason, unresolved_questions, measured_limitations=None, problem_reviews=None):
    references(state, evidence_ids, field='evidence_ids')
    from tetra.research.request_reasoning import enabled, require_selection, site_option_inventory
    require_selection(state)
    if initial_prediction_required(state):
        raise ValueError('Initial prediction remains available; resolve the primary task before scientific stop.')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('State the reason for stopping.')
    if not isinstance(unresolved_questions, list) or any(not isinstance(q, str) for q in unresolved_questions):
        raise ValueError('Unresolved questions must be a list of strings.')
    if not isinstance(remaining_actions, list) or any(not isinstance(row, dict) for row in remaining_actions):
        raise ValueError('Remaining actions must be structured records.')
    from tetra.research.question_board import is_open
    open_board = is_open(state['research'])
    expected_limitations = current_measured_limitations(state)
    if open_board and (measured_limitations or {}) != expected_limitations:
        raise ValueError('measured_limitations must preserve current measured failures: ' + str(expected_limitations))
    if not open_board and sorted(row.get('action', '') for row in remaining_actions) != sorted(required_actions(state)):
        raise ValueError('Audit every implemented action family: ' + ', '.join(required_actions(state)))
    issues = []
    for index, row in enumerate(remaining_actions):
        path = f'remaining_actions[{index}]'
        field = path
        action = row.get('action')
        try:
            if not isinstance(action, str) or not action.strip():
                field += '.action'
                raise ValueError('Describe the considered action.')
            if row.get('disposition') not in ('not_needed', 'unsupported_by_evidence', 'unavailable', 'deferred'):
                field += '.disposition'
                raise ValueError('Record an explicit disposition for each remaining action.')
            for key in ('reason', 'expected_information'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    field = f'{path}.{key}'
                    raise ValueError('Explain the action and its potentially discriminating observation.')
            field = f'{path}.evidence_ids'
            references(state, row.get('evidence_ids'))
            if enabled(state) and action == 'site_challenge':
                inventory = site_option_inventory(state)
                if (not open_board or 'site_option_status' in row) and row.get('site_option_status') != inventory['status']:
                    field = f'{path}.site_option_status'
                    raise ValueError('site_option_status must match measured inventory: ' + inventory['status'])
                if not set(inventory['evidence_ids']).issubset(row['evidence_ids']):
                    raise ValueError('Site challenge audit must cite current site-option evidence, even when deferred.')
        except ValueError as error:
            issues.append({'path': field, 'action': action, 'message': str(error)})
    from tetra.research.problem_followup import enabled as followup_enabled, validate_problem_reviews
    reviewed_problems = validate_problem_reviews(state, unresolved_questions, problem_reviews, issues=issues) if followup_enabled(state) else None
    if issues:
        raise AuditValidationError(issues)
    return {'reason': reason, 'reason_kind': 'model_judgment_not_verified_scientific_conclusion',
            'measured_limitations': expected_limitations, 'unresolved_questions': unresolved_questions,
            'remaining_actions': remaining_actions, 'fingerprint': audit_fingerprint(state),
            **({'problem_reviews': reviewed_problems} if reviewed_problems is not None else {}),
            **({'site_option_inventory': site_option_inventory(state)} if enabled(state) else {}),
            'evidence_ids': evidence_ids, 'validation': 'provenance_only_no_exhaustiveness_claim' if open_board else 'coverage_and_provenance_only',
            'limitations': ['Action utility is a model judgment; this audit does not prove search exhaustion.']}


def delivery_audit_ready(state):
    """Check audit presence and freshness without changing session state."""
    if state.get('decision_policy_version') != 'decision-v1':
        return True
    if not state['candidates'] and state.get('resolved', {}).get('status') == 'input_error':
        return True
    fingerprint = (state.get('delivery_audit') or {}).get('fingerprint')
    return bool(fingerprint and fingerprint == audit_fingerprint(state))


def has_current_empty_selection(state):
    """Identify a completed selection that found no deliverable candidate."""
    from tetra.research.request_reasoning import selection_fingerprint

    selection = state.get('selected')
    return (bool(state.get('literature_enabled') and state.get('candidates'))
            and isinstance(selection, dict) and selection.get('selected_ids') == []
            and state.get('selection_fingerprint') == selection_fingerprint(state))


def require_stop_audit(state):
    from tetra.research.request_reasoning import require_selection
    require_selection(state)
    if has_current_empty_selection(state):
        return
    if state.get('decision_policy_version') != 'decision-v1':
        return
    if not state['candidates'] and state.get('resolved', {}).get('status') == 'input_error':
        state['delivery_audit'] = {'validation': 'deterministic_input_failure',
                                   'reason': 'No valid input or candidate; scientific revision is unavailable.'}
        return
    if not delivery_audit_ready(state):
        raise ValueError('Call audit_delivery with current evidence and remaining actions before ending.')
    require_delivery_problem_reviews(state)


def require_delivery_problem_reviews(state, *, decision=None):
    """Validate audit coverage against committed or proposed unresolved questions."""
    from tetra.research.problem_followup import enabled as followup_enabled, validate_problem_reviews
    if not followup_enabled(state):
        return
    audit = state['delivery_audit']
    reviews = audit.get('problem_reviews')
    if isinstance(reviews, list):
        reviews = [{k: v for k, v in row.items() if k not in ('resolved', 'validation')} for row in reviews]
    validate_problem_reviews(state, audit.get('unresolved_questions', []), reviews, decision=decision)
