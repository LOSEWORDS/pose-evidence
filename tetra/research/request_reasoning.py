"""Public request-to-experiment provenance; never a validator of scientific truth."""
from copy import deepcopy

VERSION = 'request-v2'
PLAN_ACTIONS = {'plan_revision', 'plan_site_challenge', 'plan_fragment_experiment', 'plan_docking'}
INQUIRY_FIELDS = {'request_relation', 'question', 'hypothesis', 'result_implications'}


def enabled(state):
    return state.get('decision_workflow_version') == VERSION


def site_option_inventory(state):
    """Count only current candidate-bound measured options, separate from their utility."""
    reports = {key: row for key, row in state.get('site_options', {}).items()
               if key in state.get('candidates', {})
               and row.get('candidate_sha256') == state['candidates'][key]['sha256']}
    count = sum(bool(o.get('eligible')) for r in reports.values() for o in r.get('options', []))
    refs = [key for key, item in state.get('research', {}).get('evidence', {}).items()
            if item['observation']['action'] == 'measure_site_options'
            and item['observation'].get('result') in reports.values()]
    return {'status': 'not_measured' if not reports else
            'eligible_options_present' if count else 'no_eligible_options',
            'eligible_count': count, 'evidence_ids': refs,
            'experiment_status': {k: v['status'] for k, v in state.get('site_challenges', {}).items()},
            'meaning': 'Geometric eligibility only; utility and biological relevance remain hypotheses.'}


def validate_inquiry(team, decision):
    inquiry = decision.get('inquiry')
    if inquiry is None and not (enabled(team) and decision['next_action'] in PLAN_ACTIONS):
        return
    from tetra.prediction.experiment_lifecycle import enabled as experiment_enabled, validate_design
    allowed = INQUIRY_FIELDS | ({'experiment'} if experiment_enabled(team) else set())
    if not isinstance(inquiry, dict) or not INQUIRY_FIELDS.issubset(inquiry) or set(inquiry) - allowed or any(
            not isinstance(inquiry[k], str) or not inquiry[k].strip() for k in INQUIRY_FIELDS):
        raise ValueError('Planning needs inquiry: request_relation, question, hypothesis and result_implications.')
    validate_design(team, decision)
    if not decision['evidence_ids']:
        raise ValueError('An intervention inquiry must cite measured evidence.')


def require_selection(state):
    if enabled(state) and state.get('candidates') and (state.get('selected') is None
            or state.get('selection_fingerprint') != selection_fingerprint(state)):
        raise ValueError('Call select_candidates before audit_delivery and finish/stop; an empty eligible selection is allowed.')


def decision_snapshot(state):
    lead = state['research']['lead']
    return {'memory_version': lead['version'], 'request': state.get('goal', ''),
            'decision': deepcopy(lead.get('decision')),
            'hypotheses': deepcopy(lead.get('hypotheses', [])),
            'kind': 'pre_action_model_interpretation'}


def require_plan_inquiry(state, name):
    if not enabled(state) or name not in PLAN_ACTIONS:
        return
    lead = state['research']['lead']
    decision = lead.get('decision') or {}
    if decision.get('next_action') != name or lead.get('decision_consumed', False):
        raise ValueError('Update memory with a pre-action inquiry and this planning action first.')
    validate_inquiry(state['research'], decision)


def selection_fingerprint(state):
    """Bind selection to its inputs, excluding its own output and delivery bookkeeping."""
    from tetra.runtime.inputs import digest, stable
    relevant = {'assess_structure', 'score_candidates', 'compare_structures', 'measure_contacts',
                'evaluate_revision', 'execute_revision', 'execute_fragment_experiment', 'execute_docking'}
    return digest(stable({
        'candidates': {k: c['sha256'] for k, c in state.get('candidates', {}).items()},
        'assessments': state.get('assessments', {}), 'revisions': state.get('revisions', {}),
        'evidence': {k: v for k, v in state['research']['evidence'].items()
                     if v['observation']['action'] in relevant
                     and 'error' not in v['observation'].get('result', {})}}))
