"""Permit a non-mutating measurement to reconsider an unconsumed decision."""
VERSION = 'measurement-redirect-v1'
MEASUREMENTS = frozenset({
    'assess_structure', 'assess_structure_batch', 'measure_contacts', 'compare_structures',
    'detect_pockets', 'measure_site_options', 'measure_fragment_variability',
})
POLICY = '''
Within an already recorded research question, you may switch from the committed action to
an available non-mutating measurement (assessment, contacts, structural comparison or site/
fragment measurement) without first repeating update_memory. The executor records this as
an action deviation, not as support for your prior explanation. You must read and integrate
the new result before another scientific action. This does not bypass tool prerequisites,
experiment reviews, or the need to plan and commit any structural intervention or delivery.
'''


def allowed(state, name):
    if state.get('measurement_redirect_version') != VERSION or name not in MEASUREMENTS:
        return False
    team=state.get('research', {})
    lead=team.get('lead', {})
    decision=lead.get('decision') or {}
    if (team.get('review_required') or team.get('unreviewed_evidence')
            or lead.get('decision_consumed') or not decision
            or decision.get('next_action') == name):
        return False
    from tetra.research.question_board import is_open, validate_focus
    if not is_open(team):
        return False
    try:
        validate_focus(team, decision)
    except ValueError:
        return False
    return True


def transition(state, name):
    if not allowed(state, name):
        return None
    lead=state['research']['lead']
    return {'kind':'executor_recorded_action_deviation', 'policy':VERSION,
            'memory_version':lead['version'],
            'question_id':lead['decision']['research_focus']['question_id'],
            'committed_action':lead['decision']['next_action'], 'requested_action':name,
            'scope':'Non-mutating measurement; not a scientific justification or new hypothesis.'}
