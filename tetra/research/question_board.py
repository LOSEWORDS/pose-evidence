"""Request-first public research questions, not a fixed scientific workflow."""
from copy import deepcopy

VERSION = 'questions-v2'
LEGACY_VERSION = 'questions-v1'
OPEN_FIELDS = {'question_id', 'question', 'request_relation', 'why_now', 'status'}
GUIDANCE_FIELDS = {
    'considerations': 'Relevant scientific perspectives for this question, not a required checklist.',
    'evidence_basis': 'What relevant read Evidence and prior Memory establish, and what they do not. Put actual citations in decision.evidence_ids.',
    'hypothesis_update': 'How those observations revise an existing hypothesis or motivate a new testable explanation; retain uncertainty where warranted.',
    'decision_implications': 'What possible results would change the next decision, what the proposed check cannot establish, or why limited stopping remains justified.',
}
OPTIONAL_FIELDS = {'alternative_question', 'why_not_alternative'} | set(GUIDANCE_FIELDS)
MODULES = {
    'request': {'question': 'What must the requested deliverable establish, and what remains unanswered?',
                'tools': ['resolve_input', 'audit_delivery']},
    'identity': {'question': 'Do sequence, ligand graph, stereochemistry, charge and represented chemical state match the input?',
                 'tools': ['resolve_input', 'assess_structure']},
    'mechanism': {'question': 'Which chemical/mechanistic interpretations are supported, and which are only hypotheses?',
                  'tools': ['measure_contacts', 'inspect_contact_chemistry', 'measure_atom_geometry'],
                  'limitation': 'Match any proposed mechanism to the actual atom chemistry; measure only reaction-relevant geometry. No online literature retrieval.'},
    'site': {'question': 'What supports the current binding region and what plausible alternatives remain?',
             'tools': ['predict_complex', 'detect_pockets', 'measure_site_options', 'plan_site_challenge', 'plan_docking']},
    'conformation': {'question': 'Where do candidates disagree: core, tail, orientation or protein? Which intervention could distinguish explanations?',
                     'tools': ['compare_structures', 'measure_fragment_variability', 'plan_fragment_experiment', 'plan_revision']},
    'interactions': {'question': 'Which interactions or geometric defects matter to the request, and could different causes explain them?',
                     'tools': ['measure_contacts', 'assess_structure', 'register_trigger', 'plan_revision']},
    'discrimination': {'question': 'Which feasible experiment would change the decision most, and what outcomes separate competing explanations?',
                       'tools': ['plan_revision', 'plan_site_challenge', 'plan_fragment_experiment', 'plan_docking']},
    'delivery': {'question': 'Which candidate is supported, what remains uncertain, and could further evidence change selection?',
                 'tools': ['score_candidates', 'select_candidates', 'audit_delivery', 'finish', 'stop']},
}
FIELDS = {'module', 'question', 'request_relation', 'why_now', 'alternative_question', 'why_not_alternative', 'status'}
STATUSES = ('investigating', 'supported_with_limits', 'unresolved', 'deferred')
EXEMPT = {'measure_reference_pocket', 'resolve_input', 'read_evidence', 'read_evidence_bundle', 'update_memory'}


def enabled(team):
    return team.get('question_board_version') in (LEGACY_VERSION, VERSION)


def validate_focus(team, decision):
    if not enabled(team):
        return
    focus = decision.get('research_focus')
    if is_open(team):
        if not isinstance(focus, dict) or not OPEN_FIELDS.issubset(focus) or set(focus) - OPEN_FIELDS - OPTIONAL_FIELDS or any(
                not isinstance(value, str) or not value.strip() for value in focus.values()):
            raise ValueError('research_focus needs a free question_id, question, request_relation, why_now and status.')
        if focus['status'] not in STATUSES:
            raise ValueError('Unknown research_focus status.')
        return
    if not isinstance(focus, dict) or set(focus) != FIELDS or any(
            not isinstance(value, str) or not value.strip() for value in focus.values()):
        raise ValueError('decision.research_focus must describe the selected research question, request relation, priority and alternative question.')
    if focus['module'] not in MODULES or focus['status'] not in STATUSES:
        raise ValueError('research_focus module/status must use the supplied board schema.')


def commit_focus(team, decision):
    if not enabled(team):
        return
    focus = decision['research_focus']
    team['lead'].setdefault('question_board', {})[focus['question_id'] if is_open(team) else focus['module']] = {
        **deepcopy(focus), 'evidence_ids': list(decision['evidence_ids']),
        'next_action': decision['next_action'], 'memory_version': team['lead']['version'],
        'kind': 'model_question_priority_and_interpretation'}


def require_focus(state, name):
    team = state['research']
    if not enabled(team) or name in EXEMPT:
        return
    from tetra.research.measurement_redirect import allowed
    if allowed(state, name):
        return
    lead = team['lead']
    decision = lead.get('decision') or {}
    if decision.get('next_action') != name or lead.get('decision_consumed', False):
        raise ValueError('First update_memory with research_focus and commit this next_action; choose the question before the tool.')
    validate_focus(team, decision)


def context(state):
    team = state['research']
    if not enabled(team):
        return None
    if is_open(team):
        board = {
            'current_questions': deepcopy(team['lead'].get('question_board', {})),
            'optional_reflection_fields': deepcopy(GUIDANCE_FIELDS),
            'rule': 'Generate questions from the request and observations. No predefined topics or order. '
                    'Untested is not refuted; deferred is not resolved. Alternatives are optional when warranted.',
        }
        if state.get('candidates'):
            board['optional_contact_chemistry'] = {
                'measurement_tools': ['inspect_contact_chemistry', 'measure_atom_geometry'],
                'when_relevant': 'If a chemical interpretation could change the decision, identify the measured pair '
                                 'and ligand group. Test explicit geometry only for a specified hypothesis.',
                'scope': 'Keep unmeasured chemical state and mechanisms unresolved. Chemistry measurements do not '
                         'replace a before-and-after experiment when the question is whether repair preserves a pose.',
            }
        if len(state.get('candidates', {})) >= 2:
            board['optional_candidate_questions'] = {
                'when_relevant': 'Use these prompts when an unmeasured similarity or difference could change '
                                 'candidate choice or how a measured defect is handled.',
                'questions': [
                    'Do the available candidates share a binding region, or could their ligand poses differ '
                    'in a way that changes the choice? Choose a decision-relevant pair to compare.',
                    'If the same check fails on several candidates, does it involve the same local contact '
                    'or different contacts? A shared failed check alone does not identify the contact.',
                    'If the contact region is shared, where do the aligned poses differ? What would a shared '
                    'defect or an alternative pose imply for comparison, revision, or bounded delivery?',
                ],
                'measurement_tools': ['compare_structures', 'measure_contacts'],
                'scope': 'Cite only measured pairs and localized candidates. Treat recurrence and local '
                         'differences as separate observations. No fixed pair count, tool order, or revision.',
            }
        return board
    return {'directions': deepcopy(MODULES), 'current_questions': deepcopy(team['lead'].get('question_board', {})),
            'rule': 'Select the most request-relevant unresolved question. These modules are prompts, not stages or proof of capability. Use currently_available_tools for actual availability. Do not force an experiment or mark an untested direction resolved.'}


def is_open(team):
    return team.get('question_board_version') == VERSION


def extend_schema(schema, team=None):
    if team is not None and is_open(team):
        schema['properties']['research_focus'] = {
            'type': 'object', 'additionalProperties': False,
            'properties': {key: {'type': 'string'} for key in sorted(OPEN_FIELDS | OPTIONAL_FIELDS)},
            'required': sorted(OPEN_FIELDS),
            'description': 'A request-grounded question with a stable self-chosen ID. No prescribed scientific topics, order, or mandatory alternative question.'}
        for key, description in GUIDANCE_FIELDS.items():
            schema['properties']['research_focus']['properties'][key]['description'] = description
        schema['properties']['research_focus']['properties']['status']['enum'] = list(STATUSES)
        schema['required'].append('research_focus')
        return
    properties = {key: {'type': 'string'} for key in FIELDS}
    properties['module']['enum'] = list(MODULES)
    properties['status']['enum'] = list(STATUSES)
    schema['properties']['research_focus'] = {'type': 'object', 'additionalProperties': False,
        'properties': properties, 'required': sorted(FIELDS),
        'description': 'Choose the scientific question before the next tool; compare its priority with another unresolved question. Concise public rationale, not private reasoning.'}
    schema['required'].append('research_focus')


POLICY = """
OPEN QUESTION BOARD:
Generate research questions from the original request and actual observations. There is no prescribed
set or sequence of scientific topics. Use a self-chosen stable question_id to track each question;
update its status as evidence changes. A question can span multiple actions, and a new observation
can create a different question. Do not create questions or experiments merely to fill a template.
Alternatives are optional: record them when a real ambiguity matters to the request, not to satisfy a quota.
Keep measured observations, interpretations and unknowns distinct. An unmeasured possibility is not
an absent or refuted possibility. Lack of observed variation without a comparison is not stability.
A model prediction does not independently validate its own interpretation. A score for one candidate
is not a comparative ranking. Use the recorded tool provenance rather than recalling method names.
Explain the expected decision value of an action, or why stopping with remaining uncertainty serves
the request. You may defer work without claiming it was disproved or that all search is exhausted.
No mandated number of questions, hypotheses, candidates, or experiments.
Selection-policy-v4 permits qualified original predictions with explicit defects, not computational
acceptance or biological validation. Failed revision contracts remain failed. Delivery permission is
not evidence that further investigation has no value. Preserve measured defects in the final report.
"""

POLICY += """
EVIDENCE-GUIDED HYPOTHESIS RENEWAL:
Use relevant perspectives (for example physical geometry, represented chemical state, biological
interpretation, or prediction uncertainty) to notice a question, not to choose a predetermined tool.
Then revisit the relevant read Evidence and prior Memory: distinguish measured observations from
prior model interpretations, identify their scope and dependencies, and retain unresolved defects.
Use that evidence to revise an existing hypothesis or propose a new testable explanation when
warranted. State what observation would support or weaken it; do not merely append 'unknown' and
continue relying on the same unestablished explanation. No new hypothesis is required for a clear result.
Choose an action for what it could change in the request-relevant judgment. Check that its possible
results could actually distinguish the proposed interpretations: establishing identity or proximity
alone does not establish chemical validity, function, or experimental accuracy.
If an explanation remains unestablished, reconsider whether the action or stop is still justified
without assuming that explanation. You may stop with disclosed limitations when the request and
available evidence justify doing so; uncertainty alone does not require another experiment.
Use the optional research_focus fields considerations, evidence_basis, hypothesis_update and
decision_implications to record this concise public update when useful. Cite actual read sources in
decision.evidence_ids and update the hypotheses list consistently. Fields guide inquiry, not completion
of a checklist; do not enumerate every perspective or invent evidence to fill them.
"""
