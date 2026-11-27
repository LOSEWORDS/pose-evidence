"""Public decision records on the scientific board, not private model reasoning."""
from copy import deepcopy

DECISION_FIELDS = {'next_action', 'reason', 'expected_observation', 'unresolved_questions', 'evidence_ids'}

SELECTION_FOLLOWUP_FIELDS = {
    'decision_gap': 'Name the unresolved request-relevant question that could change the choice.',
    'option_considered': 'Name a concrete potentially informative action, or explain which capabilities are missing.',
    'effect_on_selection': 'Describe how distinguishable outcomes would change the candidate choice, revision, or limitations.',
    'reason': 'Explain why to act now or why a bounded deferral is justified using the cited Evidence and available options.',
}
SELECTION_FOLLOWUP_GUIDANCE = (
    'Qualified selection preserves candidates with measured limitations. Check whether an available action '
    'could change which candidate to deliver or how to handle an unresolved defect. If so, investigate and '
    'commit the action. Before auditing with the defect unresolved, record a bounded deferral. For the most '
    'informative feasible action, compare what you would deliver if it satisfies the acceptance and '
    'preservation criteria versus if it fails them. Include any candidate produced by the proposed action '
    'in that comparison, even if one candidate exists now. Treat an untested risk as a hypothesis and '
    'distinguish it from a measured loss, a tool limitation, or a failed attempt. Before deferring, '
    'state what decision-relevant premise each cited record actually measures. Check whether the '
    'stopping rationale still holds with untested premises left unknown. If it depends on an unknown '
    'that an available action can resolve, commit that action. Use defer_basis for a qualified audit '
    'and carry unresolved claims into the delivery limitations. When chemistry matters to the decision, '
    'identify the exact contact and ligand graph first. Measure a named angle or dihedral only for a '
    'specified hypothesis. Do not use an untested reaction interpretation to clear a measured defect '
    'or skip a scientifically justified geometric test. If a bounded, reversible test could change '
    'delivery, use the possible pose disturbance to set preservation and rejection criteria before testing. '
    'Keep the original as a fallback. Distinguish a failed tool execution from measured failure to meet '
    'the criteria. Missing experimental validation '
    'does not by itself make computational comparisons or tests uninformative. '
    'Repeated reads of the same sources provide no new measurement. No specific tool or question order is required.'
)


def needs_revision_decision(team):
    for key in team['unreviewed_evidence']:
        observation = team['evidence'][key]['observation']
        result = observation.get('result', {})
        if observation.get('action') in ('prediction_failed', 'prediction_import_failed'):
            return True
        if observation.get('action') == 'execute_revision' and (result.get('error') or result.get('status') == 'failed'):
            return True
        if observation.get('action') == 'evaluate_revision' and any(
                row.get('decision') in ('REJECT', 'UNRESOLVED') for row in result.get('acceptance_results', [])):
            return True
    return False


def validate_decision(team, decision):
    from tetra.orchestration.tool_registry import TOOLS
    from tetra.evaluation.revision_tools import PARAMETERS
    if not isinstance(decision, dict) or not DECISION_FIELDS.issubset(decision) or set(decision) - (DECISION_FIELDS | {'inquiry', 'experiment_reviews', 'research_focus', 'selection_followup'}):
        raise ValueError('Board decision needs next_action, reason, expected_observation, unresolved_questions and evidence_ids.')
    if decision['next_action'] not in {*TOOLS, *PARAMETERS, 'update_memory'}:
        raise ValueError('Board next_action must be an exact tool name; got ' + repr(decision['next_action'])[:120])
    if any(not isinstance(decision[k], str) or not decision[k].strip() for k in ('reason', 'expected_observation')):
        raise ValueError('Board decision needs a concise rationale and expected observable result.')
    if not isinstance(decision['unresolved_questions'], list) or any(
            not isinstance(q, str) or not q.strip() for q in decision['unresolved_questions']):
        raise ValueError('Open scientific questions must be explicit nonempty strings.')
    from tetra.research.request_reasoning import validate_inquiry
    validate_inquiry(team, decision)
    refs = decision['evidence_ids']
    if not isinstance(refs, list) or any(not isinstance(k, str) or k not in team['evidence'] for k in refs):
        raise ValueError('Decision references must identify actual evidence.')
    if not set(team['unreviewed_evidence']).issubset(refs):
        raise ValueError('Next decision is missing new evidence IDs: ' + ', '.join(sorted(set(team['unreviewed_evidence']) - set(refs))))

    from tetra.prediction.experiment_lifecycle import validate_reviews
    validate_reviews(team, decision)
    from tetra.research.question_board import validate_focus
    validate_focus(team, decision)

def incremental_memory(team):
    return team.get('memory_submission_version') == 'incremental-v1'


def bind_review_citations(arguments):
    """Allow one shared review list while preserving explicit narrower decision citations."""
    result = deepcopy(arguments)
    decision = result.get('decision')
    if not isinstance(decision, dict):
        return result
    if 'evidence_ids' not in result and 'evidence_ids' in decision:
        result['evidence_ids'] = list(decision['evidence_ids']) if isinstance(decision['evidence_ids'], list) else decision['evidence_ids']
    elif 'evidence_ids' in result and 'evidence_ids' not in decision:
        decision['evidence_ids'] = list(result['evidence_ids']) if isinstance(result['evidence_ids'], list) else result['evidence_ids']
    return result


def bind_deferral_citations(arguments):
    """Include explicit measured-premise citations in the decision read/validation path."""
    decision = arguments.get('decision')
    if not decision:
        return
    followup = decision.get('selection_followup') or {}
    basis = followup.get('defer_basis') or {}
    if basis.get('kind') == 'measured':
        cited = [eid for row in basis['premises'] for eid in row['evidence_ids']]
        decision['evidence_ids'] = list(dict.fromkeys([*decision['evidence_ids'], *cited]))


def expand_memory_changes(team, arguments):
    """Resolve immutable text and lineage from the host for existing claim IDs."""
    result = deepcopy(arguments)
    result.setdefault('hypotheses', [])
    result.setdefault('withdraw_claim_ids', [])
    result['lessons'] = []
    claims = team['lead'].get('belief_state', {}).get('claims', {})
    for item in result['hypotheses']:
        previous = claims.get(item['claim_id'])
        if previous:
            item.setdefault('claim', previous['claim'])
            if previous.get('parent_claim_id') is not None:
                item.setdefault('parent_claim_id', previous['parent_claim_id'])
        elif not item.get('claim', '').strip():
            raise ValueError(f"{item['claim_id']}: new hypotheses require a claim statement.")
    return result


def memory_schema(state=None):
    from tetra.research.memory_tool import memory_tool_schema
    schema = deepcopy(memory_tool_schema())
    params = schema['function']['parameters']
    from tetra.orchestration.tool_registry import TOOLS
    from tetra.evaluation.revision_tools import PARAMETERS
    text = {'type': 'string'}
    params['properties']['decision'] = {'type': 'object', 'additionalProperties': False,
        'properties': {'next_action': {'type': 'string', 'enum': sorted({*TOOLS, *PARAMETERS, 'update_memory'})}, 'reason': text, 'expected_observation': text,
                       'unresolved_questions': {'type': 'array', 'items': text},
                       'evidence_ids': {'type': 'array', 'items': text}},
        'required': sorted(DECISION_FIELDS)}
    from tetra.research.request_reasoning import INQUIRY_FIELDS
    params['properties']['decision']['properties']['inquiry'] = {
        'type': 'object', 'additionalProperties': False,
        'properties': {k: {'type': 'string'} for k in sorted(INQUIRY_FIELDS)},
        'required': sorted(INQUIRY_FIELDS),
        'description': 'Required before planning an intervention. Link the request to a testable question, hypothesis and result-dependent next decisions.'}
    from tetra.prediction.experiment_lifecycle import enabled, extend_schema
    if state is not None and enabled(state['research']):
        extend_schema(params['properties']['decision'])
        if state.get('literature_enabled'):
            from tetra.prediction.experiment_lifecycle import pending
            reviews = params['properties']['decision']['properties']['experiment_reviews']
            pending_ids = list(pending(state['research']))
            if pending_ids:
                reviews['items']['properties']['experiment_id'] = {'type': 'string', 'enum': pending_ids}
                blocked = [key for key in pending_ids if key.startswith('prediction:')
                           and state['research']['experiments'][key].get('conditioned')
                           and not state['research']['experiments'][key].get('condition_satisfied_candidates')]
                if blocked:
                    fields = reviews['items']['properties']
                    reviews['description'] = (
                        'No candidate met all conditions for: ' + ', '.join(blocked) +
                        '. For these experiments use ineffective or unknown and inconclusive. '
                        'Describe any individually satisfied conditions separately.')
                    if len(blocked) == len(pending_ids):
                        fields['manipulation_status']['enum'] = ['ineffective', 'unknown']
                        fields['hypothesis_outcome']['enum'] = ['inconclusive']
            else:
                reviews['maxItems'] = 0
                reviews['description'] = 'No experiments are pending review. Omit or pass an empty array.'
    from tetra.research.question_board import enabled as questions_enabled, extend_schema as question_schema
    if state is not None and questions_enabled(state['research']):
        question_schema(params['properties']['decision'], state['research'])
    params['required'].append('decision')
    from tetra.research.belief_tracking import enabled as beliefs_enabled, extend_schema as belief_schema
    if state is not None and beliefs_enabled(state['research']):
        belief_schema(params)
        from tetra.research.fact_references import enabled as fact_ids_enabled, extend_schema as fact_id_schema
        if fact_ids_enabled(state['research']):
            fact_id_schema(params, state)
    if state is not None and incremental_memory(state['research']):
        params['properties']['decision']['required'].remove('evidence_ids')
        if (state.get('selected') or {}).get('delivery_qualification') == 'prediction_with_limitations':
            from tetra.evaluation.selection_grounding import defer_basis_schema
            params['properties']['decision']['properties']['selection_followup'] = {
                'type': 'object', 'additionalProperties': False,
                'properties': {
                    'disposition': {'type': 'string', 'enum': ['investigate', 'defer']},
                    'defer_basis': defer_basis_schema(),
                    **{key: {'type': 'string', 'description': description}
                       for key, description in SELECTION_FOLLOWUP_FIELDS.items()},
                },
                'required': ['disposition', *SELECTION_FOLLOWUP_FIELDS],
                'description': SELECTION_FOLLOWUP_GUIDANCE,
            }
        params['properties'].pop('lessons')
        params['required'] = [key for key in params['required'] if key not in ('hypotheses', 'lessons')]
        item = params['properties']['hypotheses']['items']
        item['required'].remove('claim')
        item['properties']['support_scope'] = {'type': 'string', 'enum': ['scientific']}
        params['properties']['hypotheses']['description'] = (
            'Submit only new or changed scientific claims. Omitted claims stay unchanged. '
            'For an existing claim_id omit claim and parent_claim_id; the host supplies both. '
            'For a new claim_id provide claim and optionally parent_claim_id. Never reuse an ID for different text.')
        params['properties']['withdraw_claim_ids'] = {
            'type': 'array', 'items': text,
            'description': 'Optional active IDs to remove from current consideration. Archived claims are preserved.'}
        schema['function']['description'] = (
            'Commit a decision and only the scientific claims that changed. '
            'Cite pending Evidence. Record execution or workflow facts in working_memory with their sources.')
    params['properties']['working_memory']['maxLength'] = 4000
    if state is not None:
        required_refs = ', '.join(state['research']['unreviewed_evidence'])
        guidance = 'Must include every pending feedback ID: ' + required_refs
        params['properties']['evidence_ids'] = deepcopy(params['properties']['evidence_ids'])
        params['properties']['evidence_ids']['description'] = guidance
        params['properties']['decision']['properties']['evidence_ids']['description'] = (
            'Optional explicit decision citations; omit to use the outer review Evidence list.'
            if incremental_memory(state['research']) else guidance)
    if state is not None and state.get('literature_enabled') and 'lessons' in params['properties']:
        lesson = params['properties']['lessons']
        from tetra.research.evidence_semantics import WORKFLOW_ACTIONS
        feedback = state['research']
        workflow_ids = [
            key for key in feedback['unreviewed_evidence']
            if feedback['evidence'][key]['observation']['action'] in WORKFLOW_ACTIONS
        ]
        if workflow_ids:
            schema['function']['description'] += (
                ' Pending workflow records: ' + ', '.join(workflow_ids) + '. '
                'Cite these IDs in the review evidence list and summarize their outcomes in '
                'working_memory. Keep selection and audit outcomes out of hypotheses and lessons. '
                'They cannot establish a scientific claim.')
        lesson['description'] = (
            'Use an empty list if no new scientific lesson is needed. Record selection, ranking, '
            'and audit outcomes in working_memory. For scientific lessons cite original measurements.'
        )
    schema['function']['description'] += (' State the next tool, why feedback justifies it, expected observation and unresolved questions. '
        'Keep the complete arguments under 12000 characters and working_memory under 4000 characters. Summarize current state, not historical tables; immutable evidence retains detail. This decision guides the next action. After a rejection change the hypothesis/action or explicitly justify fallback/stop; do not repeat unchanged experiments.')
    return schema


def check_committed_action(state, name, available_tools=None):
    from tetra.research.belief_tracking import require_current_claims
    require_current_claims(state, name)
    from tetra.prediction.experiment_lifecycle import require_review
    require_review(state['research'], name)
    if state.get('literature_enabled') and name == 'record_delivery_requirement':
        # A delivery requirement does not replace the scientific decision.
        return
    from tetra.research.measurement_redirect import allowed
    if allowed(state, name):
        return
    lead = state['research']['lead']
    decision = lead.get('decision')
    if (state.get('literature_enabled') and decision and name == 'assess_structure_batch'
            and decision['next_action'] == 'assess_structure'):
        return
    if state.get('literature_enabled') and decision:
        if (name in ('find_reference_complexes', 'search_literature', 'read_literature', 'read_structure_details',
                     'measure_reference_pocket')
                and available_tools is not None and name in available_tools):
            # Evidence gathering can inform a committed action. Availability still
            # enforces observation review and the search-before-reading workflow.
            return
        previous_action = decision['next_action']
        if (available_tools is not None and name in available_tools
                and previous_action not in available_tools):
            # Review and eligibility checks still apply to the replacement.
            return
        if (previous_action in ('read_evidence', 'read_evidence_bundle')
                and name in ('read_literature', 'plan_prediction_experiment', 'execute_prediction_experiment')):
            return
    from tetra.research.request_reasoning import enabled
    if enabled(state) and decision and decision['next_action'] in ('finish', 'stop'):
        if name in ('select_candidates', 'audit_delivery'):
            return
    if decision and not lead.get('decision_consumed', False) and name not in ('update_memory', 'read_evidence', 'read_evidence_bundle', 'stop', decision['next_action']):
        raise ValueError('Action differs from the current board decision; update the board with a justification first.')


def committed_prediction_retry(state, arguments, available_tools):
    """Reoffer a pending plan after malformed memory with no evidence to review."""
    team = state['research']
    lead = team['lead']
    action = 'plan_prediction_experiment'
    if (state.get('literature_enabled') and not team.get('review_required')
            and not lead.get('decision_consumed')
            and (lead.get('decision') or {}).get('next_action') == action
            and action in available_tools and isinstance(arguments, dict)
            and isinstance(arguments.get('decision'), str)):
        return action
    return None


def validate_memory_size(arguments):
    """Keep working memory concise; full observations remain in evidence files."""
    import json
    if len(arguments.get('working_memory', '')) > 4000:
        raise ValueError('Working memory exceeds 4000 characters; summarize and retain detail in evidence files.')
    if len(json.dumps(arguments, ensure_ascii=False)) > 12000:
        raise ValueError('Memory arguments exceed 12000 characters. Summarize current hypotheses and next decision; reference evidence IDs instead of copying measurements or history.')


def next_protocol_failure_count(previous, action, failed):
    if failed:
        return previous + 1
    if action in ('read_evidence', 'read_evidence_bundle'):
        return previous
    return 0


def validate_memory_shape(arguments, state):
    """Validate schema container types before code indexes nested model arguments."""
    schema = memory_schema(state)['function']['parameters']
    # Historical callers may omit a next decision; semantic gates handle that case.
    schema['required'] = [key for key in schema['required'] if key != 'decision']

    def check(value, spec, path):
        if 'enum' in spec and value not in spec['enum']:
            options = spec['enum']
            suffix = f' (showing 8 of {len(options)}; use the tool schema or read selectable_facts)' if len(options) > 8 else ''
            raise ValueError(f'{path} must be one of {options[:8]!r}{suffix}.')
        for clause in spec.get('allOf', []):
            check(value, clause, path)
        if 'if' in spec:
            try:
                check(value, spec['if'], path)
            except ValueError:
                check(value, spec.get('else', {}), path)
            else:
                check(value, spec.get('then', {}), path)
        kind = spec.get('type')
        types = {'object': dict, 'array': list, 'string': str,
                 'integer': int, 'number': (int, float), 'boolean': bool}
        if kind in types and (not isinstance(value, types[kind]) or
                              kind in ('integer', 'number') and isinstance(value, bool)):
            raise ValueError(f'{path} must be {kind}; got {type(value).__name__}. Use the structured tool schema.')
        if kind == 'object' or isinstance(value, dict) and ('properties' in spec or 'required' in spec):
            for key in spec.get('required', []):
                if key not in value:
                    raise ValueError(f'{path}.{key} is required.')
            properties = spec.get('properties', {})
            for key, item in value.items():
                if key in properties:
                    check(item, properties[key], f'{path}.{key}')
                elif spec.get('additionalProperties') is False:
                    raise ValueError(f'{path}.{key} is not an allowed field.')
        elif kind == 'array':
            for index, item in enumerate(value):
                check(item, spec.get('items', {}), f'{path}[{index}]')

    from tetra.research.belief_tracking import enabled as beliefs_enabled, lesson_scope_issue
    if beliefs_enabled(state['research']) and isinstance(arguments, dict):
        lessons = arguments.get('lessons')
        if isinstance(lessons, list):
            for index, lesson in enumerate(lessons):
                issue = lesson_scope_issue(lesson)
                if issue:
                    raise ValueError(f'update_memory.lessons[{index}].support_scope: {issue}')
    check(arguments, schema, 'update_memory')
