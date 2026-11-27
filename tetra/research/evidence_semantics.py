"""Source boundaries, not a natural-language scientific entailment checker."""
from copy import deepcopy

WORKFLOW_ACTIONS = {'search_literature',
    'select_candidates', 'audit_delivery', 'register_trigger',
    'plan_revision', 'plan_docking', 'plan_fragment_experiment', 'plan_site_challenge',
    'interpret_site_challenge',
}
MEASUREMENT_ACTIONS = {
    'analyze_prediction_batch', 'assess_structure', 'measure_contacts', 'compare_structures', 'score_candidates',
    'inspect_contact_chemistry', 'measure_atom_geometry',
    'detect_pockets', 'measure_site_options', 'measure_fragment_variability',
    'prediction_completed', 'execute_revision', 'refine_structure', 'execute_docking',
    'execute_site_challenge', 'execute_fragment_experiment', 'evaluate_revision',
}
AUDIT_MODEL_FIELDS = {'reason', 'remaining_actions', 'unresolved_questions', 'problem_reviews'}


def record_type(observation: dict) -> str:
    action = observation.get('action')
    if action in WORKFLOW_ACTIONS:
        return 'workflow_record'
    if action in MEASUREMENT_ACTIONS:
        return 'computational_observation'
    if action == 'resolve_input':
        return 'input_record'
    return 'execution_record'


def audit_view(result: dict) -> dict:
    """Split model-submitted scientific rationale from executor audit metadata."""
    return {
        'workflow_result': {key: deepcopy(value) for key, value in result.items() if key not in AUDIT_MODEL_FIELDS},
        'model_interpretations': {key: deepcopy(value) for key, value in result.items() if key in AUDIT_MODEL_FIELDS},
        'support_boundary': 'Audit completion and disclosure are workflow facts; submitted reasons are model interpretations, not new scientific evidence.',
    }


def evidence_view(record: dict) -> dict:
    """Read projection; never change the archived record or its source hash."""
    view = deepcopy(record)
    from tetra.research.belief_tracking import fact_catalog, source_hash
    view['measurement_facts'] = fact_catalog(record)
    view.setdefault('measurement_source_sha256', source_hash(record))
    from tetra.research.fact_references import choices
    view['selectable_facts'] = choices(view)
    view['measurement_fact_boundary'] = 'Executor-derived values; paths identify observations, not validated mechanisms. Repeated measurements of one structure are not independent biological confirmation.'
    observation = view['observation']
    if observation.get('action') == 'assess_structure':
        from tetra.evaluation.assessment import measurement_semantics
        result = observation.get('result', {})
        if isinstance(result.get('raw_metrics'), dict):
            result['measurement_semantics'] = {
                **result.get('measurement_semantics', {}),
                **measurement_semantics(result['raw_metrics']),
            }
    model = deepcopy(view.get('model_interpretations', {}))
    for field in ('decision_snapshot', 'arguments'):
        if field in observation:
            model[field] = observation.pop(field)
    if observation.get('action') == 'audit_delivery':
        split = audit_view(observation.get('result', {}))
        observation['result'] = split['workflow_result']
        if split['model_interpretations'] or 'submitted_audit_interpretations' not in model:
            model['submitted_audit_interpretations'] = split['model_interpretations']
    view.update(epistemic_type=record_type(observation), model_interpretations=model,
                support_boundary='An E-ID identifies an archived record, not scientific truth. Workflow output and repeated interpretations do not independently support scientific hypotheses.',
                projection='source-separated read view; immutable archive retains original arguments, snapshots and result')
    return view


def validate_interpretation_sources(team: dict, item: dict, decisive: bool) -> None:
    scope = item.get('support_scope', 'scientific')
    if scope not in ('scientific', 'workflow', 'execution'):
        raise ValueError('Unknown interpretation support_scope.')
    refs = item.get('evidence_ids', [])
    if scope == 'scientific' and decisive and refs and all(
            team['evidence'][ref]['observation'].get('action') in WORKFLOW_ACTIONS for ref in refs):
        raise ValueError('Scientific support cannot come solely from workflow/planning/selection/audit records. Cite the applicable original observations or retain the hypothesis as unresolved; use support_scope=workflow only for workflow facts.')
