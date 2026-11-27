"""Evidence-bound public experiment declarations and model reviews, not truth scoring."""
from copy import deepcopy

VERSION = 'experiment-v1'
PLAN_ACTIONS = {'plan_revision', 'plan_fragment_experiment'}
GATED_ACTIONS = {'plan_revision', 'plan_fragment_experiment', 'plan_docking', 'plan_site_challenge',
                 'execute_revision', 'execute_fragment_experiment', 'execute_docking',
                 'execute_site_challenge', 'select_candidates', 'audit_delivery', 'finish', 'stop'}
PREDICTION_OUTCOMES = {'prediction_completed', 'prediction_failed', 'prediction_import_failed'}
PREDICTION_GATED_ACTIONS = {'plan_prediction_experiment', 'execute_prediction_experiment', 'predict_complex'}
DESIGN_FIELDS = {'alternative', 'intervention', 'control', 'manipulation_check',
                 'support_if', 'refute_if', 'inconclusive_if'}
REVIEW_TEXT = {'experiment_id', 'manipulation_check', 'control_result', 'criteria_evaluation',
               'interpretation', 'next_decision'}
REVIEW_FIELDS = REVIEW_TEXT | {'evidence_ids', 'manipulation_status', 'hypothesis_outcome', 'limitations'}


def enabled(team):
    return team.get('experiment_workflow_version') == VERSION


def validate_design(team, decision):
    if not enabled(team):
        return
    inquiry = decision.get('inquiry') or {}
    design = inquiry.get('experiment')
    if design is None and decision['next_action'] not in PLAN_ACTIONS:
        return
    if not isinstance(design, dict) or set(design) != DESIGN_FIELDS or any(
            not isinstance(v, str) or not v.strip() for v in design.values()):
        raise ValueError('Planning requires an experiment declaration: ' + ', '.join(sorted(DESIGN_FIELDS)))
    if design['alternative'].strip() == inquiry.get('hypothesis', '').strip():
        raise ValueError('The competing explanation must differ from the hypothesis.')


def require_plan_arguments(state, action, arguments):
    team = state['research']
    if not enabled(team) or action not in PLAN_ACTIONS:
        return
    inquiry = (team['lead'].get('decision') or {}).get('inquiry') or {}
    expected = {'hypothesis': inquiry.get('hypothesis')}
    if action == 'plan_fragment_experiment':
        expected.update(question=inquiry.get('question'),
                        alternative=inquiry.get('experiment', {}).get('alternative'))
    if any(not isinstance(value, str) or not value.strip() for value in expected.values()):
        raise ValueError('Save a complete inquiry before planning.')
    if any(key in arguments and arguments[key] != value for key, value in expected.items()):
        raise ValueError('Plan hypothesis fields are bound to the saved inquiry. Omit them; update_memory first to change the inquiry.')
    arguments.update(expected)


def plan_schema(state, name, schema):
    """Omit duplicate prose inputs; the executor binds them to the saved inquiry."""
    if not enabled(state['research']) or name not in PLAN_ACTIONS:
        return schema
    result = deepcopy(schema)
    fields = {'hypothesis'} if name == 'plan_revision' else {'question', 'hypothesis', 'alternative'}
    params = result['function']['parameters']
    for field in fields:
        params['properties'].pop(field, None)
    params['required'] = [key for key in params['required'] if key not in fields]
    result['function']['description'] += ' Question/hypothesis/alternative are bound automatically to the current saved inquiry; do not retype them.'
    return result


def filter_tools(team, names):
    return [name for name in names if not pending_for_action(team, name)]


def sync(state):
    """Project executor evidence into a ledger; never alter immutable observations."""
    team = state['research']
    if not enabled(team):
        return
    ledger = team.setdefault('experiments', {})
    for evidence_id, evidence in team['evidence'].items():
        observation = evidence['observation']
        action = observation.get('action')
        result = observation.get('result', {})
        args = observation.get('arguments') or {}
        if action == 'plan_prediction_experiment' or action in PREDICTION_OUTCOMES:
            identifier = result.get('experiment_id')
            source = state.get('prediction_experiments', {}).get(identifier)
            if source is None or action == 'plan_prediction_experiment' and 'error' in result:
                continue
            key = 'prediction:' + identifier
            if action == 'plan_prediction_experiment':
                ledger.setdefault(key, {'experiment_id': key, 'plan_evidence_id': evidence_id,
                    'declaration': deepcopy(result), 'execution_status': 'planned',
                    'execution_evidence_ids': [], 'review': None})
            elif key in ledger:
                row = ledger[key]
                row['execution_evidence_ids'] = [evidence_id]
                row['execution_status'] = 'succeeded' if action == 'prediction_completed' else 'failed'
                config = source['config']
                condition_count = len(config.get('contact_constraints', [])) + sum(
                    len(pocket['contacts']) for pocket in config.get('pocket_constraints', []))
                row['conditioned'] = condition_count > 0
                row['condition_satisfied_candidates'] = [
                    candidate_id for candidate_id in result.get('candidate_ids', [])
                    if (distances := result.get('condition_measurements', {})
                        .get(candidate_id, {}).get('distances'))
                    and len(distances) == condition_count
                    and all(distance.get('satisfied') is True for distance in distances)
                ]
            continue
        if action not in ('plan_revision', 'execute_revision', 'plan_fragment_experiment', 'execute_fragment_experiment'):
            continue
        revision = action.endswith('revision')
        identifier = result.get('revision_id' if revision else 'experiment_id') or args.get('revision_id' if revision else 'experiment_id')
        if not identifier:
            continue
        source = state.get('revisions' if revision else 'fragment_experiments', {}).get(identifier)
        if source is None or (revision and source.get('method') != 'local_refinement'):
            continue
        key = ('revision:' if revision else 'fragment:') + identifier
        if action.startswith('plan_'):
            if 'error' in result:
                continue
            ledger.setdefault(key, {'experiment_id': key, 'plan_evidence_id': evidence_id,
                'declaration': deepcopy(observation.get('decision_snapshot', {})),
                'execution_status': 'planned', 'execution_evidence_ids': [], 'review': None})
        elif key in ledger:
            row = ledger[key]
            if evidence_id not in row['execution_evidence_ids']:
                row['execution_evidence_ids'].append(evidence_id)
            row['execution_status'] = 'failed' if 'error' in result else source.get('status', 'unknown')


def pending(team):
    if not enabled(team):
        return []
    return [key for key, row in team.get('experiments', {}).items()
            if row['execution_evidence_ids'] and row['execution_status'] not in ('planned', 'running')
            and row.get('review') is None]


def pending_for_action(team, action):
    keys = pending(team)
    if action in GATED_ACTIONS:
        return keys
    if action in PREDICTION_GATED_ACTIONS:
        return [key for key in keys if key.startswith('prediction:')]
    return []


def require_review(team, action, reviewed=()):
    missing = set(pending_for_action(team, action)) - set(reviewed)
    if missing:
        raise ValueError('Record an experiment review before another intervention or delivery: ' + ', '.join(sorted(missing)))


def bind_review_evidence(state, arguments):
    """Bind known provenance and propagate model citations; never infer an outcome."""
    bound = deepcopy(arguments)
    team = state['research']
    if not enabled(team):
        return bound, []
    decision = bound.get('decision') or {}
    audit = []
    for review in decision.get('experiment_reviews', []):
        key = review['experiment_id']
        if key not in pending(team):
            raise ValueError(f'Experiment {key} is not pending review; use an ID from experiment_ledger.')
        row = team['experiments'][key]
        automatic = [row['plan_evidence_id'], *row['execution_evidence_ids']]
        submitted = review['evidence_ids']
        if any(ref not in team['evidence'] for ref in [*submitted, *automatic]):
            raise ValueError(f'Experiment {key} review contains unknown evidence IDs.')
        review['evidence_ids'] = list(dict.fromkeys([*submitted, *automatic]))
        decision['evidence_ids'] = list(dict.fromkeys([*decision['evidence_ids'], *review['evidence_ids']]))
        bound['evidence_ids'] = list(dict.fromkeys([*bound['evidence_ids'], *review['evidence_ids']]))
        audit.append({'experiment_id': key, 'submitted_review_evidence_ids': list(submitted),
                      'automatically_linked_evidence_ids': automatic,
                      'rule': 'Link recorded plan/execution and propagate submitted review citations; no scientific inference.'})
    return bound, audit


def validate_reviews(team, decision):
    """Validate provenance and conservative failure handling; semantics remain model judgments."""
    reviews = decision.get('experiment_reviews', [])
    if not isinstance(reviews, list):
        raise ValueError('experiment_reviews must be a list.')
    seen = set()
    for review in reviews:
        if not enabled(team):
            raise ValueError('Experiment reviews require experiment-v1.')
        if not isinstance(review, dict) or set(review) != REVIEW_FIELDS or any(
                not isinstance(review[k], str) or not review[k].strip() for k in REVIEW_TEXT):
            raise ValueError('Incomplete experiment review.')
        key = review['experiment_id']
        if key in seen or key not in pending(team):
            raise ValueError('Review must identify one pending experiment, without duplicates or overwriting past reviews.')
        seen.add(key)
        row = team['experiments'][key]
        refs = review['evidence_ids']
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in team['evidence'] for ref in refs):
            raise ValueError('Experiment review references must identify real evidence.')
        required = {row['plan_evidence_id'], *row['execution_evidence_ids']}
        if not required.issubset(refs) or not set(refs).issubset(decision['evidence_ids']):
            missing_plan = sorted(required - set(refs))
            missing_decision = sorted(set(refs) - set(decision['evidence_ids']))
            raise ValueError(f'Experiment {key}: review missing plan/execution IDs {missing_plan}; decision.evidence_ids missing {missing_decision}.')
        if review['manipulation_status'] not in ('effective', 'ineffective', 'unknown'):
            raise ValueError('Invalid manipulation status.')
        if review['hypothesis_outcome'] not in ('supported', 'contradicted', 'inconclusive'):
            raise ValueError('Invalid hypothesis outcome.')
        if not isinstance(review['limitations'], list) or not review['limitations'] or any(
                not isinstance(v, str) or not v.strip() for v in review['limitations']):
            raise ValueError('Experiment review requires explicit limitations.')
        if (row['execution_status'] not in ('succeeded', 'completed') or
                review['manipulation_status'] != 'effective') and review['hypothesis_outcome'] != 'inconclusive':
            raise ValueError('Failed or unverified manipulation requires an inconclusive hypothesis outcome.')
        if (key.startswith('prediction:') and row.get('conditioned')
                and not row['condition_satisfied_candidates']):
            if (review['manipulation_status'] == 'effective'
                    or review['hypothesis_outcome'] != 'inconclusive'):
                raise ValueError('No prediction candidate passed all measured conditions; '
                                 'report unknown/ineffective and inconclusive.')
        if key.startswith('fragment:'):
            attempts = [attempt for ref in row['execution_evidence_ids']
                        for attempt in team['evidence'][ref]['observation'].get('result', {}).get('attempts', [])]
            effective_arm = any(attempt.get('status') == 'completed' and
                                attempt.get('manipulation_check', {}).get('status') == 'PASS'
                                for attempt in attempts)
            if not effective_arm and (review['hypothesis_outcome'] != 'inconclusive' or
                                      review['manipulation_status'] == 'effective'):
                raise ValueError('No fragment arm passed manipulation checks; report unknown/ineffective and inconclusive.')
        if key.startswith('revision:') and review['hypothesis_outcome'] != 'inconclusive':
            evaluations = [team['evidence'][ref]['observation'] for ref in refs]
            if not any(item.get('action') == 'evaluate_revision' and
                       item.get('result', {}).get('revision_id') == key.split(':', 1)[1] and
                       item['result'].get('acceptance_results') and 'error' not in item['result']
                       for item in evaluations):
                raise ValueError('A conclusive local-repair review requires its recorded evaluation evidence.')
    require_review(team, decision['next_action'], seen)


def commit_reviews(team, decision):
    for review in decision.get('experiment_reviews', []):
        team['experiments'][review['experiment_id']]['review'] = {
            **deepcopy(review), 'memory_version': team['lead']['version'],
            'kind': 'post_experiment_model_interpretation'}


def publish_outcomes(state):
    """Expose interpretation beside execution/acceptance, without changing either verdict."""
    for key, row in state['research'].get('experiments', {}).items():
        review = row.get('review')
        if key.startswith('revision:') and review:
            plan = state['revisions'][key.split(':', 1)[1]]
            plan['hypothesis_outcome'] = review['hypothesis_outcome'].upper()
            plan['hypothesis_review'] = deepcopy(review)


def extend_schema(decision_schema):
    text = {'type': 'string', 'minLength': 1}
    design = {'type': 'object', 'additionalProperties': False,
              'properties': {k: text for k in sorted(DESIGN_FIELDS)}, 'required': sorted(DESIGN_FIELDS)}
    decision_schema['properties']['inquiry']['properties']['experiment'] = design
    props = {k: text for k in REVIEW_TEXT}
    props.update(evidence_ids={'type': 'array', 'items': text, 'description': 'Cite measured results used in your interpretation. The executor links the recorded plan/execution and propagates these citations to decision and memory; all must have been read.'},
                 manipulation_status={'type': 'string', 'enum': ['effective', 'ineffective', 'unknown']},
                 hypothesis_outcome={'type': 'string', 'enum': ['supported', 'contradicted', 'inconclusive']},
                 limitations={'type': 'array', 'minItems': 1, 'items': text})
    decision_schema['properties']['experiment_reviews'] = {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False, 'properties': props, 'required': sorted(REVIEW_FIELDS)}}


POLICY = '''
EXPERIMENT-V1: Before plan_revision or plan_fragment_experiment, add inquiry.experiment:
alternative, intervention, control, manipulation_check, support_if, refute_if, inconclusive_if.
Planning tools bind question/hypothesis/alternative to the saved inquiry automatically. Do not retype them; use update_memory to change them before planning.
Describe actual implemented controls; a retained parent is not a sham. If no matched sham
exists, explicitly limit causal attribution. Do not invent a control run. Declare what
measurement would discriminate explanations and how it changes the original request decision.
After local repair or fragment execution, continue necessary measurements and deterministic
acceptance evaluation. Before another intervention, selection, audit or voluntary stop, add
experiment_reviews to update_memory.decision for every pending experiment in experiment_ledger.
Cite relevant measured results in each review. The executor links its recorded plan/execution
evidence and propagates review citations into decision and memory evidence_ids; all must be read.
Report manipulation_status, manipulation_check, control_result, criteria_evaluation,
hypothesis_outcome, interpretation, limitations and next_decision. Revisit EACH predeclared
criterion, including natural-language thresholds; unmeasured criteria remain unknown. Failed
or ineffective manipulation is inconclusive, not hypothesis refutation. Execution success,
computational acceptance and scientific support are separate. These reviews are explicitly
model interpretations, not independently validated biological findings. Do not force experiments.
'''
