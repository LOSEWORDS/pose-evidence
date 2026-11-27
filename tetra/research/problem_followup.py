"""Require explicit unresolved-problem dispositions, not a prescribed experiment."""
from copy import deepcopy

VERSION = 'problem-followup-v1'
OUTCOMES = ('attempted_unresolved', 'blocked', 'deferred_unresolved')
ATTEMPT_ACTIONS = {'execute_revision', 'refine_structure', 'execute_docking',
                   'execute_fragment_experiment', 'execute_site_challenge', 'compare_structures', 'prediction_completed'}
TEXT_FIELDS = ('problem_id', 'resolution_criteria', 'hypothesis', 'evidence_assessment',
               'next_test', 'stopping_boundary')


def enabled(state):
    return state.get('problem_followup_version') == VERSION


def problem_catalog(state, unresolved_questions, *, decision=None):
    from tetra.research.decision_policy import current_measured_limitations
    result = {}
    for candidate, checks in current_measured_limitations(state).items():
        for check in checks:
            result[f'defect:{candidate}:{check}'] = dict(kind='measured_defect', candidate_id=candidate,
                candidate_sha256=state['candidates'][candidate]['sha256'], check=check,
                status='unresolved', rule='A failed check cannot become resolved through a mechanistic explanation.')
    if decision is None:
        decision = state.get('research', {}).get('lead', {}).get('decision') or {}
    current = decision.get('unresolved_questions', [])
    for question in [*current, *unresolved_questions]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Unresolved questions must be nonempty strings.')
        result['question:' + question] = dict(kind='declared_unknown', question=question, status='unresolved')
    return result


def _related(state, observation, candidate):
    result = observation.get('result', {})
    arguments = observation.get('arguments', {})
    child = result.get('candidate', {})
    direct = candidate in [result.get('candidate_id'), result.get('original_id'), result.get('parent_id'),
                         arguments.get('candidate_id'), arguments.get('original_id'),
                         child.get('parent_id'), child.get('candidate_id')]
    if direct:
        return True
    plans = []
    revision_id = result.get('revision_id') or arguments.get('revision_id')
    if revision_id in state.get('revisions', {}):
        plans.append(state['revisions'][revision_id])
    if observation.get('action') == 'prediction_completed' and result.get('job_id'):
        plans.extend(plan for plan in state.get('revisions', {}).values() if plan.get('job_id') == result['job_id'])
    table = {'execute_site_challenge':'site_challenges', 'execute_docking':'docking_experiments',
             'execute_fragment_experiment':'fragment_experiments'}.get(observation.get('action'))
    experiment_id = result.get('experiment_id') or arguments.get('experiment_id')
    if table and experiment_id in state.get(table, {}):
        record = state[table][experiment_id]
        if record.get('status') != 'completed':
            return False
        plans.append(record.get('plan', {}))
    current_hash = state['candidates'][candidate]['sha256']
    for plan in plans:
        if (plan.get('candidate_id', plan.get('parent_id')) == candidate
                and plan.get('candidate_sha256', plan.get('parent_sha256')) == current_hash
                and (not result.get('plan_sha256') or result['plan_sha256'] == plan.get('sha256'))):
            return True
    return False


def validate_problem_reviews(state, unresolved_questions, reviews, *, decision=None, issues=None):
    from tetra.research.decision_policy import AuditValidationError, references
    own_issues = issues is None
    if own_issues:
        issues = []
    catalog = problem_catalog(state, unresolved_questions, decision=decision)
    if not isinstance(reviews, list):
        issues.append({'path': 'problem_reviews', 'message': 'Review every unresolved problem before stopping.'})
        if own_issues:
            raise AuditValidationError(issues)
        return []
    expected_fields = set(TEXT_FIELDS) | {'evidence_ids', 'attempt_evidence_ids', 'outcome'}
    seen = set()
    output = []
    for index, review in enumerate(reviews):
        field = None
        evidence_id = None
        key = review.get('problem_id') if isinstance(review, dict) else None
        try:
            if not isinstance(review, dict) or set(review) != expected_fields:
                raise ValueError('Invalid problem review fields.')
            if any(not isinstance(review[k], str) or not review[k].strip() for k in TEXT_FIELDS):
                raise ValueError('Problem reviews need criteria, hypothesis, evidence assessment, next test and stopping boundary.')
            key = review['problem_id']
            if key not in catalog or key in seen:
                raise ValueError('Unknown or duplicate unresolved problem ID.')
            seen.add(key)
            if review['outcome'] not in OUTCOMES:
                raise ValueError('These problems are still unresolved; do not mark them resolved or not_needed.')
            field = 'evidence_ids'
            basis = references(state, review['evidence_ids'])
            problem = catalog[key]
            if problem['kind'] == 'measured_defect' and not any(
                    obs.get('action') == 'assess_structure'
                    and obs.get('result', {}).get('candidate_id') == problem['candidate_id']
                    and obs.get('result', {}).get('candidate_sha256') == problem['candidate_sha256']
                    and obs.get('result', {}).get('checks', {}).get(problem['check']) not in (None, 'pass')
                    for obs in basis):
                raise ValueError('Review the current failed assessment for this candidate and check, not only a plan or interpretation.')
            field = 'attempt_evidence_ids'
            ids = review['attempt_evidence_ids']
            if not isinstance(ids, list) or any(not isinstance(ref, str) for ref in ids) or len(set(ids)) != len(ids):
                raise ValueError('Attempt evidence IDs must be distinct strings.')
            observations = references(state, ids) if ids else []
            if review['outcome'] == 'attempted_unresolved' and not observations:
                raise ValueError('An attempted solution needs actual execution evidence, not a plan.')
            if review['outcome'] == 'deferred_unresolved' and ids:
                raise ValueError('Unattempted deferral must not claim attempt evidence.')
            for evidence_id, observation in zip(ids, observations):
                if (observation.get('action') not in ATTEMPT_ACTIONS or observation.get('result', {}).get('error')
                        or observation.get('result', {}).get('status') in ('queued', 'running', 'pending', 'planned')):
                    raise ValueError('Planning, audit or rejected execution is not an executed scientific attempt.')
                problem = catalog[key]
                if problem['kind'] == 'measured_defect' and not _related(state, observation, problem['candidate_id']):
                    raise ValueError('Attempt evidence concerns a different candidate.')
            output.append(dict(deepcopy(review), resolved=False,
                               validation='coverage_and_execution_provenance_only; scientific adequacy remains a model judgment'))
        except ValueError as error:
            path = f'problem_reviews[{index}]'
            if field:
                path += '.' + field
            issue = {'path': path, 'problem_id': key, 'message': str(error)}
            if evidence_id:
                issue['evidence_id'] = evidence_id
            issues.append(issue)
    if seen != set(catalog):
        missing = sorted(set(catalog) - seen)
        issues.append({'path': 'problem_reviews', 'message': 'Missing problem reviews: ' + ', '.join(missing),
                       'missing_problem_ids': missing})
    if own_issues and issues:
        raise AuditValidationError(issues)
    return output


def extend_audit_schema(schema):
    properties = {key: {'type': 'string'} for key in TEXT_FIELDS}
    properties.update(outcome={'type':'string','enum':list(OUTCOMES)},
                      evidence_ids={'type':'array','items':{'type':'string'}},
                      attempt_evidence_ids={'type':'array','items':{'type':'string'}})
    schema['properties']['problem_reviews'] = {'type':'array', 'items':{
        'type':'object','additionalProperties':False,'properties':properties,'required':sorted(properties)},
        'description': 'Review every current measured defect and every unresolved_questions entry. Copy IDs from open_problem_catalog; additional question IDs are question: followed by the exact question. Define resolution, evidence, hypothesis and next informative test. Explain bounded stopping without assuming an unverified explanation. Unattempted deferral remains explicitly unresolved; attempt claims require actual read execution evidence.'}
    schema['required'].append('problem_reviews')


POLICY = '''
PROBLEM FOLLOW-UP BEFORE STOPPING:
An unresolved request-relevant problem is a reason to continue scientific consideration, not just
append a limitation. Revisit relevant Evidence and Memory; identify what remains unexplained and
revise or propose a testable hypothesis. Define observable resolution and preservation criteria
before any chosen intervention. Check whether possible outcomes actually distinguish explanations;
a result compatible with both does not resolve their difference. Mechanistic plausibility does not
clear a measured defect, and model confidence does not establish physical validity.
If a feasible, relevant test could materially change selection or the handling of a measured defect,
continue with that test. Do not replace it with an unsupported claim that intervention would harm
function. If an attempt fails, retain what was learned and consider a genuinely different justified
test; do not repeat unchanged failures or infer that all methods are impossible.
Before delivery, review every open_problem_catalog entry and every declared unresolved question in
problem_reviews: criteria, evidence assessment, hypothesis, next informative test, actual attempt IDs,
and stopping boundary. An unattempted problem is not resolved. Bounded stopping is possible after
failed or inconclusive attempts, actual capability limits, or an explicitly justified deferral under
the request; explain why the next test would not be warranted without assuming the unverified
hypothesis. Declaring a limitation alone is insufficient. Do not invent an execution to satisfy review.
Reuse exact question text/IDs for unchanged questions; new questions after audit require a new review.
A scientific question can be dispositioned for this run while its underlying defect remains unresolved.
Final delivery checks files, identity and limitations after this scientific review; audit completion
itself does not establish scientific adequacy. No fixed tool, distance, research route or Q count.
'''
