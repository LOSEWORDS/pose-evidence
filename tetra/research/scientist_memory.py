"""Versioned scientist boards with immutable observations and scoped interpretations."""
from copy import deepcopy
from typing import Any

ROLE_TOOLS = {
    'evidence': {'resolve_input'},
    'analysis': {'assess_structure', 'select_candidates'},
    'modeling': {'predict_complex', 'refine_structure'},
}


def initial_team(goal: str) -> dict[str, Any]:
    """Create session-local memory; no cross-case learning is loaded."""
    return {
        'version': 1,
        'lead': {'goal': goal, 'version': 0, 'working_memory': '',
                 'hypotheses': [], 'lessons': [], 'history': []},
        'boards': {role: [] for role in ROLE_TOOLS},
        'tasks': {}, 'evidence': {}, 'active_task': None,
        'review_required': False, 'unreviewed_evidence': [],
    }


def delegate(team: dict, scientist: str, objective: str,
             tools: list[str], max_steps: int) -> dict:
    """Issue a task with an isolated snapshot and explicit tool authorization."""
    if team['active_task'] or team['review_required']:
        raise ValueError('Read and integrate previous feedback before delegating.')
    if scientist not in ROLE_TOOLS or not isinstance(objective, str) or not objective.strip():
        raise ValueError('A valid scientist and explicit objective are required.')
    if not isinstance(tools, list) or not tools or any(
        not isinstance(name, str) or name not in ROLE_TOOLS[scientist] for name in tools
    ):
        raise ValueError('Tools exceed scientist permissions.')
    if type(max_steps) is not int or not 1 <= max_steps <= 8:
        raise ValueError('Specialist max_steps must be 1..8.')
    task_id = f"T{len(team['tasks']) + 1}"
    snapshot = {key: deepcopy(value) for key, value in team['lead'].items() if key != 'history'}
    task = {'id': task_id, 'scientist': scientist, 'objective': objective,
            'tools': list(tools), 'max_steps': max_steps, 'steps': 0,
            'lead_snapshot': snapshot, 'evidence_ids': [], 'status': 'running',
            'messages': []}
    team['tasks'][task_id] = task
    team['active_task'] = task_id
    return task


def record_observation(team: dict, observation: dict) -> str:
    """Only the executor calls this function; model text cannot create observations."""
    evidence_id = f"E{len(team['evidence']) + 1}"
    task_id = team['active_task']
    team['evidence'][evidence_id] = {
        'id': evidence_id, 'task_id': task_id,
        'kind': 'tool_observation', 'origin': observation_origin(observation),
        'independent_biological_evidence': False, 'observation': deepcopy(observation),
    }
    if task_id:
        team['tasks'][task_id]['evidence_ids'].append(evidence_id)
    team['unreviewed_evidence'].append(evidence_id)
    return evidence_id


def _references(team: dict, refs: list[str]) -> None:
    if not isinstance(refs, list) or any(
        not isinstance(ref, str) or ref not in team['evidence'] for ref in refs
    ):
        raise ValueError('Evidence references must identify recorded observations.')


def report_task(team: dict, summary: str, evidence_ids: list[str],
                status: str = 'reported') -> None:
    """Publish a specialist interpretation separately from raw observations."""
    if not team['active_task']:
        raise ValueError('No active task.')
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError('A result summary is required.')
    _references(team, evidence_ids)
    task = team['tasks'][team['active_task']]
    if any(ref not in task['evidence_ids'] for ref in evidence_ids):
        raise ValueError('Specialist report must cite its own task observations.')
    report = {'task_id': task['id'], 'lead_version': task['lead_snapshot']['version'],
              'status': status, 'summary': summary, 'evidence_ids': list(evidence_ids),
              'interpretation_is_model_generated': status == 'reported'}
    team['boards'][task['scientist']].append(report)
    task['status'] = status
    team['active_task'] = None
    team['review_required'] = True


def update_memory(team: dict, working_memory: str, hypotheses: list[dict],
                  lessons: list[dict], evidence_ids: list[str], decision: dict | None = None,
                  *, withdraw_claim_ids: list[str] | None = None) -> None:
    """Validate the complete update before changing the Lead Board."""
    from tetra.research.fact_references import enabled as fact_ids_enabled, normalize_arguments
    if fact_ids_enabled(team):
        normalized = normalize_arguments(team, {'hypotheses': hypotheses, 'lessons': lessons,
                                               'evidence_ids': evidence_ids, 'decision': decision})
        hypotheses, lessons = normalized['hypotheses'], normalized['lessons']
        evidence_ids, decision = normalized['evidence_ids'], normalized['decision']
    if team['active_task']:
        raise ValueError('Lead cannot update memory while a specialist owns the task.')
    if not isinstance(working_memory, str) or not working_memory.strip():
        raise ValueError('Working memory must be nonempty.')
    _references(team, evidence_ids)
    if not set(team['unreviewed_evidence']).issubset(evidence_ids):
        raise ValueError('Memory update is missing new evidence IDs: ' + ', '.join(sorted(set(team['unreviewed_evidence']) - set(evidence_ids))))
    if not isinstance(hypotheses, list) or not isinstance(lessons, list):
        raise ValueError('Hypotheses and lessons must be lists.')
    for item in hypotheses + lessons:
        if not isinstance(item, dict) or not isinstance(item.get('claim'), str) or not item['claim'].strip():
            raise ValueError('Each interpretation needs a claim.')
        _references(team, item.get('evidence_ids'))
        from tetra.research.evidence_semantics import validate_interpretation_sources
        try:
            validate_interpretation_sources(team, item, decisive=item in lessons or item.get('status') in ('supported', 'contradicted', 'weakened'))
        except ValueError as error:
            collection = 'lessons' if item in lessons else 'hypotheses'
            index = (lessons if collection == 'lessons' else hypotheses).index(item)
            raise ValueError(f"{collection}[{index}] claim={item['claim']!r}, evidence_ids={item['evidence_ids']!r}: {error}") from error
    from tetra.research.belief_tracking import enabled as beliefs_enabled, STATUSES, prepare_commit
    statuses = STATUSES if beliefs_enabled(team) else ('untested', 'supported', 'contradicted', 'unresolved')
    for hypothesis in hypotheses:
        if hypothesis.get('status') not in statuses:
            raise ValueError('Invalid hypothesis status; supported is not proven.')
    for lesson in lessons:
        if not lesson['evidence_ids'] or not isinstance(lesson.get('scope'), str) or not lesson['scope'].strip():
            raise ValueError('Lessons require observed evidence and an explicit scope.')
    from tetra.orchestration.board_loop import needs_revision_decision, validate_decision
    from tetra.prediction.experiment_lifecycle import pending, commit_reviews
    from tetra.research.question_board import enabled as questions_enabled, commit_focus
    if decision is None and questions_enabled(team):
        raise ValueError('Question-first board requires decision.research_focus.')
    if decision is None and pending(team):
        raise ValueError('Pending experiment review requires an explicit next decision.')
    if decision is None and needs_revision_decision(team):
        raise ValueError('Failed revision feedback requires an explicit next board decision, including justified fallback or stop.')
    if decision is not None:
        validate_decision(team, decision)
    belief_commit = (prepare_commit(team, hypotheses, lessons, withdraw_claim_ids=withdraw_claim_ids)
                     if beliefs_enabled(team) else None)
    lead = team['lead']
    if withdraw_claim_ids is not None:
        if belief_commit is None:
            raise ValueError('Incremental updates require the scientific claim ledger.')
        replaced = set(withdraw_claim_ids)
        replaced.update(item['claim_id'] for item in hypotheses + lessons)
        hypotheses = [h for h in lead['hypotheses'] if h.get('claim_id') not in replaced] + hypotheses
        lessons = [item for item in lead['lessons'] if item.get('claim_id') not in replaced] + lessons
    lead['history'].append({key: deepcopy(value) for key, value in lead.items() if key != 'history'})
    lead.update(version=lead['version'] + 1, working_memory=working_memory,
                hypotheses=deepcopy(hypotheses), lessons=deepcopy(lessons),
                reviewed_evidence_ids=list(evidence_ids), interpretation_is_model_generated=True)
    if belief_commit is not None:
        lead['belief_state'] = belief_commit[0]
        team.setdefault('belief_events', []).extend(belief_commit[1])
    if decision is not None:
        lead.update(decision=deepcopy(decision), decision_consumed=False)
        commit_reviews(team, decision)
        commit_focus(team, decision)
    elif 'decision' in lead:
        lead['decision_consumed'] = True
    team['review_required'] = False
    team['unreviewed_evidence'] = []


def evidence_summary(value: Any) -> Any:
    """Keep source identifiers and URLs while omitting bulky molecular payloads."""
    if isinstance(value, dict):
        return {key: evidence_summary(item) for key, item in value.items()
                if key not in ('sequence', 'smiles', 'raw_metrics')}
    if isinstance(value, list):
        return [evidence_summary(item) for item in value]
    return value


def observation_origin(observation: dict) -> str:
    """Assign provenance in the executor, never from an LLM-supplied origin label."""
    action = observation.get('action')
    if action in ('assess_structure', 'detect_pockets', 'score_candidates', 'measure_contacts', 'compare_structures', 'inspect_contact_chemistry', 'measure_atom_geometry', 'prediction_completed', 'execute_revision', 'refine_structure'):
        return 'model_derived'
    if action == 'resolve_input':
        return 'database_identity'
    if action == 'plan_revision':
        return 'model_interpretation'
    if action in ('evaluate_revision', 'select_candidates'):
        return 'policy_derived'
    return 'execution_record'
