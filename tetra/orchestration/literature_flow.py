"""Retrieval readiness and evidence delivery for conditioned prediction."""

from tetra.orchestration.tool_registry import TOOLS


def deliver_search_result(state, result):
    """Record delivery of discovery metadata without triggering scientific review."""
    team = state['research']
    evidence_id = next(reversed(team['evidence']))
    team['unreviewed_evidence'].remove(evidence_id)
    team['review_required'] = bool(team['unreviewed_evidence'])
    payload = {'evidence_id': evidence_id, 'discovery_result': result,
               'scope': 'Search results only. Screen a relevant source to acquire scientific evidence.'}
    if state.get('evidence_access_mode') == 'files-v1':
        from tetra.runtime.inputs import digest, stable

        record = team['evidence'][evidence_id]
        state.setdefault('evidence_read_ids', [])
        if evidence_id not in state['evidence_read_ids']:
            state['evidence_read_ids'].append(evidence_id)
        state.setdefault('evidence_reads', []).append({
            'ids': [evidence_id], 'board_version': team['lead']['version'],
            'sha256': {evidence_id: digest(stable(record).encode())},
            'delivery': 'inline_search_result',
        })
    return payload


def first_condition_tools(state, names):
    """Choose the next retrieval stage for the first conditioned prediction."""
    if not state.get('conditioned_first') or state.get('prediction_count'):
        return names
    observations = state.get('observations', [])
    mapped = next((i for i, row in reversed(list(enumerate(observations)))
                   if row['action'] == 'read_structure_details'
                   and row['result'].get('status') == 'retrieved'
                   and any(draft.get('target_positions') for draft in row['result'].get('pocket_drafts', []))), None)
    if mapped is not None:
        failed_plan = any((row.get('attempted_action') or row['action']) == 'plan_prediction_experiment'
                          and 'error' in row.get('result', {}) for row in observations[mapped + 1:])
        if not failed_plan and any(
                name in names for name in ('plan_prediction_experiment', 'execute_prediction_experiment')):
            blocked = {'search_literature', 'find_reference_complexes', 'read_literature',
                       'read_structure_details', 'measure_reference_pocket'}
            return [name for name in names if name not in blocked]
    if 'measure_reference_pocket' in names:
        return [name for name in names if name in ('update_memory', 'measure_reference_pocket')]
    attempted = any(row['action'] == 'search_literature'
                    and (row.get('arguments') or {}).get('route') == 'sequence' for row in observations)
    if attempted:
        return [name for name in names if name != 'find_reference_complexes']
    if 'find_reference_complexes' in names:
        return [name for name in names if name not in ('search_literature', 'read_literature')]
    return names


def literature_progress(state):
    from tetra.research.literature import current_literature_round, pending_literature_sources, structural_followup

    observations = state.get('observations', [])
    current = current_literature_round(observations)
    details = [row['result'] for row in current if row['action'] == 'read_structure_details'
               and row.get('result', {}).get('status') == 'retrieved']
    draft_refs = [(key, row['observation']['result'])
                  for key, row in state.get('research', {}).get('evidence', {}).items()
                  if row['observation']['action'] == 'read_structure_details'
                  and row['observation']['result'] in details]
    blocker = structural_followup(observations)
    return {
        'pending_source_ids': pending_literature_sources(observations),
        'detailed_source_ids': [row.get('source_id') for row in details],
        'mapped_pockets': [dict(draft, source_id=row.get('source_id'), evidence_id=key)
                           for key, row in draft_refs for draft in row.get('pocket_drafts', [])][:8],
        'condition_goal': ('Register a first pocket experiment with a mapped site and cite its Evidence ID.'
                           if any(row.get('pocket_drafts') for row in details) else
                           'Obtain a mapped pocket from a comparable structure or usable source passage.'),
        'next_retrieval_action': blocker[0] if blocker else None,
        'prediction_blocker': blocker[1] if blocker else None,
    }


def literature_read_schema(state, name='read_literature'):
    schema = TOOLS[name].schema(name)
    observations = state['observations']
    if name == 'measure_reference_pocket':
        from tetra.research.literature import pending_reference_sources

        ids = pending_reference_sources(observations)
    elif name == 'read_structure_details':
        ids = dict.fromkeys(
            row['result']['source_id'] for row in observations
            if row['action'] == 'read_literature'
            and row['result'].get('status') == 'retrieved'
        )
    else:
        from tetra.research.literature import pending_literature_sources

        ids = pending_literature_sources(observations)
    if ids:
        schema['function']['parameters']['properties']['source_id'] = {
            'type': 'string', 'enum': list(ids),
        }
    return schema
