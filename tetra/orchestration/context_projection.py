"""Project archived evidence into a bounded wire view."""
from copy import deepcopy
import json
from typing import Any

from tetra.runtime.inputs import stable
from tetra.orchestration.workflow import compact

TABLE_TAG = 'residue-table-v1'


def project(value: Any) -> Any:
    """Replace repeated column names, never omit residue rows or round numbers."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == 'residues' and isinstance(item, list) and item and all(isinstance(row, dict) for row in item):
                columns = list(item[0])
                if columns and all(set(row) == set(columns) for row in item):
                    output[key] = {'format': TABLE_TAG, 'columns': columns,
                                   'rows': [[project(row[column]) for column in columns] for row in item]}
                    continue
            output[key] = project(item)
        return output
    if isinstance(value, list):
        return [project(item) for item in value]
    return value


def restore(value: Any) -> Any:
    """Reference decoder for verification of the documented table representation."""
    if isinstance(value, dict):
        if value.get('format') == TABLE_TAG and set(value) == {'format', 'columns', 'rows'}:
            return [{column: restore(item) for column, item in zip(value['columns'], row)}
                    for row in value['rows']]
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore(item) for item in value]
    return value


def comparison_coverage(state):
    """Report only hash-current measured pairs; a star is not all-pairs coverage."""
    from itertools import combinations
    candidates = state.get('candidates', {})
    expected = {tuple(sorted(pair)) for pair in combinations(candidates, 2)}
    pairs = {}
    for observation in state.get('observations', []):
        if observation.get('action') != 'compare_structures':
            continue
        result = observation.get('result', {})
        a, b = result.get('original_id'), result.get('candidate_id')
        if a not in candidates or b not in candidates or a == b:
            continue
        if (result.get('original_sha256') != candidates[a].get('sha256') or
                result.get('candidate_sha256') != candidates[b].get('sha256')):
            continue
        metric = result.get('ligand_pose_comparison', {})
        value = metric.get('symmetry_corrected_pose_rmsd_A')
        if metric.get('status') == 'measured' and isinstance(value, (int, float)):
            pairs[tuple(sorted((a, b)))] = value
    return {'expected_pair_count': len(expected), 'measured_pair_count': len(pairs),
            'missing_pairs': [list(pair) for pair in sorted(expected-set(pairs))],
            'complete': bool(expected) and set(pairs) == expected,
            'measured_pair_max_rmsd_A': max(pairs.values()) if pairs else None,
            'rule': 'Extrema describe measured pairs only. Do not claim all-pairs convergence from leading-vs-alternative comparisons. Sampling agreement does not prove accuracy.'}


def request_messages(state: dict, context: dict) -> list[dict]:
    """Construct an API view without editing durable messages or scientific state."""
    evidence = state['research']['evidence']
    lookup = {}
    for key, entry in evidence.items():
        result = compact(entry['observation']['result'])
        lookup.setdefault(stable(result), []).append(key)
    messages = deepcopy(state['messages'])
    if state.get('board_context_mode') == 'checkpoint-v1':
        checkpoint = None
        for index, message in enumerate(messages):
            if message.get('role') != 'tool':
                continue
            try:
                payload = json.loads(message.get('content', ''))
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get('memory_version') == state['research']['lead']['version']:
                if index and any(call.get('id') == message.get('tool_call_id') and
                                 call.get('function', {}).get('name') == 'update_memory'
                                 for call in messages[index-1].get('tool_calls', [])):
                    checkpoint = index
        if checkpoint is not None:
            messages = [messages[0], {'role': 'user', 'content': state['goal'] + '\nContinue from the canonical board and complete evidence map. Older dialogue is archived; hypotheses remain interpretations.'}, *messages[checkpoint+1:]]
    for message in messages:
        if message.get('role') != 'tool' or not isinstance(message.get('content'), str):
            continue
        try:
            payload = json.loads(message['content'])
            refs = lookup.get(stable(payload))
            if state.get('evidence_access_mode') == 'files-v1' and isinstance(payload, dict):
                summaries = payload.get('evidence_summaries')
                if isinstance(summaries, dict) and len(message['content'].encode('utf-8')) > 64_000:
                    pending = set(state['research'].get('unreviewed_evidence', []))
                    archived = [key for key in summaries if key in evidence and key not in pending]
                    if archived:
                        payload['evidence_summaries'] = {
                            key: value for key, value in summaries.items() if key not in archived}
                        payload['archived_evidence_ids'] = archived
                        payload['archive_note'] = (
                            'Older bundle entries are indexed here by ID. Read a specific ID '
                            'again if its contents are needed; pending evidence is included.')
                        refs = None
        except (ValueError, TypeError):
            continue
        message['content'] = json.dumps(
            {'evidence_refs': refs, 'location': 'current context evidence; full result retained there'}
            if refs else project(payload), ensure_ascii=False, separators=(',', ':'))
    view = deepcopy(context)
    view['candidate_comparison_coverage'] = comparison_coverage(state)
    from tetra.evaluation.revision_tools import scoring_readiness
    view['scoring_readiness'] = scoring_readiness(state) if state.get('candidates') else None
    view['verified_molecular_facts'] = {key: {'protein_sequence_length': len(c['sequence']) if isinstance(c.get('sequence'), str) else None} for key, c in state.get('candidates', {}).items()}
    view['fact_reporting_rule'] = 'Copy molecular lengths from verified_molecular_facts; never infer lengths from prose memory.'
    view['dialogue_context_mode'] = state.get('board_context_mode', 'full_archive_projection')
    scientific_state = view.get('scientific_state', {})
    recent = scientific_state.pop('recent_observations', [])
    scientific_state['recent_observations'] = [
        {'action': item['action'], 'evidence_refs': lookup[stable(compact(item['result']))]}
        if stable(compact(item['result'])) in lookup else item for item in recent]
    # Avoid a misleading empty duplicate observation section.
    if recent and all('evidence_refs' in item for item in scientific_state['recent_observations']):
        scientific_state['recent_evidence_refs'] = scientific_state.pop('recent_observations')
    view['wire_format'] = (
        'residue-table-v1 is lossless: every row maps positionally to columns. '
        'All residue rows and numerical precision are retained, including distant residues. '
        'Historical tool evidence_refs resolve to the evidence map in this current context. '
        'The same evidence may be referenced repeatedly; references are not new measurements.'
    )
    messages.append({'role': 'user', 'content': json.dumps(project(view), ensure_ascii=False, separators=(',', ':'))})
    if state.get('evidence_access_mode') == 'files-v1':
        compact_evidence_history(messages, evidence, state['research'].get('unreviewed_evidence', []))
    return messages


def compact_evidence_history(messages, evidence, pending_ids):
    """Reduce oversized tool history while keeping one detailed copy of pending evidence."""
    from tetra.runtime.cost_budget import MAX_REQUEST_BYTES

    # Leave room for tool schemas and API framing. The transport still checks the exact size.
    budget = MAX_REQUEST_BYTES - 64_000
    if len(json.dumps(messages).encode()) <= budget:
        return
    pending = set(pending_ids)
    seen = set()
    for message in reversed(messages):
        if message.get('role') != 'tool':
            continue
        try:
            payload = json.loads(message['content'])
        except (KeyError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for field in ('evidence', 'evidence_summaries'):
            records = payload.get(field)
            if not isinstance(records, dict) or not records or not records.keys() <= evidence.keys():
                continue
            retained = {key: row for key, row in records.items() if key in pending and key not in seen}
            seen.update(retained)
            payload[field] = retained
            payload['archived_evidence_ids'] = list(dict.fromkeys([
                *payload.get('archived_evidence_ids', []),
                *(key for key in records if key not in retained),
            ]))
            payload['archive_note'] = (
                'Older or duplicate detail is archived. Request a specific evidence ID if needed.')
            message['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
