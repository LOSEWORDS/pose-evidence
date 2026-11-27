"""Immutable evidence files and separate versioned Lead memory, read by evidence ID."""
import json
from pathlib import Path
import re
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest, stable
from tetra.research.evidence_semantics import evidence_view, record_type


def persist(state, folder):
    folder = Path(folder)
    evidence_dir = folder / 'evidence'
    memory_dir = folder / 'memory'
    evidence_dir.mkdir(exist_ok=True)
    memory_dir.mkdir(exist_ok=True)
    index = {}
    for key, record in state['research']['evidence'].items():
        if not re.fullmatch(r'E[1-9][0-9]*', key):
            raise ValueError('Invalid evidence identifier.')
        data = stable(record).encode()
        path = evidence_dir / (key + '.json')
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Immutable evidence file changed: ' + key)
        else:
            path.write_bytes(data)
        observation = record['observation']
        result = observation.get('result', {})
        index[key] = {'id': key, 'action': observation['action'], 'sha256': digest(data),
                      'bytes': len(data), 'path': str(path.resolve()), 'origin': record['origin'],
                      'epistemic_type': record_type(observation),
                      'candidate_id': result.get('candidate_id'), 'status': result.get('status'),
                      'has_error': 'error' in result}
    lead = state['research']['lead']
    for board in [*lead['history'], lead]:
        snapshot = {k: v for k, v in board.items() if k not in ('history', 'decision_consumed')}
        path = memory_dir / f"version-{board['version']:04d}.json"
        if path.exists() and json.loads(path.read_text()) != snapshot:
            raise ValueError('Immutable memory version changed.')
        if not path.exists():
            atomic_json(path, snapshot)
    atomic_json(memory_dir / 'current.json', {k: v for k, v in lead.items() if k != 'history'})
    atomic_json(evidence_dir / 'index.json', index)
    state['evidence_index'] = index
    return index


class EvidenceReadTooLarge(ValueError):
    pass


def read_evidence(state, folder, evidence_ids, *, max_bytes=180000):
    if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 4 or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError('Read one to four distinct evidence IDs at a time.')
    index = persist(state, folder)
    output = {}
    for key in evidence_ids:
        if not isinstance(key, str) or key not in index:
            raise ValueError('Unknown evidence ID; arbitrary filesystem paths are not accepted.')
        data = (Path(folder) / 'evidence' / (key + '.json')).read_bytes()
        if digest(data) != index[key]['sha256']:
            raise ValueError('Evidence file hash changed.')
        output[key] = evidence_view(json.loads(data))
    if max_bytes is not None and len(stable(output).encode()) > max_bytes:
        raise EvidenceReadTooLarge('Full evidence exceeds the bounded read size. Use read_evidence_bundle for summaries or request one ID.')
    state.setdefault('evidence_read_ids', [])
    state['evidence_read_ids'] = list(dict.fromkeys([*state['evidence_read_ids'], *evidence_ids]))
    state.setdefault('evidence_reads', []).append({'ids': evidence_ids,
        'board_version': state['research']['lead']['version'], 'sha256': {k: index[k]['sha256'] for k in evidence_ids}})
    return {'evidence': output, 'source': 'source-separated view of immutable evidence files'}


def inline_batch_evidence(state, prior_ids):
    """Deliver newly produced measurements while leaving their interpretation pending."""
    team = state['research']
    records = {key: record for key, record in team['evidence'].items() if key not in prior_ids}
    if not records:
        return None
    views = {key: evidence_view(record) for key, record in records.items()}
    for view in views.values():
        view.pop('model_interpretations', None)
    if len(stable(views).encode()) > 180000:
        return None
    ids = list(records)
    state['evidence_read_ids'] = list(dict.fromkeys([*state.get('evidence_read_ids', []), *ids]))
    state.setdefault('evidence_reads', []).append({
        'ids': ids, 'board_version': team['lead']['version'],
        'sha256': {key: digest(stable(record).encode()) for key, record in records.items()},
        'delivery': 'inline_batch_analysis',
    })
    return {'evidence': views, 'instruction': 'Review these measurements together in update_memory.'}


def require_reads(state, evidence_ids):
    if state.get('evidence_access_mode') != 'files-v1':
        return
    missing = set(evidence_ids) - set(state.get('evidence_read_ids', []))
    if missing:
        raise ValueError('Use read_evidence before citing unread evidence: ' + ', '.join(sorted(missing)))
