"""Selectable immutable fact IDs; source paths and citation lists are host-derived."""
from copy import deepcopy
import re

from tetra.runtime.inputs import digest, stable

VERSION = 'fact-ids-v1'
POLICY = '''Fact reference interface:
In update.fact_refs, select the exact string IDs listed in selectable_facts, for example
an E-ID followed by :F and a digest. Do not type JSON paths or reference objects.
The executor fills the original Evidence ID, canonical path, exact value and provenance,
and adds selected sources to the interpretation, memory and decision evidence_ids.
Keep any additional relevant Evidence IDs (including source-only and workflow records).
Selecting a fact does not establish that it supports a mechanism or stopping decision.
Use read_evidence or read_evidence_bundle to inspect sources before citing their facts.
'''


def enabled(team: dict) -> bool:
    return team.get('fact_reference_version') == VERSION


def choices(record: dict) -> dict:
    from tetra.research.belief_tracking import fact_catalog, source_hash
    eid = record.get('id')
    if not isinstance(eid, str) or not re.fullmatch(r'E[1-9][0-9]*', eid):
        return {}
    original_hash = record.get('measurement_source_sha256') or source_hash(record)
    result = {}
    for path, value in fact_catalog(record).items():
        identifier = eid + ':F' + digest(stable([original_hash, path]).encode())[:10]
        if identifier in result:
            raise ValueError('Fact ID collision; refusing ambiguous references.')
        result[identifier] = {'evidence_id': eid, 'path': path, 'value': deepcopy(value)}
    return result


def resolve(team: dict, identifier: str) -> dict:
    if not isinstance(identifier, str) or not re.fullmatch(r'E[1-9][0-9]*:F[0-9a-f]{10}', identifier):
        raise ValueError('Copy an exact fact ID from selectable_facts; do not invent IDs or type paths.')
    eid = identifier.split(':', 1)[0]
    record = team['evidence'].get(eid)
    choice = choices(record).get(identifier) if record is not None else None
    if choice is None:
        raise ValueError(f'Unknown or stale fact ID {identifier!r}; re-read {eid} and select a listed ID.')
    return {'evidence_id': eid, 'path': choice['path']}


def normalize_arguments(team: dict, arguments: dict) -> dict:
    """Infer redundant citation lists only; never change a scientific interpretation."""
    output = deepcopy(arguments)
    if not enabled(team):
        return output
    inferred = []
    for collection in ('hypotheses', 'lessons'):
        for item in output.get(collection, []):
            for identifier in item.get('update', {}).get('fact_refs', []):
                ref = resolve(team, identifier)
                eid = ref['evidence_id']
                item['evidence_ids'] = list(dict.fromkeys([*item.get('evidence_ids', []), eid]))
                inferred.append(eid)
    output['evidence_ids'] = list(dict.fromkeys([*output.get('evidence_ids', []), *inferred]))
    if isinstance(output.get('decision'), dict):
        output['decision']['evidence_ids'] = list(dict.fromkeys([
            *output['decision'].get('evidence_ids', []), *inferred]))
    return output


def extend_schema(parameters: dict, state: dict) -> None:
    team = state['research']
    visible = (state.get('evidence_read_ids', []) if state.get('evidence_access_mode') == 'files-v1'
               else team['evidence'])
    identifiers = sorted(identifier for eid in visible if eid in team['evidence']
                         for identifier in choices(team['evidence'][eid]))
    for collection in ('hypotheses', 'lessons'):
        item = parameters['properties'][collection]['items']
        refs = item['properties']['update']['properties']['fact_refs']
        refs['items'] = {'type': 'string'}
        if identifiers:
            refs['items']['enum'] = identifiers
        else:
            refs['maxItems'] = 0
        refs['description'] = 'Select exact IDs from selectable_facts. No path objects or values. Source Evidence IDs are filled by the host; scientific support remains your judgment.'
        item['properties']['evidence_ids']['description'] = 'Additional source IDs. Selected fact sources are added automatically; keep other relevant sources.'
        item['description'] = 'Scientific interpretations require claim_id and update. Select fact IDs; do not hand-write paths. Source binding does not validate the scientific inference.'
