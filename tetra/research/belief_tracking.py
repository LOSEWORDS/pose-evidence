"""Exact measurement bindings and revisable interpretations, not an entailment oracle.

Design references: SHELDON State.observe/apply_exploit, FOREST validate_claims
and graph dependency invalidation, POPPER's implementation-status routing.
This is a local implementation; no upstream code is vendored.
"""
from copy import deepcopy
import math
import re
from difflib import get_close_matches

from tetra.runtime.inputs import digest, stable

VERSION = 'source-bound-beliefs-v1'
STATUSES = ('untested', 'supported', 'contradicted', 'unresolved', 'weakened')
BASES = ('new_evidence', 'reinterpretation', 'unchanged')
MEASUREMENTS = {'assess_structure', 'measure_contacts', 'compare_structures', 'score_candidates',
                'inspect_contact_chemistry', 'measure_atom_geometry'}
CATALOG_ACTIONS = MEASUREMENTS | {'resolve_input', 'prediction_completed'}
POLICY = '''Source-bound scientific memory:
Use the measurement_facts paths returned by read_evidence to cite exact observations.
Values and candidate versions are filled by the executor; never invent fact values.
Working memory is interpretation, not the authoritative measurement table.
Give each scientific hypothesis a stable claim_id. Preserve its statement; a changed
statement gets a new claim_id with parent_claim_id. Split measured observations from
mechanistic explanations. Each update states its basis, rationale, evidence boundary
and decision impact. Weakened is not contradicted; unresolved is not false.
For tools without a measurement_facts adapter, cite the original successful source
with fact_refs=[]; this is explicitly source-only validation, not a measured fact table.
new_evidence requires a new measurement/source relative to this claim. Re-reading or
renaming a record is not new evidence. reinterpretation permits correcting an old
judgment from old evidence. unchanged must preserve status and bindings.
Source checks validate references, NOT scientific support. A supported interpretation
is still model-generated, not an independently verified mechanism. Measurements from
the same structure share a source; different tools do not establish independent biology.
For an unresolved claim affecting the request, explain what available observation could
distinguish the remaining explanations and how outcomes would change the next decision.
If none is feasible or worthwhile, retain the unknown and explain stopping with limits.
Do not prescribe a fixed tool sequence, a minimum question count or mandatory repair.
Do not erase scientific interpretations just to pass protocol checks. Empty hypotheses
and lessons are allowed for factual/workflow updates, but mean no structured scientific
judgment was recorded. Free-text working_memory is not a substitute for claim updates.
For scientific lessons the same claim_id and update fields are required as hypotheses.
Copy a listed measurement_facts path and its own E-ID; confidence belongs to prediction
evidence, not structural-assessment evidence. /result is not a measurement path.
Execution failure supplies no measurement; successful execution alone does not validate
the hypothesis. Reassess claims whose source bindings are stale before relying on them.
'''


def enabled(team: dict) -> bool:
    return team.get('belief_tracking_version') == VERSION


def source_hash(record: dict) -> str:
    return digest(stable(record).encode())


def fact_catalog(record: dict) -> dict:
    """Project selected measured fields; never promote result prose or tool arguments."""
    observation = record['observation']
    result = observation.get('result', {})
    action = observation.get('action')
    if action == 'resolve_input' and result.get('status') != 'resolved':
        return {}
    if (action not in CATALOG_ACTIONS or not isinstance(result, dict)
            or result.get('error') or result.get('status') in ('failed', 'error', 'unavailable')):
        return {}
    facts = {}

    def add(path, value):
        if isinstance(value, dict):
            for key, item in value.items():
                add(path + '/' + str(key).replace('~', '~0').replace('/', '~1'), item)
        elif isinstance(value, list):
            if all(type(item) in (int, float, str, bool) for item in value):
                if all(not isinstance(item, float) or math.isfinite(item) for item in value):
                    facts[path] = deepcopy(value)
            else:
                for index, item in enumerate(value):
                    add(path + '/' + str(index), item)
        elif type(value) in (str, int, float, bool):
            if not isinstance(value, float) or math.isfinite(value):
                facts[path] = value

    # Explicit roots keep model-written rationales and policy acceptance out of facts.
    roots = {
        'assess_structure': ('raw_metrics', 'checks'),
        'measure_contacts': ('minimum_distance_A', 'cutoff_A', 'contact_positions', 'residue_mapping'),
        'inspect_contact_chemistry': ('contacts', 'ligand_groups', 'group_presence', 'protein_protonation_status'),
        'measure_atom_geometry': ('atom_ids', 'geometry', 'value', 'unit'),
        'compare_structures': ('protein_sequence_length', 'protein_CA_aligned_change_rms_A',
                              'maximum_CA_displacement_A', 'lost_contact_positions',
                              'gained_contact_positions', 'retained_contact_fraction',
                              'minimum_distance_before_A', 'minimum_distance_after_A'),
        'score_candidates': (),
        'resolve_input': ('protein_length',),
        'prediction_completed': (),
    }
    for key in roots[action]:
        if key in result:
            add('/result/' + key, result[key])
    if action == 'resolve_input' and result.get('status') == 'resolved':
        request = result.get('provenance', {}).get('request', {})
        for key in ('sequence', 'smiles'):
            if isinstance(request.get(key), str):
                add('/result/provenance/request/' + key, request[key])
    if action == 'prediction_completed':
        for index, row in enumerate(result.get('candidate_metadata', [])):
            confidence = row.get('original_confidence')
            if not isinstance(confidence, dict):
                continue
            prefix = f'/result/candidate_metadata/{index}/original_confidence'
            add(prefix, confidence)
            # An explicitly scoped numeric confidence block is safe to bind as a group.
            # Do not expose arbitrary result objects, rationales or provenance as facts.
            def numeric_tree(value):
                if isinstance(value, dict):
                    return bool(value) and all(numeric_tree(v) for v in value.values())
                return type(value) in (float, int) and math.isfinite(value)
            if numeric_tree(confidence):
                facts[prefix] = deepcopy(confidence)
    if action == 'score_candidates':
        for cid, row in result.get('scores', {}).items():
            if not isinstance(row, dict) or cid in result.get('errors', {}) or row.get('error'):
                continue
            for key in ('cnn_score', 'cnn_affinity', 'minimized_affinity'):
                if key in row:
                    escaped = cid.replace('~', '~0').replace('/', '~1')
                    add(f'/result/scores/{escaped}/{key}', row[key])
    if action == 'measure_contacts':
        positions = result.get('contact_positions')
        if isinstance(positions, list) and all(type(p) is int for p in positions):
            facts['/derived/contact_count'] = len(set(positions))
        for index, row in enumerate(result.get('residues', [])):
            if not isinstance(row, dict) or row.get('input_position') not in (positions or []):
                continue
            for key in ('input_position', 'chain', 'residue_name', 'pdb_number',
                        'pdb_insertion_code', 'minimum_distance_A', 'atom_pairs_within_cutoff'):
                if key in row:
                    add(f'/result/residues/{index}/{key}', row[key])
    return facts


def bind_fact(team: dict, ref: dict, evidence_ids: list) -> dict:
    if isinstance(ref, str):
        from tetra.research.fact_references import resolve
        resolved = resolve(team, ref)
        bound = bind_fact(team, resolved, list(dict.fromkeys([*evidence_ids, resolved['evidence_id']])))
        bound['fact_id'] = ref
        return bound
    if not isinstance(ref, dict) or set(ref) != {'evidence_id', 'path'}:
        raise ValueError('fact_refs contain only evidence_id and path; values are executor-derived.')
    eid, path = ref['evidence_id'], canonical_path(ref['path'])
    if not isinstance(eid, str) or eid not in evidence_ids or eid not in team['evidence']:
        raise ValueError('Fact reference must cite evidence_ids on this interpretation.')
    record = team['evidence'][eid]
    catalog = fact_catalog(record)
    if not isinstance(path, str) or path not in catalog:
        raise ValueError(f'{eid}: unknown measurement path {path!r}; read measurement_facts. Failed/workflow records do not supply measurement facts.')
    result = record['observation']['result']
    metadata = source_binding(team, eid)
    if record['observation'].get('action') == 'prediction_completed':
        row = result['candidate_metadata'][int(path.split('/')[3])]
        metadata['candidate_id'] = row.get('candidate_id')
        metadata['candidate_sha256'] = row.get('sha256')
    if record['observation'].get('action') == 'score_candidates':
        cid = path.split('/')[3].replace('~1', '/').replace('~0', '~')
        row = result['scores'][cid]
        metadata['candidate_id'] = row.get('candidate_id')
        metadata['candidate_sha256'] = row.get('candidate_sha256')
        metadata['measurement_parameters'] = {'profile': row.get('profile')}
    return {**metadata, 'path': path, 'value': deepcopy(catalog[path]),
            'source_sha256': source_hash(record), 'action': record['observation']['action'],
           }


def canonical_path(path):
    """Accept an exact alternate notation, never guess fields or change the E-ID."""
    if not isinstance(path, str) or path.startswith('/'):
        return path
    if not re.fullmatch(r'observation\.result(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*', path):
        return path
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+', path)
    return '/' + '/'.join(tokens[1:])


def source_binding(team: dict, eid: str) -> dict:
    record = team['evidence'][eid]
    result = record['observation'].get('result', {})
    dependencies = {}

    def collect(value):
        if isinstance(value, dict):
            for id_key, hash_key in (('candidate_id', 'candidate_sha256'),
                                     ('candidate_id', 'sha256'), ('original_id', 'original_sha256'),
                                     ('parent_id', 'parent_sha256')):
                if value.get(id_key) and value.get(hash_key):
                    dependencies[value[id_key]] = value[hash_key]
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(result)
    return {'evidence_id': eid, 'source_sha256': source_hash(record),
            'measurement_sha256': digest(stable([record['observation'].get('action'), result]).encode()),
            'candidate_id': result.get('candidate_id'), 'candidate_sha256': result.get('candidate_sha256'),
            'candidate_dependencies': dependencies,
            'measurement_parameters': {key: deepcopy(result[key]) for key in ('cutoff_A', 'profile', 'residue_positions', 'atom_ids')
                                       if key in result}}


def usable_source(record: dict) -> bool:
    from tetra.research.evidence_semantics import record_type
    result = record['observation'].get('result', {})
    if record['observation'].get('action') == 'resolve_input' and result.get('status') != 'resolved':
        return False
    if record['observation'].get('action') in MEASUREMENTS and not fact_catalog(record):
        return False
    return (record_type(record['observation']) in ('computational_observation', 'input_record')
            and not result.get('error') and result.get('status') not in ('failed', 'error', 'unavailable')
            and record['observation'].get('action') != 'evaluate_revision')


def measurement_key(fact: dict) -> str:
    """Same source/action/field/value is not independent evidence under a new E-ID."""
    dependencies = fact.get('candidate_dependencies') or fact['measurement_sha256']
    return stable([dependencies, fact.get('measurement_parameters', {}), fact['action'],
                   fact['path'], fact['value']])


class NoNewMeasurement(ValueError):
    def __init__(self, claim_id: str, can_keep_unchanged: bool):
        super().__init__(f'{claim_id}: no new measurement; use reinterpretation or unchanged, not new_evidence.')
        self.claim_id = claim_id
        self.allowed_bases = ['reinterpretation']
        if can_keep_unchanged:
            self.allowed_bases.append('unchanged')


def prepare_commit(team: dict, hypotheses: list, lessons: list, *,
                   withdraw_claim_ids: list[str] | None = None) -> tuple[dict, list]:
    """Validate everything and return a new snapshot without mutating live memory."""
    old = team['lead'].get('belief_state', {}).get('claims', {})
    claims = deepcopy(old)
    events = []
    seen = set()
    version = team['lead']['version'] + 1
    for item in [*hypotheses, *lessons]:
        if item.get('support_scope', 'scientific') != 'scientific':
            continue
        cid = item.get('claim_id')
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', cid) or cid in seen:
            raise ValueError('Scientific interpretations need unique stable claim_id values.')
        seen.add(cid)
        previous = old.get(cid)
        if previous and previous['claim'] != item['claim']:
            raise ValueError(f'{cid}: statement changed; use a new claim_id and parent_claim_id.')
        parent = item.get('parent_claim_id')
        if parent is not None and (parent not in old or parent == cid):
            raise ValueError(f'{cid}: parent_claim_id must identify an existing different claim.')
        if previous and parent != previous.get('parent_claim_id'):
            raise ValueError(f'{cid}: parent lineage cannot be rewritten.')
        update = item.get('update')
        fields = {'basis', 'rationale', 'boundary', 'decision_impact', 'fact_refs'}
        if not isinstance(update, dict) or set(update) != fields or update.get('basis') not in BASES:
            raise ValueError(f'{cid}: update needs basis, rationale, boundary, decision_impact, fact_refs.')
        if any(not isinstance(update[k], str) or not update[k].strip()
               for k in ('rationale', 'boundary', 'decision_impact')):
            raise ValueError(f'{cid}: explain the update, evidence boundary and decision impact.')
        if not isinstance(update['fact_refs'], list):
            raise ValueError(f'{cid}: fact_refs must be a list.')
        facts = [bind_fact(team, ref, item['evidence_ids']) for ref in update['fact_refs']]
        sources = [source_binding(team, eid) for eid in item['evidence_ids']
                   if usable_source(team['evidence'][eid])]
        has_catalog = any(fact_catalog(team['evidence'][eid]) for eid in item['evidence_ids'])
        if len({(f['evidence_id'], f['path']) for f in facts}) != len(facts):
            raise ValueError(f'{cid}: duplicate fact bindings.')
        status = item.get('status', 'supported')  # A scientific lesson asserts support.
        if status not in STATUSES:
            raise ValueError(f'{cid}: invalid belief status.')
        if status in ('supported', 'contradicted', 'weakened') and (not sources or has_catalog and not facts):
            raise ValueError(f'{cid}: this status requires a bound measurement; otherwise retain unresolved.')
        prior = previous or old.get(parent, {})
        old_keys = {measurement_key(f) for f in prior.get('facts', [])}
        new_keys = {measurement_key(f) for f in facts}
        old_keys.update(s['measurement_sha256'] for s in prior.get('sources', [])
                        if s.get('binding_level') == 'source_only')
        for source in sources:
            source['binding_level'] = ('measurement_paths' if fact_catalog(team['evidence'][source['evidence_id']])
                                       else 'source_only')
            if source['binding_level'] == 'source_only':
                new_keys.add(source['measurement_sha256'])
        has_new = bool(new_keys - old_keys)
        can_keep_unchanged = previous is not None and previous['status'] == status and new_keys == old_keys
        if update['basis'] == 'new_evidence' and not has_new:
            raise NoNewMeasurement(cid, can_keep_unchanged)
        if update['basis'] == 'unchanged' and not can_keep_unchanged:
            raise ValueError(f'{cid}: unchanged cannot change status or measurements.')
        claim = {'claim_id': cid, 'claim': item['claim'], 'status': status,
                 'parent_claim_id': parent, 'active': True, 'facts': facts, 'sources': sources,
                 'update': deepcopy(update), 'memory_version': version,
                 'interpretation_is_model_generated': True,
                 'independent_biological_confirmation': False,
                 'validation_scope': 'source_binding_only' if facts else 'source_references_only'}
        claims[cid] = claim
        events.append({'claim_id': cid, 'memory_version': version,
                       'operation': 'revise' if previous else 'create',
                       'previous_status': previous.get('status') if previous else None,
                       'status': status, 'basis': update['basis'], 'has_new_measurement': has_new,
                       'parent_claim_id': parent, 'update': deepcopy(update),
                       'facts': deepcopy(facts), 'sources': deepcopy(sources)})
    withdrawals = None
    if withdraw_claim_ids is not None:
        if (not isinstance(withdraw_claim_ids, list)
                or any(not isinstance(cid, str) for cid in withdraw_claim_ids)):
            raise ValueError('withdraw_claim_ids must be a list of active claim IDs.')
        withdrawals = set(withdraw_claim_ids)
        if len(withdrawals) != len(withdraw_claim_ids) or any(
                cid not in old or not old[cid]['active'] or cid in seen for cid in withdrawals):
            raise ValueError('Withdraw distinct active claims that are not being updated this round.')
    for cid, claim in claims.items():
        withdraw = cid not in seen if withdrawals is None else cid in withdrawals
        if withdraw and claim['active']:
            claim['active'] = False
            events.append({'claim_id': cid, 'memory_version': version, 'operation': 'withdraw',
                           'previous_status': claim['status'], 'status': claim['status'],
                           'reason': ('Explicit withdrawal; history retained, not refuted.' if withdrawals is not None
                                  else 'Omitted from current interpretation set; history retained, not refuted.')})
    return {'version': VERSION, 'claims': claims}, events


def freshness(state: dict, fact: dict) -> str:
    record = state['research']['evidence'].get(fact['evidence_id'])
    if record is None or source_hash(record) != fact['source_sha256']:
        return 'stale_evidence'
    dependencies = fact.get('candidate_dependencies', {})
    for cid, expected in dependencies.items():
        candidate = state.get('candidates', {}).get(cid)
        if not candidate or candidate.get('sha256') != expected:
            return 'stale_candidate'
    return 'current' if dependencies else 'source_current_candidate_unbound'


def memory_view(state: dict) -> dict:
    """Freshness belongs to the read view; archived memory versions stay immutable."""
    lead = deepcopy({k: v for k, v in state['research']['lead'].items() if k != 'history'})
    if enabled(state['research']):
        active = [c for c in lead.get('belief_state', {}).get('claims', {}).values() if c['active']]
        lead['scientific_cognition'] = {
            'status': 'model_claims_recorded' if active else 'no_structured_claims',
            'active_claim_count': len(active), 'semantic_validation_performed': False,
            'boundary': 'A successful memory write is not a validated scientific update. Free text is model interpretation.'}
    for claim in lead.get('belief_state', {}).get('claims', {}).values():
        for fact in [*claim['facts'], *claim.get('sources', [])]:
            fact['freshness'] = freshness(state, fact)
        claim['applicability'] = ('needs_review' if any(f['freshness'].startswith('stale')
                                                     for f in [*claim['facts'], *claim.get('sources', [])]) else 'source_bound')
    return lead


def validate_current_bindings(state: dict, hypotheses: list, lessons: list) -> None:
    if not enabled(state['research']):
        return
    for item in [*hypotheses, *lessons]:
        for eid in item.get('evidence_ids', []):
            if usable_source(state['research']['evidence'][eid]):
                if freshness(state, source_binding(state['research'], eid)).startswith('stale'):
                    raise ValueError('Cannot commit stale source bindings; remeasure or withdraw the current claim.')
        for ref in item.get('update', {}).get('fact_refs', []):
            fact = bind_fact(state['research'], ref, item.get('evidence_ids', []))
            if freshness(state, fact).startswith('stale'):
                raise ValueError('Cannot commit stale measurement bindings; remeasure the current candidate or withdraw the current claim.')


def require_current_claims(state: dict, action: str) -> None:
    """Delivery must not silently consume stale claims; diagnostic work stays open."""
    if not enabled(state['research']) or action not in ('select_candidates', 'audit_delivery', 'finish'):
        return
    claims = state['research']['lead'].get('belief_state', {}).get('claims', {})
    stale = [cid for cid, claim in claims.items() if claim['active'] and any(
        freshness(state, fact).startswith('stale')
        for fact in [*claim['facts'], *claim.get('sources', [])])]
    if stale:
        raise ValueError('Review or withdraw stale claims before delivery: ' + ', '.join(stale))


def lesson_scope_issue(item: dict) -> str | None:
    """Catch a classification value written in the lesson's applicability field."""
    if (isinstance(item, dict) and 'support_scope' not in item
            and item.get('scope') in ('workflow', 'execution', 'scientific')):
        return ('Required explicitly when scope contains a classification. Set support_scope '
                'to scientific, workflow, or execution as appropriate. Lesson scope describes '
                'where the lesson applies. Workflow and execution records cannot support scientific claims.')
    return None


def extend_schema(parameters: dict) -> None:
    text = {'type': 'string'}
    update = {'type': 'object', 'additionalProperties': False, 'properties': {
        'basis': {'type': 'string', 'enum': list(BASES)}, 'rationale': text,
        'boundary': text, 'decision_impact': text,
        'fact_refs': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                     'properties': {'evidence_id': text, 'path': text}, 'required': ['evidence_id', 'path']}}},
        'required': ['basis', 'rationale', 'boundary', 'decision_impact', 'fact_refs']}
    for name in ('hypotheses', 'lessons'):
        item = parameters['properties'][name]['items']
        item['properties'].update(claim_id=text, parent_claim_id=text, update=deepcopy(update))
        # JSON Schema and runtime agree, including the default scientific scope.
        item['allOf'] = [{'if': {'properties': {'support_scope': {'enum': ['scientific']}}},
                          'then': {'required': ['claim_id', 'update']}}]
        if name == 'hypotheses':
            item['properties']['status']['enum'] = list(STATUSES)
        else:
            item['properties']['scope']['description'] = (
                'Applicability of the lesson, for example this prediction run. '
                'Use support_scope to classify scientific, workflow or execution records.')
        item['description'] = 'Scientific interpretations require claim_id and update. Cite measurement_facts paths; executor fills exact values. Source validation is not scientific entailment.'


def repair_feedback(state: dict, arguments: dict, rejection: Exception | None = None) -> dict:
    """Report independent interface defects together; never auto-approve a claim."""
    if not enabled(state['research']) or not isinstance(arguments, dict):
        return {}
    team = state['research']
    committed_claims = team['lead'].get('belief_state', {}).get('claims', {})
    catalogs = {eid: fact_catalog(record) for eid, record in team['evidence'].items()}
    issues = []
    for collection in ('hypotheses', 'lessons'):
        items = arguments.get(collection, [])
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            location = f'{collection}[{index}]'
            scope_error = lesson_scope_issue(item) if collection == 'lessons' else None
            if scope_error:
                issues.append({'field': location + '.support_scope', 'error': scope_error})
                continue
            if not isinstance(item, dict) or item.get('support_scope', 'scientific') != 'scientific':
                continue
            claim_id = item.get('claim_id')
            if isinstance(rejection, NoNewMeasurement) and rejection.claim_id == claim_id:
                issues.append({
                    'field': location + '.update.basis',
                    'error': 'This claim does not cite a new measurement.',
                    'claim_id': claim_id,
                    'allowed_bases': rejection.allowed_bases,
                    'repair': ('Use unchanged if the status and measurements stay the same. '
                               'Use reinterpretation only for a justified change in interpretation. '
                               'Apply new_evidence only to claims that actually cite a new measurement.'),
                })
            if collection == 'lessons' and 'support_scope' not in item:
                from tetra.research.evidence_semantics import record_type
                refs = item.get('evidence_ids')
                if isinstance(refs, list) and refs and all(
                    isinstance(eid, str) and eid in team['evidence']
                    and record_type(team['evidence'][eid]['observation']) in ('execution_record', 'workflow_record')
                    for eid in refs
                ):
                    issues.append({
                        'field': location + '.support_scope',
                        'error': ('The cited sources are only execution or workflow records. If the lesson describes '
                                  'those events, explicitly classify it as execution or workflow and keep the claim '
                                  'within what those records show. Otherwise cite applicable scientific evidence.'),
                    })
                    continue
            previous = committed_claims.get(claim_id) if isinstance(claim_id, str) else None
            if previous and isinstance(item.get('claim'), str) and item['claim'] != previous['claim']:
                issues.append({
                    'field': location + '.claim',
                    'error': 'This claim_id already identifies a different statement.',
                    'claim_id': claim_id,
                    'committed_claim': previous['claim'],
                    'committed_parent_claim_id': previous.get('parent_claim_id'),
                    'new_claim_parent_id': claim_id,
                    'repair': ('For the same proposition, copy committed_claim and its existing lineage '
                               'exactly, then update its status or evidence assessment as warranted. '
                               'For a different proposition, submit a new claim_id with parent_claim_id '
                               'set to new_claim_parent_id. Preserve the prior claim if still relevant.'),
                })
            for field in ('claim_id', 'update'):
                if field not in item:
                    issues.append({'field': location + '.' + field, 'error': 'Required for scientific interpretations, including lessons.'})
            update = item.get('update', {})
            refs = update.get('fact_refs', []) if isinstance(update, dict) else []
            for offset, ref in enumerate(refs if isinstance(refs, list) else []):
                try:
                    bind_fact(team, ref, item.get('evidence_ids', []))
                except (ValueError, TypeError, KeyError) as error:
                    issue = {'field': f'{location}.update.fact_refs[{offset}]', 'error': str(error)}
                    if isinstance(ref, dict):
                        eid, path = ref.get('evidence_id'), canonical_path(ref.get('path'))
                        if isinstance(path, str):
                            leaf = path.rsplit('/', 1)[-1]
                            suggestions = [{'evidence_id': key, 'path': p} for key, catalog in catalogs.items()
                                           for p in catalog if p.rsplit('/', 1)[-1] == leaf]
                            if not suggestions and isinstance(eid, str):
                                suggestions = [{'evidence_id': eid, 'path': p} for p in
                                               get_close_matches(path, list(catalogs.get(eid, {})), n=3, cutoff=0.4)]
                            issue['available_references_to_inspect'] = suggestions[:4]
                    issues.append(issue)
    return {'kind': 'protocol_repair_not_scientific_evidence', 'issues': issues[:12],
            'omitted_issue_count': max(0, len(issues) - 12),
            'instruction': 'Fix all listed interface issues. Inspect suggested sources before citing; suggestions do not support your claim. Do not erase hypotheses merely to pass validation.'}
