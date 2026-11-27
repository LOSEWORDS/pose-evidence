"""Validate the measurement scope declared by a qualified-delivery decision."""

from itertools import combinations
from math import isfinite

KINDS = ('pose_comparison', 'contact_localization', 'contact_chemistry', 'atom_geometry', 'observed_result')
ATOM_PREMISES = {'contact_chemistry': 'inspect_contact_chemistry', 'atom_geometry': 'measure_atom_geometry'}


def defer_basis_schema():
    text = {'type': 'string'}
    premise = {
        'type': 'object', 'additionalProperties': False,
        'properties': {'kind': {'type': 'string', 'enum': list(KINDS)},
                       'candidate_ids': {'type': 'array', 'items': text},
                       'evidence_ids': {'type': 'array', 'items': text},
                       'atom_ids': {'type': 'array', 'items': text, 'minItems': 2, 'maxItems': 4,
                                    'description': ('Required for contact_chemistry and atom_geometry only. '
                                                    'Use either order for a contact pair. Preserve the measured order for atom_geometry.')}},
        'required': ['kind', 'candidate_ids', 'evidence_ids'],
    }
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'kind': {'type': 'string', 'enum': ['measured', 'request_scope']},
            'premises': {'type': 'array', 'items': premise},
            'request_quote': text,
        },
        'required': ['kind', 'premises'],
        'description': (
            'Required when disposition=defer. Declare every measurement premise used to skip exploration. '
            'pose_comparison covers all pairs among its named IDs; use separate rows for individual pairs. '
            'contact_localization requires a measurement for every named candidate; it does not establish an '
            'identical atom pair or a mechanism. Use contact_chemistry for atom identity or ligand group '
            'premises and atom_geometry for an explicitly measured distance, angle or dihedral. Cite '
            'the matching candidate, atom IDs and actual measurement scope. These observations do '
            'not establish a reaction. Use observed_result for other current scientific results. '
            'Cite the source IDs in decision.evidence_ids. If deferral holds even while the proposed '
            'measurements remain unknown, request_scope requires an exact applicable scope quote and no '
            'measured premises. Explain why the scope supports stopping without those assumptions.'),
    }


def _current_scope(state, eid, kind, atom_ids=None):
    entry = state['research']['evidence'][eid]
    observation = entry['observation']
    action = observation.get('action')
    result = observation.get('result', {})
    if result.get('error'):
        return ()
    candidates = state.get('candidates', {})
    if kind == 'pose_comparison' and action == 'compare_structures':
        a, b = result.get('original_id'), result.get('candidate_id')
        metric = result.get('ligand_pose_comparison', {})
        value = metric.get('symmetry_corrected_pose_rmsd_A')
        if (a in candidates and b in candidates and a != b
                and result.get('original_sha256') == candidates[a]['sha256']
                and result.get('candidate_sha256') == candidates[b]['sha256']
                and metric.get('status') == 'measured'
                and type(value) in (int, float) and isfinite(value) and value >= 0):
            return tuple(sorted((a, b)))
    if kind == 'contact_localization' and action == 'measure_contacts':
        cid = result.get('candidate_id')
        value = result.get('minimum_distance_A')
        if (cid in candidates and result.get('candidate_sha256') == candidates[cid]['sha256']
                and result.get('residues') and type(value) in (int, float)
                and isfinite(value) and value >= 0):
            return (cid,)
    if kind in ATOM_PREMISES and action == ATOM_PREMISES[kind]:
        from tetra.research.belief_tracking import usable_source
        cid = result.get('candidate_id')
        if (cid in candidates and result.get('status') == 'measured' and usable_source(entry)
                and result.get('candidate_sha256') == candidates[cid]['sha256']):
            if kind == 'atom_geometry' and result.get('atom_ids') == atom_ids:
                return (cid,)
            if kind == 'contact_chemistry' and any(
                    {row['protein']['atom_id'], row['ligand']['atom_id']} == set(atom_ids or ())
                    for row in result.get('contacts', [])):
                return (cid,)
    if kind == 'observed_result' and action not in ('compare_structures', 'measure_contacts', *ATOM_PREMISES.values()):
        from tetra.research.belief_tracking import freshness, source_binding, usable_source
        if usable_source(entry):
            source = source_binding(state['research'], eid)
            if not freshness(state, source).startswith('stale'):
                return tuple(source.get('candidate_dependencies', {})) or (source.get('candidate_id'),)
    return ()


def coverage(state):
    """Report what is measured without choosing a scientific threshold."""
    candidates = state.get('candidates', {})
    ids = state['research']['evidence']
    pairs = {_current_scope(state, eid, 'pose_comparison') for eid in ids} - {()}
    contacts = {_current_scope(state, eid, 'contact_localization') for eid in ids} - {()}
    return {'pose_pairs_measured': len(pairs), 'pose_pairs_total': len(candidates) * (len(candidates) - 1) // 2,
            'contact_candidates_measured': len(contacts), 'candidate_count': len(candidates)}


def _unique_strings(items):
    return (isinstance(items, list) and all(isinstance(item, str) and item.strip() for item in items)
            and len(set(items)) == len(items))


class DeferralBasisMissing(ValueError):
    """A proposed deferral lacks an explicit measurement or request basis."""


def evidence_gap_guidance(state, available):
    """Offer current sources or relevant measurements for an unresolved distance failure."""
    evidence = state['research']['evidence']
    read = set(state.get('evidence_read_ids', []))
    from tetra.research.decision_policy import current_measured_limitations
    options = []
    for cid, failed in current_measured_limitations(state).items():
        if 'minimum_distance_to_protein' not in failed:
            continue
        candidate_hash = state['candidates'][cid]['sha256']
        sources = {}
        for eid, entry in evidence.items():
            observation = entry['observation']
            result = observation.get('result', {})
            if (result.get('candidate_id') != cid or result.get('candidate_sha256') != candidate_hash
                    or result.get('error')):
                continue
            action = observation['action']
            if action == 'assess_structure' and result.get('checks', {}).get('minimum_distance_to_protein') != 'fail':
                continue
            if action == 'measure_contacts' and not _current_scope(state, eid, 'contact_localization'):
                continue
            if action == 'inspect_contact_chemistry' and not (
                    result.get('status') == 'measured' and not result.get('residue_positions') and result.get('contacts')):
                continue
            sources[action] = eid
        assessment = sources.get('assess_structure')
        contacts = sources.get('measure_contacts')
        chemistry = sources.get('inspect_contact_chemistry')
        if not assessment:
            continue
        if state.get('evidence_access_mode') == 'files-v1' or 'evidence_read_ids' in state:
            unread = [eid for eid in (assessment, contacts, chemistry) if eid and eid not in read]
            if unread:
                options.append({'action': 'read_evidence', 'arguments': {'evidence_ids': unread},
                                'answers': 'Inspect the existing current measurement before repeating it.'})
                continue
        if not chemistry and 'inspect_contact_chemistry' in available:
            options.append({
                'action': 'inspect_contact_chemistry',
                'arguments': {'candidate_id': cid, 'residue_positions': []},
                'source_evidence_ids': [eid for eid in (assessment, contacts) if eid],
                'answers': 'Which atoms and ligand group are involved in the measured shortest contact?',
                'leaves_open': 'Reaction geometry and whether a revision preserves the pose.',
            })
    if not options:
        return {}
    return {
        'kind': 'deferral_evidence_gap_not_a_measurement',
        'options': options,
        'instruction': (
            'Use a listed unread source, or commit one relevant new measurement and execute it. '
            'For a chemistry premise, the distance check and residue contacts alone do not identify a mechanism. '
            'For a repair-risk premise, decide whether a feasible before-and-after test can change delivery. '
            'Do not repeat the same audit submission. Defer only if a valid basis already supports stopping '
            'with the untested premises left unknown.'
        ),
    }


def bounded_revision_option(state, available):
    """Describe executable branches for one as-yet untested local geometry repair."""
    if not state.get('refinement_enabled') or not {'register_trigger', 'plan_revision'} & set(available):
        return None
    if not state.get('uncapped_experiment') and state.get('refinement_count', 0) >= 2:
        return None
    from tetra.research.decision_policy import current_measured_limitations
    from tetra.evaluation.refinement import refinement_available
    from tetra.evaluation.revision_policy import _contacts
    from tetra.evaluation.revision_tools import revision_trigger_candidates

    evidence = state['research']['evidence']
    read = set(state.get('evidence_read_ids', evidence))
    triggers = revision_trigger_candidates(state)
    for cid, failed in current_measured_limitations(state).items():
        assessment = state['assessments'][cid]
        if ('minimum_distance_to_protein' not in failed or not assessment.get('refinement_applicable')
                or assessment.get('checks', {}).get('identity') != 'pass'
                or any(plan.get('candidate_id') == cid and plan.get('candidate_sha256') == state['candidates'][cid]['sha256']
                       for plan in state.get('revisions', {}).values())):
            continue
        candidate_hash = state['candidates'][cid]['sha256']
        sources = {}
        for eid, record in evidence.items():
            obs = record['observation']
            result = obs.get('result', {})
            if (eid not in read or result.get('candidate_id') != cid
                    or result.get('candidate_sha256') != candidate_hash or result.get('error')):
                continue
            if obs['action'] == 'assess_structure' and result.get('checks', {}).get('minimum_distance_to_protein') == 'fail':
                sources['assessment'] = eid
            if obs['action'] == 'inspect_contact_chemistry' and result.get('status') == 'measured' and result.get('contacts'):
                sources['chemistry'] = eid
        _, contacts = _contacts(state, cid)
        if not all(k in sources for k in ('assessment', 'chemistry')) or contacts not in read:
            continue
        if not refinement_available():
            return None
        question = 'Can a bounded local revision relieve the measured distance failure while preserving the declared pose and contacts?'
        refs = [sources['assessment'], contacts, sources['chemistry']]
        action = 'plan_revision' if cid in triggers else 'register_trigger'
        option = {
            'action': action, 'candidate_id': cid, 'evidence_ids': refs, 'question': question,
            'possible_result_changes_candidate_pool': True,
            'parent_retained_in_all_outcomes': True,
            'outcomes': {
                'pass': 'admit_qualified_child_for_selection',
                'fail': 'retain_parent_with_limitations',
                'tool_failure': 'no_scientific_result_retain_parent',
            },
            'instruction': (
                'Compare the post-test candidate pools before deciding whether to defer. Untested pose '
                'disturbance belongs in the frozen preservation criteria. A passing child can change '
                'selection even when there is only one candidate now. If you defer, cite a real constraint '
                'or measured result that still justifies stopping under these different outcomes.'
            ),
        }
        if action == 'register_trigger':
            option['arguments'] = {'candidate_id': cid, 'trigger_type': 'physical_issue',
                                   'evidence_ids': refs, 'goal_quote': '', 'question': question}
        return option
    return None


def require_defer_grounding(state, decision):
    draft = decision['selection_followup']
    basis = draft.get('defer_basis')
    if not isinstance(basis, dict) or not {'kind', 'premises'} <= set(basis) or set(basis) - {
            'kind', 'premises', 'request_quote'}:
        measured = coverage(state)
        raise DeferralBasisMissing('Provide decision.selection_followup.defer_basis. Evidence coverage: '
                         f"pose comparison {measured['pose_pairs_measured']}/{measured['pose_pairs_total']}, "
                         f"contact localization {measured['contact_candidates_measured']}/{measured['candidate_count']}. "
                         'Keep unmeasured premises unresolved; consider a discriminating measurement.')
    kind, premises = basis['kind'], basis['premises']
    if kind == 'request_scope':
        quote = basis.get('request_quote')
        if not isinstance(quote, str) or not quote.strip() or quote not in state.get('goal', '') or premises != []:
            raise ValueError('request_scope needs an exact quote from the request and no measured premises.')
        return
    if kind != 'measured' or not isinstance(premises, list) or not premises:
        raise ValueError('Measured deferral needs at least one decision-relevant measurement premise.')
    observed = state['research']['evidence']
    cited = set(decision.get('evidence_ids', []))
    for index, premise in enumerate(premises):
        location = f'defer_basis.premises[{index}]'
        fields = {'kind', 'candidate_ids', 'evidence_ids'}
        if not isinstance(premise, dict) or not fields <= set(premise):
            raise ValueError(f'{location} needs kind, candidate_ids and evidence_ids.')
        purpose, ids, refs = premise['kind'], premise['candidate_ids'], premise['evidence_ids']
        atom_ids = premise.get('atom_ids')
        if isinstance(purpose, str) and purpose in ATOM_PREMISES:
            lengths = (2,) if purpose == 'contact_chemistry' else (2, 3, 4)
            if not _unique_strings(atom_ids) or len(atom_ids) not in lengths:
                raise ValueError(f'{location}.atom_ids must list distinct measured atom IDs (allowed counts: {lengths}).')
            fields.add('atom_ids')
        if set(premise) != fields:
            raise ValueError(f'{location}: unexpected premise fields.')
        if (purpose not in KINDS or not _unique_strings(ids) or not _unique_strings(refs) or not refs
                or any(cid not in state.get('candidates', {}) for cid in ids)
                or any(eid not in cited or eid not in observed for eid in refs)):
            raise ValueError(f'{location} must name current candidates and actual Evidence cited by the decision.')
        scopes = [_current_scope(state, eid, purpose, atom_ids) for eid in refs]
        if purpose == 'pose_comparison':
            missing = set(combinations(sorted(ids), 2)) - set(scopes)
            if len(ids) < 2 or missing:
                raise ValueError(f'{location}: unmeasured pose pairs {sorted(missing)}. '
                                 'Compare a decision-relevant pair or keep the equivalence assumption unresolved.')
        elif purpose == 'contact_localization':
            missing = set(ids) - {scope[0] for scope in scopes if scope}
            if not ids or missing:
                raise ValueError(f'{location}: unmeasured contact localization for {sorted(missing)}. '
                                 'Measure the relevant contacts or keep recurrence unresolved.')
        elif purpose in ATOM_PREMISES:
            missing = set(ids) - {scope[0] for scope in scopes if scope}
            if not ids or missing:
                atom_scope = 'the measured atom pair' if purpose == 'contact_chemistry' else 'the ordered atom IDs'
                raise ValueError(f'{location}: cite current {ATOM_PREMISES[purpose]} evidence '
                                 f'for the named candidates and {atom_scope}.')
        elif not any(scopes) or not set(ids).issubset({cid for scope in scopes for cid in scope}):
            raise ValueError(f'{location}: cited records lack a current scientific result for the stated scope.')
