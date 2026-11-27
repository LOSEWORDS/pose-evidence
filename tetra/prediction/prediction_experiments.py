"""Freeze and execute evidence-linked Boltz condition groups."""
from copy import deepcopy

from tetra.research.evidence_files import require_reads
from tetra.runtime.inputs import digest, stable, validate_config
from tetra.prediction.prediction_conditions import map_ligand_atom


def require_mapped_reference_support(conditions, cited_details):
    """Coordinate-only references ground residues, not atoms of the target ligand."""
    if not cited_details or any('pocket_drafts' not in row
                                or row.get('fulltext_detail_status') == 'retrieved'
                                for row in cited_details):
        return
    sites = [set(draft['target_positions']) for row in cited_details for draft in row['pocket_drafts']]
    for condition in conditions:
        if 'contact' in condition:
            raise ValueError('Reference proximity alone does not identify the target ligand atom. '
                             'Use mapped pocket positions or cite transferable atom-level evidence.')
        if 'pocket' in condition:
            tokens = condition['pocket'].get('contacts', [])
            if not any(all(chain == 'A' and position in site for chain, position in tokens)
                       for site in sites):
                raise ValueError('Each pocket input residue must map to a single cited reference site.')


def register(state, record, arguments):
    ids = arguments['evidence_ids']
    evidence = state['research']['evidence']
    if not isinstance(ids, list) or not ids or any(
        not isinstance(key, str) or key not in evidence for key in ids
    ):
        raise ValueError('Cite existing literature or measurement evidence.')
    require_reads(state, ids)
    if state.get('literature_enabled'):
        from tetra.research.literature import structural_followup_blocker

        blocker = structural_followup_blocker(state.get('observations', []))
        if blocker:
            raise ValueError(blocker)
    plans = state.setdefault('prediction_experiments', {})
    completed = {key for key, plan in plans.items()
                 if plan['status'] in ('succeeded', 'failed')}
    outcome_ids = [
        key for key, entry in evidence.items()
        if entry.get('observation', {}).get('action') in
        ('prediction_completed', 'prediction_failed', 'prediction_import_failed')
        and entry['observation'].get('result', {}).get('experiment_id') in completed
    ]
    if completed and not set(ids).intersection(outcome_ids):
        options = ', '.join(outcome_ids) or 'none recorded'
        raise ValueError('Cite a completed prediction outcome Evidence ID: ' + options)
    followup = None
    if completed and state.get('literature_enabled'):
        latest = next((index for index, observation in
                       reversed(list(enumerate(state.get('observations', []))))
                       if observation.get('action') in
                       ('prediction_completed', 'prediction_failed', 'prediction_import_failed')
                       and observation.get('result', {}).get('experiment_id') in completed), None)
        if latest is None:
            raise ValueError('No recorded prediction outcome for a focused literature follow-up.')
        observations = state['observations'][latest:]
        previous_id = observations[0]['result']['experiment_id']
        if not any(evidence[key]['observation']['result']['experiment_id'] == previous_id
                   for key in set(ids).intersection(outcome_ids)):
            raise ValueError('Cite the latest completed prediction outcome.')
        search_index = next((index for index in range(len(observations) - 1, 0, -1)
                             if observations[index]['action'] == 'search_literature'), None)
        if search_index is None:
            raise ValueError('Run one focused literature search based on the latest group before planning the next.')
        search = observations[search_index]
        reads = observations[search_index + 1:]
        hits = {hit['source_id'] for hit in search['result'].get('results', [])}
        cited_reads = [
            key for key in ids
            if (observation := evidence[key].get('observation', {})).get('action') == 'read_literature'
            and observation['result'].get('status') == 'retrieved'
            and observation['result'].get('source_id') in hits
            and any(row.get('action') == 'read_literature'
                    and row.get('result') == observation['result'] for row in reads)
        ]
        status = ('retrieval_failed' if 'error' in search['result'] else
                  'no_hits' if not hits else 'cited_source_read' if cited_reads else
                  'hits_without_cited_read')
        followup = {'status': status, 'prediction_experiment_id': previous_id,
                    'query': search.get('arguments', {}).get('query') or search['result'].get('query'),
                    'source_evidence_ids': cited_reads}
    conditions = arguments['conditions']
    if not isinstance(conditions, list):
        raise ValueError('Conditions must be a list.')
    conditioned_first = state.get('conditioned_first') and not state.get('prediction_count', 0)
    if conditioned_first and not conditions:
        raise ValueError('An unconditioned baseline already exists. Submit a first group with an '
                         'evidence-backed pocket or contact.')
    if state.get('literature_enabled'):
        observations = state.get('observations', [])
        terminal = {'prediction_completed', 'prediction_failed', 'prediction_import_failed'}
        start = max((i for i, row in enumerate(observations) if row['action'] in terminal), default=-1)
        details = [row.get('result', {}) for row in observations[start + 1:]
                   if row['action'] == 'read_structure_details']
        if conditioned_first and not any(row.get('status') == 'retrieved' for row in details):
            raise ValueError('Retrieve structural details or mapped reference measurements before '
                             'registering the first conditioned group.')
        if details:
            basis = arguments.get('literature_condition_basis')
            if not isinstance(basis, str) or not basis.strip():
                raise ValueError('Explain how the structural sources map to the submitted conditions, '
                                 'or why the mapping is unsupported, in literature_condition_basis.')
            if conditions and any(row.get('status') == 'retrieved' for row in details):
                cited = [evidence[key].get('observation', {}) for key in ids]
                if not any(row.get('action') == 'read_structure_details'
                           and row.get('result') in details
                           and row['result'].get('status') == 'retrieved' for row in cited):
                    raise ValueError('Cite the detailed structural passage used for the proposed conditions.')
    purpose = arguments.get('condition_purpose')
    if purpose is not None:
        expected = 'proximity' if conditions else 'unconstrained_sampling'
        if purpose != expected:
            raise ValueError(f'These inputs can express {expected}. Pocket/contact use only upper '
                             'distance bounds and cannot directly test atomic separation, '
                             'reaction geometry or covalent bonding. Redesign the intervention.')
    config = arguments['config']
    if not isinstance(config, dict) or set(config) - {'seed', 'diffusion_samples', 'recycling_steps', 'sampling_steps'}:
        raise ValueError('Use supported sampling settings; put constraints in conditions.')
    config = dict(config)
    mappings = []
    for condition in conditions:
        if not isinstance(condition, dict) or len(condition) != 1:
            raise ValueError('Each condition must contain a pocket or contact.')
        kind, value = next(iter(condition.items()))
        if kind not in ('pocket', 'contact') or not isinstance(value, dict):
            raise ValueError('Each condition must contain a pocket or contact.')
        value = deepcopy(value)
        if kind == 'contact':
            for key in ('token1', 'token2'):
                token = value.get(key)
                if isinstance(token, list) and len(token) == 2 and token[0] == 'B':
                    selector = token[1]
                    if not isinstance(selector, dict) or set(selector) != {'smarts', 'atom', 'element'}:
                        raise ValueError('Ligand token requires smarts, atom and the intended element.')
                    mapping = map_ligand_atom(record['smiles'], selector['smarts'], selector['atom'])
                    if selector['element'] != mapping['element']:
                        raise ValueError(
                            f"Expected ligand element {selector['element']}, but SMARTS atom "
                            f"{selector['atom']} resolved to {mapping['boltz_atom_name']} "
                            f"({mapping['element']}). Correct the selector before planning."
                        )
                    mappings.append(mapping)
                    value[key] = ['B', mapping['boltz_atom_name']]
        config.setdefault(kind + '_constraints', []).append(value)
    from tetra.prediction.backend import input_payload
    config = validate_config(config)
    input_payload(record, config.get('pocket_constraints'), config.get('contact_constraints'))
    if state.get('literature_enabled') and conditions:
        cited_details = [evidence[key]['observation']['result'] for key in ids
                         if evidence[key].get('observation', {}).get('action') == 'read_structure_details'
                         and evidence[key]['observation']['result'].get('status') == 'retrieved']
        require_mapped_reference_support(conditions, cited_details)
    input_id = state['resolved']['input_id']
    experiment_id = digest(stable({'input_id': input_id, 'config': config}))[:24]
    if experiment_id in plans:
        raise ValueError('This prediction configuration is already registered.')
    controls = [
        {'experiment_id': key, 'sampling_differences': {
            field: {'control': prior['config'].get(field), 'current': config[field]}
            for field in ('seed', 'diffusion_samples', 'recycling_steps', 'sampling_steps')
            if prior['config'].get(field) != config[field]
        }}
        for key, prior in plans.items()
        if prior['status'] == 'succeeded' and prior['input_id'] == input_id and not prior['conditions']
    ]
    plan = {**deepcopy(arguments), 'experiment_id': experiment_id, 'input_id': input_id,
            'config': config, 'atom_mappings': mappings, 'status': 'planned',
            'control_comparisons': controls, 'literature_followup': followup,
            'condition_scope': 'Protein tokens identify whole residues. Contact sets an upper distance '
                               'bound and does not specify a protein atom, a bond or a separation.'}
    plans[experiment_id] = plan
    return deepcopy(plan)


def measure_conditions(cif, config):
    """Measure the declared distance bounds on the generated CIF."""
    import gemmi
    import numpy as np

    model = gemmi.read_structure(str(cif))[0]

    def coordinates(token):
        chain = model[token[0]]
        selector = token[1]
        if type(selector) is int:
            residues = [residue for residue in chain if residue.seqid.num == selector]
            if len(residues) != 1:
                raise ValueError(f'Unresolved output residue: {token}')
            atoms = list(residues[0])
        else:
            atoms = [atom for residue in chain for atom in residue
                     if selector is None or atom.name == selector]
            if selector is not None and len(atoms) != 1:
                raise ValueError(f'Unresolved output atom: {token}')
        points = [list(atom.pos) for atom in atoms if not atom.element.is_hydrogen]
        if not points:
            raise ValueError(f'No heavy atoms for {token}')
        return np.array(points)

    rows = []
    for kind in ('pocket', 'contact'):
        for condition in config.get(kind + '_constraints', []):
            if kind == 'pocket':
                pairs = [([condition['binder'], None], token) for token in condition['contacts']]
            else:
                pairs = [(condition['token1'], condition['token2'])]
            for left, right in pairs:
                a, b = coordinates(left), coordinates(right)
                distance = float(np.linalg.norm(a[:, None] - b[None, :], axis=2).min())
                rows.append({'kind': kind, 'tokens': [left, right], 'distance_A': distance,
                             'max_distance_A': condition['max_distance'],
                             'satisfied': distance <= condition['max_distance']})
    return rows


def analyze_batch(state, experiment_id, assess):
    """Assess the group and retain usable measurements when an individual check fails."""
    from itertools import combinations

    from tetra.evaluation.structural_metrics import compare_structures, measure_contacts

    plan = state.get('prediction_experiments', {}).get(experiment_id)
    if not plan or plan['status'] != 'succeeded':
        raise ValueError('Choose a completed prediction experiment.')
    candidates = [state['candidates'][key] for key in plan.get('candidate_ids', [])]
    if not candidates:
        raise ValueError('The experiment has no imported candidates.')
    assessments, contacts, comparable, excluded, comparison_errors = [], [], [], [], []
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        try:
            assessment = assess(candidate_id)
        except (ValueError, OSError) as error:
            assessment = {'candidate_id': candidate_id, 'candidate_sha256': candidate['sha256'],
                          'status': 'unavailable', 'checks': {}, 'error': str(error)}
        assessments.append(assessment)
        if assessment.get('status') != 'assessed' or assessment.get('checks', {}).get('identity') != 'pass':
            excluded.append({'candidate_id': candidate_id,
                             'reason': assessment.get('error') or 'Identity check did not pass.'})
            continue
        try:
            row = measure_contacts(candidate)
        except (ValueError, OSError) as error:
            excluded.append({'candidate_id': candidate_id, 'reason': str(error)})
            continue
        comparable.append(candidate)
        contacts.append({key: row[key] for key in (
            'candidate_id', 'candidate_sha256', 'cutoff_A', 'minimum_distance_A', 'contact_positions'
        )})
    comparisons = []
    for before, after in combinations(comparable, 2):
        try:
            row = compare_structures(before, after)
        except (ValueError, OSError) as error:
            comparison_errors.append({'original_id': before['candidate_id'],
                                      'candidate_id': after['candidate_id'], 'error': str(error)})
            continue
        comparisons.append({key: row[key] for key in (
            'original_id', 'candidate_id', 'original_sha256', 'candidate_sha256', 'ligand_pose_comparison',
            'protein_CA_aligned_change_rms_A', 'lost_contact_positions', 'gained_contact_positions'
        )})
    outcomes = [row['result'].get('condition_measurements', {})
                for row in state['observations'] if row['action'] == 'prediction_completed'
                and row['result'].get('experiment_id') == experiment_id]
    measured_pairs = sum(row['ligand_pose_comparison'].get('status') == 'measured' for row in comparisons)
    if len(comparable) < 2:
        comparison_status = 'insufficient_candidates'
    elif not measured_pairs:
        comparison_status = 'unavailable'
    elif comparison_errors or measured_pairs < len(comparisons):
        comparison_status = 'partial'
    else:
        comparison_status = 'completed'
    return {'experiment_id': experiment_id, 'contacts': contacts, 'comparisons': comparisons,
            'comparison_status': comparison_status, 'comparison_errors': comparison_errors,
            'excluded_candidates': excluded, 'assessment_results': assessments,
            'condition_measurements': outcomes[-1] if outcomes else {},
            'interpretation_scope': 'Coordinate measurements on predicted candidates.'}
