"""Deliver real artifacts and evidence-linked board rounds, including explicit fallback."""
from pathlib import Path
import shutil
from tetra.evaluation.assessment import verify_candidate
from tetra.evaluation.structure_export import selected_complex
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest


def build_delivery(state, folder):
    folder = Path(folder) / 'delivery'
    folder.mkdir(parents=True, exist_ok=True)
    selected = (state.get('selected') or {}).get('selected_ids', [])
    verdicts = (state.get('selected') or {}).get('revision_verdicts', {})
    accepted_revisions = {key for key, row in verdicts.items() if row['decision'] == 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE'}
    failed = any(row['decision'] != 'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE' for row in verdicts.values()) or any(
        p.get('status') == 'failed' for p in state.get('revisions', {}).values())
    fallback = bool(selected and state.get('revisions') and selected[0] not in accepted_revisions)
    status = 'accepted_fallback' if fallback else 'accepted_candidate' if selected else 'diagnostic_only'
    qualified = (state.get('selected') or {}).get('delivery_qualification') == 'prediction_with_limitations'
    if qualified:
        status = 'prediction_with_limitations'
    candidate_decisions = (state.get('selected') or {}).get('candidate_decisions', {})
    artifacts = []
    for key, candidate in state['candidates'].items():
        role = ('primary' if selected and key == selected[0] else 'accepted_alternative' if key in selected else 'diagnostic')
        entry = {'candidate_id': key, 'role': role, 'candidate_sha256': candidate['sha256'], 'files': {}}
        entry['assessment'] = candidate_decisions.get(key, {})
        if role == 'accepted_alternative' and entry['assessment'].get('computational_acceptance') == 'NOT_ACCEPTED':
            entry['role'] = 'qualified_alternative'
        try:
            verify_candidate(candidate)
            destination = folder / key
            destination.mkdir(exist_ok=True)
            for name, suffix in [('receptor', '.pdb'), ('ligand', '.sdf')]:
                output = destination / (name + suffix)
                shutil.copyfile(candidate[name], output)
                entry['files'][name] = {'path': str(output.resolve()), 'sha256': digest(output.read_bytes())}
            if role != 'diagnostic':
                output = destination / 'complex.cif'
                selected_complex(candidate, output)
                entry['files']['complex'] = {'path': str(output.resolve()), 'sha256': digest(output.read_bytes())}
        except Exception as error:
            entry['error'] = str(error)
        artifacts.append(entry)
    primary = next((row for row in artifacts if row['role'] == 'primary'), None)
    if selected and (not primary or primary.get('error') or 'complex' not in primary['files']):
        status = 'delivery_error'
    delivered_count = sum(
        row['candidate_id'] in selected and not row.get('error') and 'complex' in row['files']
        for row in artifacts
    )
    requested_count = (state.get('delivery_requirement') or {}).get('count')
    if delivered_count < len(selected):
        status = 'delivery_error'
    elif requested_count is not None and delivered_count < requested_count and delivered_count:
        status = 'delivery_partial'
    lead = state['research']['lead']
    rounds = []
    for board in [*lead['history'], {k: v for k, v in lead.items() if k != 'history'}]:
        if not board['version']:
            continue
        refs = board.get('reviewed_evidence_ids', [])
        rounds.append({'board_version': board['version'], 'kind': 'board_update_not_structure_revision',
                       'observations': [{'evidence_id': key, 'action': state['research']['evidence'][key]['observation']['action']}
                                        for key in refs if key in state['research']['evidence']],
                       'working_memory': board['working_memory'], 'hypotheses': board['hypotheses'],
                       'decision': board.get('decision'), 'interpretation_is_model_generated': True})
    result = {'delivery_status': status, 'execution_status': state['status'], 'primary_candidate_id': selected[0] if selected else None,
              'goal_satisfaction': 'UNRESOLVED' if qualified or fallback or not selected else 'NOT_AUTOMATICALLY_VERIFIED',
              'requested_count': requested_count, 'delivered_count': delivered_count,
              'candidate_decisions': candidate_decisions, 'artifacts': artifacts, 'rounds': rounds, 'revision_plan_count': len(state.get('revisions', {})),
              'revision_execution_attempts': sum(bool(p.get('attempt_id')) and p.get('status') not in ('planned', 'cache_reused') for p in state.get('revisions', {}).values()),
              'structural_revision_count': sum(p.get('status') == 'succeeded' and bool(p.get('candidate_ids') or p.get('result', {}).get('candidate')) for p in state.get('revisions', {}).values()),
              'accepted_revision_candidate_ids': sorted(accepted_revisions),
              'unresolved_questions': lead.get('decision', {}).get('unresolved_questions', []),
              'limitations': ['Qualified predictions retain measured defects and are not computationally accepted.',
                             'Diagnostic files are not accepted predictions.',
                             'Fallback delivery does not satisfy an unachieved new-site hypothesis.',
                             'Board summaries are model interpretations linked to real observations, not internal chain of thought.']}
    atomic_json(folder / 'manifest.json', result)
    text = ['# Task 1 delivery', '', 'Status: ' + status, '', state['goal'], '',
            'Goal satisfaction: ' + result['goal_satisfaction'], '',
            'Board updates below are not counts of structural revision experiments.', '']
    for key in selected:
        detail = candidate_decisions.get(key, {})
        if detail:
            text += ['Candidate: ' + key, 'Qualification: ' + detail['delivery_qualification'],
                     'Failed checks: ' + ', '.join(detail['failed_checks']), '']
    for row in rounds:
        text += [f"## Board update {row['board_version']}", '', row['working_memory'], '']
        if row['decision']:
            d = row['decision']
            text += ['Next action: ' + d['next_action'], '', 'Basis: ' + d['reason'], '',
                     'Expected observation: ' + d['expected_observation'], '',
                     'Evidence: ' + ', '.join(d['evidence_ids']), '']
    (folder / 'rounds.md').write_text('\n'.join(text))
    return result
