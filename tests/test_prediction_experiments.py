import tempfile
import unittest
from pathlib import Path

import gemmi

from tetra.prediction.prediction_experiments import measure_conditions, register


class PredictionExperimentTests(unittest.TestCase):
    def test_plan_requires_read_evidence_and_deduplicates(self):
        state = {'resolved': {'input_id': 'fixture'}, 'evidence_access_mode': 'files-v1',
                 'research': {'evidence': {'E1': {}}}}
        record = {'sequence': 'ACD', 'smiles': 'CCO'}
        arguments = {'question': 'Where does the ligand sit?',
                     'hypothesis': 'The oxygen is close to residue 2.',
                     'evidence_ids': ['E1'], 'expected_observations': 'Distance',
                     'decision_implications': 'Compare poses', 'config': {'diffusion_samples': 2},
                     'conditions': [{'contact': {'token1': ['A', 2],
                                                 'token2': ['B', {'smarts': '[OX2]', 'atom': 0, 'element': 'O'}],
                                                 'max_distance': 6, 'force': False}}]}
        with self.assertRaisesRegex(ValueError, 'unread'):
            register(state, record, arguments)
        state['evidence_read_ids'] = ['E1']
        selector = arguments['conditions'][0]['contact']['token2'][1]
        selector['element'] = 'C'
        with self.assertRaisesRegex(ValueError, r'resolved to O[0-9]+ \(O\)'):
            register(state, record, arguments)
        selector['element'] = 'O'
        plan = register(state, record, arguments)
        self.assertEqual(plan['config']['diffusion_samples'], 2)
        self.assertTrue(plan['config']['contact_constraints'][0]['token2'][1].startswith('O'))
        with self.assertRaisesRegex(ValueError, 'already registered'):
            register(state, record, arguments)

    def test_condition_is_measured_from_coordinates(self):
        pdb = (
            'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n'
            'HETATM    2  O3  LIG B   1       5.000   0.000   0.000  1.00 20.00           O\n'
            'END\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / 'model.cif'
            gemmi.read_pdb_string(pdb).make_mmcif_document().write_file(str(cif))
            first = {'token1': ['A', 1], 'token2': ['B', 'O3'], 'max_distance': 4}
            second = {**first, 'max_distance': 6}
            rows = measure_conditions(cif, {'contact_constraints': [first, second]})
        self.assertEqual([r['distance_A'] for r in rows], [5, 5])
        self.assertEqual([r['satisfied'] for r in rows], [False, True])

    def test_second_round_cites_measurement_and_changes_condition(self):
        state = {'resolved': {'input_id': 'fixture'}, 'evidence_access_mode': 'files-v1',
                 'evidence_read_ids': ['E1'], 'research': {'evidence': {'E1': {}}}}
        record = {'sequence': 'ACD', 'smiles': 'CCO'}
        args = {'question': 'Is the oxygen near the first pocket residue?',
                'hypothesis': 'The oxygen contacts residue 1.',
                'evidence_ids': ['E1'], 'expected_observations': 'Measure distance',
                'decision_implications': 'Compare proximity at different residues.',
                'config': {'diffusion_samples': 2}}
        def condition(position):
            return [{'contact': {'token1': ['A', position],
                                 'token2': ['B', {'smarts': '[OX2]', 'atom': 0, 'element': 'O'}],
                                 'max_distance': 4, 'force': False}}]
        first = register(state, record, {**args, 'conditions': condition(1)})
        state['prediction_experiments'][first['experiment_id']]['status'] = 'succeeded'
        state['research']['evidence']['E2'] = {'observation': {
            'action': 'prediction_completed', 'result': {
                'experiment_id': first['experiment_id'],
                'condition_measurements': {'candidate1': {'distances': [
                    {'distance_A': 5.0, 'satisfied': False}]}}}}}
        with self.assertRaisesRegex(ValueError, 'completed prediction outcome Evidence ID: E2'):
            register(state, record, {**args, 'conditions': condition(2)})
        second_args = {**args, 'hypothesis': 'The oxygen is nearer residue 2.',
                       'evidence_ids': ['E2'], 'conditions': condition(2)}
        with self.assertRaisesRegex(ValueError, 'unread'):
            register(state, record, second_args)
        state['evidence_read_ids'].append('E2')
        state.update(literature_enabled=True, observations=[
            {'action': 'search_literature', 'result': {'results': [{'source_id': 'MED:1'}]}},
            state['research']['evidence']['E2']['observation'],
        ])
        with self.assertRaisesRegex(ValueError, 'focused literature search'):
            register(state, record, second_args)
        state['observations'].append({'action': 'search_literature', 'result': {'results': []}})
        with self.assertRaisesRegex(ValueError, 'cannot directly test atomic separation'):
            register(state, record, {**second_args, 'condition_purpose': 'atomic_separation'})
        second = register(state, record, {**second_args, 'condition_purpose': 'proximity'})
        self.assertEqual(second['literature_followup']['status'], 'no_hits')
        state['observations'][-1]['result'] = {'error': 'service unavailable'}
        retry_plan = register(state, record, {**second_args, 'config': {'seed': 43}})
        self.assertEqual(retry_plan['literature_followup']['status'], 'retrieval_failed')
        self.assertEqual(retry_plan['literature_followup']['source_evidence_ids'], [])
        self.assertNotEqual(first['experiment_id'], second['experiment_id'])
        self.assertEqual(second['config']['contact_constraints'][0]['token1'], ['A', 2])

    def test_unsatisfied_prediction_requires_inconclusive_review_before_next_plan(self):
        from tetra.prediction.experiment_lifecycle import VERSION, sync, pending, require_review, validate_reviews

        identifier = 'batch'
        condition = {'token1': ['A', 1], 'token2': ['B', 'O3'],
                     'max_distance': 4, 'force': True}
        plan = {'experiment_id': identifier, 'status': 'succeeded',
                'config': {'contact_constraints': [condition]}}
        outcome = {'experiment_id': identifier, 'candidate_ids': ['candidate'],
                   'condition_measurements': {'candidate': {'distances': [{'satisfied': False}]}}}
        evidence = {
            'E1': {'observation': {'action': 'plan_prediction_experiment', 'result': plan}},
            'E2': {'observation': {'action': 'prediction_completed', 'result': outcome}},
        }
        team = {'experiment_workflow_version': VERSION, 'evidence': evidence}
        state = {'research': team, 'prediction_experiments': {identifier: plan}}
        sync(state)
        sync(state)
        key = 'prediction:' + identifier
        self.assertEqual(pending(team), [key])
        self.assertEqual(team['experiments'][key]['execution_evidence_ids'], ['E2'])
        require_review(team, 'search_literature')
        for action in ('plan_prediction_experiment', 'execute_prediction_experiment',
                       'select_candidates', 'stop'):
            with self.assertRaisesRegex(ValueError, key):
                require_review(team, action)
        review = dict.fromkeys(('manipulation_check', 'control_result', 'criteria_evaluation',
                                'interpretation', 'next_decision'), 'Review')
        review.update(experiment_id=key, evidence_ids=['E1', 'E2'],
                      manipulation_status='effective', hypothesis_outcome='contradicted',
                      limitations=['No sample met the bound.'])
        decision = {'next_action': 'plan_prediction_experiment', 'evidence_ids': ['E1', 'E2'],
                    'experiment_reviews': [review]}
        with self.assertRaisesRegex(ValueError, 'No prediction candidate'):
            validate_reviews(team, decision)
        review.update(manipulation_status='ineffective', hypothesis_outcome='inconclusive')
        validate_reviews(team, decision)

    def test_existing_baseline_requires_condition_and_structural_evidence_in_first_plan(self):
        detail = {'action': 'read_structure_details',
                  'result': {'status': 'retrieved', 'source_id': 'MED:123'}}
        state = {'resolved': {'input_id': 'fixture'}, 'conditioned_first': True,
                 'literature_enabled': True, 'observations': [detail], 'research': {'evidence': {
                     'E1': {'observation': {'action': 'resolve_input'}},
                     'E2': {'observation': detail}}}}
        record = {'sequence': 'ACD', 'smiles': 'CCO'}
        args = {'question': 'Does the ligand sample the mapped pocket?',
                'hypothesis': 'It can sample this site.', 'conditions': [], 'config': {},
                'expected_observations': 'Pocket distances', 'decision_implications': 'Compare poses',
                'literature_condition_basis': 'Reference residue maps to position 2.',
                'evidence_ids': ['E2']}
        with self.assertRaisesRegex(ValueError, 'baseline already exists'):
            register(state, record, args)
        args['conditions'] = [{'pocket': {'binder': 'B', 'contacts': [['A', 2]],
                                          'max_distance': 6, 'force': False}}]
        args['evidence_ids'] = ['E1']
        with self.assertRaisesRegex(ValueError, 'Cite the detailed structural passage'):
            register(state, record, args)
        args['evidence_ids'] = ['E2']
        plan = register(state, record, args)
        self.assertEqual(plan['config']['pocket_constraints'][0]['contacts'], [['A', 2]])

    def test_identity_failure_preserves_other_batch_measurements(self):
        from unittest.mock import patch
        from tetra.prediction.prediction_experiments import analyze_batch

        good = {'candidate_id': 'good', 'sha256': '1'}
        bad = {'candidate_id': 'bad', 'sha256': '2'}
        state = {'candidates': {'good': good, 'bad': bad}, 'observations': [],
                 'prediction_experiments': {'batch': {'status': 'succeeded',
                                                       'candidate_ids': ['good', 'bad']}}}

        def assess(candidate_id):
            return {'candidate_id': candidate_id, 'status': 'assessed',
                    'checks': {'identity': 'pass' if candidate_id == 'good' else 'fail'}}

        contact = {'candidate_id': 'good', 'candidate_sha256': '1', 'cutoff_A': 4.0,
                   'minimum_distance_A': 2.7, 'contact_positions': [2]}
        with patch('tetra.evaluation.structural_metrics.measure_contacts', side_effect=[
            contact, ValueError('Candidate identity/completeness check failed.')
        ]) as measure, patch('tetra.evaluation.structural_metrics.compare_structures') as compare:
            result = analyze_batch(state, 'batch', assess)
        self.assertEqual(result['contacts'], [contact])
        self.assertEqual(result['assessment_results'][1]['checks']['identity'], 'fail')
        self.assertEqual(result['excluded_candidates'][0]['candidate_id'], 'bad')
        self.assertEqual(result['comparison_status'], 'insufficient_candidates')
        measure.assert_called_once_with(good)
        compare.assert_not_called()

    def test_plan_rejects_unknown_binder_before_registering(self):
        state = {'resolved': {'input_id': 'fixture'},
                 'research': {'evidence': {'E1': {}}}}
        record = {'sequence': 'ACD', 'smiles': 'CCO'}
        pocket = {'binder': 'LIG', 'contacts': [['A', 2]], 'max_distance': 5, 'force': False}
        args = {'question': 'Does the ligand sample the pocket?', 'hypothesis': 'It can.',
                'evidence_ids': ['E1'], 'expected_observations': 'Measure position.',
                'decision_implications': 'Inspect the pose.', 'config': {},
                'conditions': [{'pocket': pocket}]}
        with self.assertRaisesRegex(ValueError, r"binder.*LIG.*ligand chain.*B"):
            register(state, record, args)
        self.assertFalse(state.get('prediction_experiments'))
        pocket['binder'] = 'B'
        self.assertEqual(register(state, record, args)['config']['pocket_constraints'][0]['binder'], 'B')
