import unittest
from tetra.orchestration.scientific_agent import ScientificAgent
from tetra.research.scientist_memory import initial_team


class ActionEfficiencyTests(unittest.TestCase):
    def state(self):
        state = {'literature_enabled': True, 'research': initial_team('Predict'),
                 'decision_log': [{'action': 'update_memory', 'success': True}],
                 'evidence_read_ids': ['E1']}
        state['research']['lead'].update(
            decision={'next_action': 'search_literature'}, decision_consumed=False)
        state['research']['evidence'] = {'E1': {}}
        return state

    def test_committed_search_runs_after_successful_memory_update(self):
        offered = ScientificAgent.offer_tools(self.state(), [
            'update_memory', 'search_literature', 'read_evidence', 'read_evidence_bundle'])
        self.assertEqual(offered, ['search_literature'])

    def test_committed_finish_uses_current_audit(self):
        from unittest.mock import patch

        state = self.state()
        state['research']['lead']['decision']['next_action'] = 'finish'
        options = ['read_evidence', 'update_memory', 'finish']
        with patch('tetra.research.decision_policy.require_stop_audit') as gate:
            self.assertEqual(ScientificAgent.offer_tools(state, options), ['finish'])
            gate.side_effect = ValueError('stale audit')
            self.assertIn('update_memory', ScientificAgent.offer_tools(state, options))

    def test_error_keeps_replanning_available(self):
        state = self.state()
        state['decision_log'].append({'action': 'search_literature', 'success': False})
        self.assertIn('update_memory', ScientificAgent.offer_tools(state, [
            'update_memory', 'search_literature', 'read_evidence']))

    def test_unread_evidence_keeps_read_and_update_available(self):
        state = self.state()
        state['research']['evidence']['E2'] = {}
        offered = ScientificAgent.offer_tools(state, [
            'update_memory', 'search_literature', 'read_evidence'])
        self.assertIn('read_evidence', offered)
        self.assertIn('update_memory', offered)

    def test_unavailable_committed_action_keeps_replanning(self):
        state = self.state()
        offered = ScientificAgent.offer_tools(state, ['update_memory', 'read_evidence'])
        self.assertIn('update_memory', offered)

    def test_conditioned_first_phase_prefers_sequence_then_plan(self):
        state = {**self.state(), 'conditioned_first': True, 'observations': []}
        options = ['update_memory', 'search_literature', 'read_literature',
                   'find_reference_complexes', 'plan_prediction_experiment']
        self.assertEqual(ScientificAgent.first_condition_tools(state, options),
                         ['update_memory', 'find_reference_complexes', 'plan_prediction_experiment'])
        state['observations'].append({'action': 'search_literature', 'arguments': {'route': 'sequence'},
                                      'result': {'results': []}})
        self.assertEqual(ScientificAgent.first_condition_tools(state, options), options[:-2] + options[-1:])
        state['observations'].append({'action': 'read_structure_details', 'result': {
            'status': 'retrieved', 'pocket_drafts': [{'target_positions': [19, 25]}]}})
        self.assertEqual(ScientificAgent.first_condition_tools(state, options),
                         ['update_memory', 'plan_prediction_experiment'])

    def test_inline_search_delivers_only_its_own_evidence(self):
        state = self.state()
        state['evidence_access_mode'] = 'files-v1'
        team = state['research']
        team['evidence'].update({'E2': {}, 'E3': {'observation': {'action': 'search_literature'}}})
        team.update(unreviewed_evidence=['E2', 'E3'], review_required=True)
        result = {'results': [{'source_id': 'MED:123'}]}
        payload = ScientificAgent.deliver_search_result(state, result)
        self.assertEqual(payload['discovery_result'], result)
        self.assertEqual(payload['evidence_id'], 'E3')
        self.assertEqual(state['evidence_read_ids'], ['E1', 'E3'])
        self.assertEqual(team['unreviewed_evidence'], ['E2'])
        self.assertTrue(team['review_required'])
        self.assertEqual(state['evidence_reads'][0]['delivery'], 'inline_search_result')

    def test_failed_plan_can_reopen_retrieval(self):
        state = {**self.state(), 'conditioned_first': True, 'observations': [
            {'action': 'search_literature', 'arguments': {'route': 'sequence'}, 'result': {}},
            {'action': 'read_structure_details', 'result': {
                'status': 'retrieved', 'pocket_drafts': [{'target_positions': [19, 25]}]}},
            {'action': 'plan_prediction_experiment', 'result': {'error': 'mapping issue'}},
        ]}
        options = ['update_memory', 'search_literature', 'plan_prediction_experiment']
        self.assertEqual(ScientificAgent.first_condition_tools(state, options), options)

    def test_sequence_hits_go_straight_to_reference_measurement(self):
        state = {**self.state(), 'conditioned_first': True, 'observations': [
            {'action': 'search_literature', 'arguments': {'route': 'sequence'}, 'result': {
                'results': [{'source_id': 'MED:21184410', 'pdb_ids': ['2XU1']},
                            {'source_id': 'MED:21898833', 'pdb_ids': ['2YJ2']}]}}]}
        options = ['update_memory', 'search_literature', 'read_evidence', 'measure_reference_pocket']
        self.assertEqual(ScientificAgent.offer_tools(state, options), ['measure_reference_pocket'])
        schema = ScientificAgent.literature_read_schema(state, 'measure_reference_pocket')
        self.assertEqual(schema['function']['parameters']['properties']['source_id']['enum'],
                         ['MED:21184410', 'MED:21898833'])

    def test_reference_measurement_uses_screened_group_without_paper_read(self):
        from unittest.mock import Mock, patch

        state = {**self.state(), 'resolved': {'input_id': 'fixture'}, 'observations': [
            {'action': 'search_literature', 'arguments': {'route': 'sequence'}, 'result': {
                'results': [{'source_id': 'MED:1', 'pdb_ids': ['2XU1']}]}}]}
        agent = ScientificAgent.__new__(ScientificAgent)
        agent.store = Mock()
        with patch('tetra.research.literature_structures.structural_references') as measure:
            measure.return_value = {'status': 'retrieved', 'entries': [{
                'pdb_id': '2XU1', 'coordinate_sha256': 'hash', 'measurements': [{
                    'chain': 'A', 'reference_ligand': 'REF', 'contacts': [{'target_position': 19}]}]}]}
            result = agent.dispatch(state, 'measure_reference_pocket', {'source_id': 'MED:1'}, 'test')
            self.assertEqual(result['pocket_drafts'][0]['target_positions'], [19])
            self.assertEqual(result['fulltext_detail_status'], 'not_read')
            with self.assertRaisesRegex(ValueError, 'screened sequence-search'):
                agent.dispatch(state, 'measure_reference_pocket', {'source_id': 'MED:unknown'}, 'test')
            self.assertEqual(measure.call_count, 1)
