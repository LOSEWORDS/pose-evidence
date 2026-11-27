import copy
import unittest

from tetra.orchestration.board_loop import check_committed_action
from tetra.research.scientist_memory import initial_team


class LiteratureBoardAccessTests(unittest.TestCase):
    def state(self):
        state = {'literature_enabled': True, 'research': initial_team('Predict a complex')}
        state['research']['lead'].update(
            decision={'next_action': 'plan_prediction_experiment'}, decision_consumed=False)
        return state

    def test_available_literature_can_inform_an_unconsumed_plan(self):
        for action in ('search_literature', 'read_literature'):
            with self.subTest(action=action):
                state = self.state()
                before = copy.deepcopy(state)
                check_committed_action(state, action, ['plan_prediction_experiment', action])
                self.assertEqual(state, before)

    def test_unavailable_literature_does_not_bypass_board(self):
        for action in ('search_literature', 'read_literature'):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, 'Action differs'):
                    check_committed_action(self.state(), action, ['plan_prediction_experiment'])

    def test_scientific_intervention_still_requires_board_decision(self):
        with self.assertRaisesRegex(ValueError, 'Action differs'):
            check_committed_action(self.state(), 'execute_prediction_experiment',
                                   ['plan_prediction_experiment', 'execute_prediction_experiment'])

    def test_non_literature_mode_retains_commitment(self):
        state = self.state()
        state['literature_enabled'] = False
        with self.assertRaisesRegex(ValueError, 'Action differs'):
            check_committed_action(state, 'search_literature',
                                   ['plan_prediction_experiment', 'search_literature'])

    def test_screened_source_makes_details_the_next_literature_action(self):
        import tempfile
        from tetra.orchestration.scientific_agent import ScientificAgent
        from tetra.runtime.store import Store

        state = self.state()
        state.update(status='running', candidates={}, resolved={'input_id': 'input'},
                     observations=[{'action': 'read_literature',
                                    'arguments': {'source_id': 'MED:1'},
                                    'result': {'source_id': 'MED:1', 'status': 'retrieved'}}])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientificAgent(Store(root))
            tools = agent.available_tools(state)
            self.assertIn('read_structure_details', tools)
            self.assertNotIn('search_literature', tools)
            self.assertNotIn('plan_prediction_experiment', tools)
            state['observations'].append({'action': 'read_structure_details',
                                          'result': {'source_id': 'MED:1', 'status': 'retrieved'}})
            self.assertIn('plan_prediction_experiment', agent.available_tools(state))

    def test_detail_schema_lists_only_successfully_screened_sources(self):
        from tetra.orchestration.scientific_agent import ScientificAgent

        state = {'observations': [
            {'action': 'search_literature', 'result': {'results': [
                {'source_id': 'MED:1'}, {'source_id': 'MED:2'}, {'source_id': 'MED:3'}]}},
            {'action': 'read_literature', 'result': {'source_id': 'MED:1', 'status': 'retrieved'}},
            {'action': 'read_literature', 'result': {'source_id': 'MED:2', 'status': 'unavailable'}},
        ]}
        schema = ScientificAgent.literature_read_schema(state, 'read_structure_details')
        ids = schema['function']['parameters']['properties']['source_id']['enum']
        self.assertEqual(ids, ['MED:1'])

    def test_preflight_skips_blocked_tools_and_offers_one_review_action(self):
        from tetra.orchestration.scientific_agent import ScientificAgent

        state = self.state()
        state.update(evidence_access_mode='files-v1', evidence_read_ids=[])
        team = state['research']
        team.update(review_required=True, unreviewed_evidence=['E1'])
        review_tools = ['update_memory', 'read_evidence', 'read_evidence_bundle', 'stop']
        self.assertEqual(ScientificAgent.offer_tools(state, review_tools), ['read_evidence_bundle'])
        state['evidence_read_ids'] = ['E1']
        self.assertEqual(ScientificAgent.offer_tools(state, review_tools), ['update_memory'])
        team['review_required'] = False
        tools = ['update_memory', 'plan_prediction_experiment', 'execute_prediction_experiment']
        self.assertEqual(ScientificAgent.offer_tools(state, tools),
                         ['update_memory', 'plan_prediction_experiment'])
