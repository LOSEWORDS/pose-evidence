import unittest
from unittest.mock import Mock, patch
from tetra.orchestration.tool_registry import TOOLS, eligible_tools, validate_arguments
from tetra.orchestration.workflow import Controller
from tetra.research.scientist_memory import initial_team, record_observation, update_memory


class SamplingTests(unittest.TestCase):
    def state(self):
        return dict(resolved={'input_id':'input'}, prediction_count=1,
                    candidates={}, assessments={}, research={}, status='running')

    def test_additional_sampling_is_available_with_budget(self):
        state = self.state()
        self.assertIn('predict_complex', eligible_tools(state))
        state['prediction_count'] = 3
        self.assertNotIn('predict_complex', eligible_tools(state))
        state['uncapped_experiment'] = True
        self.assertIn('predict_complex', eligible_tools(state))
        state['status'] = 'waiting_prediction'
        self.assertNotIn('predict_complex', eligible_tools(state))

    def test_sampling_schema_is_optional_and_bounded(self):
        params = TOOLS['predict_complex'].schema('predict_complex')['function']['parameters']
        self.assertEqual(params['required'], [])
        self.assertEqual(params['properties']['diffusion_samples']['maximum'], 5)

    def test_submit_preserves_pool_and_invalidates_selection(self):
        controller = object.__new__(Controller)
        controller.store = Mock()
        controller.store.submit.return_value = {'job_id':'new'}
        controller.save = Mock()
        controller.folder = Mock()
        state = self.state()
        state.update(selected={'selected_ids':['old']}, delivery_audit={'done':True})
        state['candidates']['old'] = {'sha256':'old'}
        with patch('tetra.orchestration.workflow.backend.launch', return_value={'job_id':'new'}):
            controller.dispatch(state, 'predict_complex', {'diffusion_samples':3,'seed':7}, 'call')
        controller.store.submit.assert_called_once_with('input', config={'diffusion_samples':3,'seed':7})
        self.assertEqual(state['prediction_count'], 2)
        self.assertIn('old', state['candidates'])
        self.assertIsNone(state['selected'])
        self.assertNotIn('delivery_audit', state)

    def test_rejection_identifies_lesson_and_correction(self):
        team = initial_team('test')
        record_observation(team, {'action':'select_candidates','result':{}})
        lesson = {'claim':'Selection completed','scope':'selection','evidence_ids':['E1']}
        with self.assertRaisesRegex(ValueError, r'lessons\[0\].*Selection completed.*E1'):
            update_memory(team,'selected',[],[lesson],['E1'])
        lesson['support_scope'] = 'workflow'
        update_memory(team,'selected',[],[lesson],['E1'])
        self.assertEqual(team['lead']['version'],1)

    def test_sampling_rejects_invalid_values_before_submission(self):
        for args in ({'diffusion_samples':0}, {'diffusion_samples':6}, {'seed':True}, {'seed':-1}, {'unknown':1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_arguments('predict_complex', args)
        validate_arguments('predict_complex', {})

    def test_cached_output_is_explicitly_not_independent(self):
        controller = object.__new__(Controller)
        controller.folder = Mock()
        controller.save = Mock()
        controller.store = Mock()
        controller.store.submit.return_value = {'job_id':'cached','reused':True}
        with patch('tetra.orchestration.workflow.backend.launch', return_value={'job_id':'cached'}):
            result = controller.dispatch(self.state(), 'predict_complex', {}, 'call')
        self.assertTrue(result['reused'])
        self.assertFalse(result['independent_biological_evidence'])
        self.assertEqual(result['backend'],'Boltz-2')
