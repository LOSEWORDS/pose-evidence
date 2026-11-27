"""Board integrity and orchestration tests; scripted clients are not scientific evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tetra.research.scientist_memory import (
    initial_team, delegate, record_observation, report_task, update_memory,
)
from tetra.orchestration.scientist_team import ScientistTeam, ScientistConfig
from tetra.runtime.store import Store


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.team = initial_team('Inspect a candidate')

    def test_snapshot_and_unknown_evidence(self):
        task = delegate(self.team, 'analysis', 'Measure clashes', ['assess_structure'], 2)
        self.team['lead']['working_memory'] = 'Later state'
        self.assertEqual(task['lead_snapshot']['working_memory'], '')
        with self.assertRaises(ValueError):
            report_task(self.team, 'Done', ['invented'])
        self.assertEqual(self.team['active_task'], task['id'])

    def test_feedback_memory_requires_actual_evidence(self):
        delegate(self.team, 'analysis', 'Measure clashes', ['assess_structure'], 2)
        eid = record_observation(self.team, {'action': 'assess_structure', 'result': {'clash': 'fail'}})
        report_task(self.team, 'A geometry issue was observed', [eid])
        before = copy.deepcopy(self.team)
        with self.assertRaises(ValueError):
            update_memory(self.team, 'Bad update', [], [{'claim': 'Always refine', 'scope': 'all', 'evidence_ids': ['fake']}], [eid])
        self.assertEqual(self.team, before)
        update_memory(self.team, 'Assess possible local correction', [],
                      [{'claim': 'Check geometry before selection', 'scope': 'this session', 'evidence_ids': [eid]}], [eid])
        self.assertFalse(self.team['review_required'])
        self.assertEqual(len(self.team['lead']['history']), 1)
        self.assertEqual(self.team['lead']['lessons'][0]['evidence_ids'], [eid])


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        name, args = next(self.actions)
        return {'message': {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_' + str(len(self.calls)), 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'usage': {}}


class TeamTests(unittest.TestCase):
    def test_role_permissions_and_feedback_review(self):
        actions = [
            ('delegate', {'scientist': 'evidence', 'objective': 'Resolve input', 'tools': ['resolve_input'], 'max_steps': 2}),
            ('assess_structure', {'candidate_id': 'made_up'}),
            ('resolve_input', {}),
            ('update_memory', {'working_memory': 'Input identity is unresolved; stop.', 'hypotheses': [], 'lessons': [], 'evidence_ids': ['E1', 'E2']}),
            ('stop', {'reason': 'Unresolved input'}),
        ]
        with tempfile.TemporaryDirectory() as root:
            agent = ScientistTeam(Store(root), ScriptedClient(actions))
            with patch('tetra.orchestration.workflow.resolve_request', return_value={'status': 'input_error', 'reason': 'Ambiguous'}) as resolve:
                state = agent.run({'goal': 'Resolve a synthetic ambiguous name', 'input': {}})
            self.assertEqual(resolve.call_count, 1)
            self.assertFalse(state['task_complete'])
            self.assertEqual(state['team']['tasks']['T1']['status'], 'budget_exhausted')
            self.assertIn('unresolved', state['team']['lead']['working_memory'])
            saved = json.loads((agent.folder(state) / 'session.json').read_text())
            self.assertEqual(saved['team'], state['team'])
            self.assertEqual(len(state['team']['evidence']), 3)
            self.assertEqual(state['team']['evidence']['E3']['observation']['action'], 'stop')

    def test_wrong_role_delegate_and_budget(self):
        client = ScriptedClient([('delegate', {'scientist': 'evidence', 'objective': 'Run GPU', 'tools': ['predict_complex'], 'max_steps': 1})])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientistTeam(Store(root), client, ScientistConfig(max_steps=1))
            state = agent.run({'goal': 'Test permissions'})
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertIsNone(state['team']['active_task'])
            self.assertEqual(state['prediction_count'], 0)

    def test_service_error_is_durable(self):
        class Broken:
            def complete(self, *args):
                raise RuntimeError('credential should not be logged')
        with tempfile.TemporaryDirectory() as root:
            agent = ScientistTeam(Store(root), Broken())
            state = agent.run({'goal': 'Test failure'})
            self.assertEqual(state['status'], 'service_error')
            report = (agent.folder(state) / 'report.json').read_text()
            self.assertNotIn('credential should not', report)
            self.assertIn('team', report)

    def test_prediction_resume_returns_feedback_to_modeling_board(self):
        actions = [
            ('delegate', {'scientist': 'evidence', 'objective': 'Resolve', 'tools': ['resolve_input'], 'max_steps': 1}),
            ('resolve_input', {}),
            ('update_memory', {'working_memory': 'Ready to predict', 'hypotheses': [], 'lessons': [], 'evidence_ids': ['E1']}),
            ('delegate', {'scientist': 'modeling', 'objective': 'Predict', 'tools': ['predict_complex'], 'max_steps': 1}),
            ('predict_complex', {}),
            ('update_memory', {'working_memory': 'Prediction finished; insufficient exported candidates in this synthetic test', 'hypotheses': [], 'lessons': [], 'evidence_ids': ['E2', 'E3']}),
            ('stop', {'reason': 'Synthetic resume test complete'}),
        ]
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            client = ScriptedClient(actions)
            agent = ScientistTeam(store, client)
            with patch('tetra.orchestration.workflow.resolve_request', return_value={'input_id': 'input', 'status': 'resolved'}), \
                 patch.object(store, 'submit', return_value={'job_id': 'job'}), \
                 patch('tetra.orchestration.workflow.backend.launch', return_value={'status': 'running'}):
                state = agent.run({'goal': 'Synthetic asynchronous test'})
            self.assertEqual(state['status'], 'waiting_prediction')
            reloaded = ScientistTeam(store, client)
            with patch('tetra.orchestration.workflow.backend.status', return_value={'status': 'succeeded'}), \
                 patch.object(reloaded, 'import_prediction'):
                state = reloaded.resume(state['session_id'])
            self.assertEqual(state['status'], 'stopped')
            self.assertEqual(state['team']['boards']['modeling'][0]['evidence_ids'], ['E2', 'E3'])
            self.assertEqual(state['team']['lead']['version'], 2)
            modeling = state['team']['tasks']['T2']['messages']
            self.assertEqual(modeling[-1]['role'], 'tool')

    def test_pending_action_restores_specialist_protocol(self):
        actions = [
            ('delegate', {'scientist': 'evidence', 'objective': 'Resolve', 'tools': ['resolve_input'], 'max_steps': 3}),
            ('resolve_input', {}),
            ('report_task', {'summary': 'Identity unresolved', 'evidence_ids': ['E1']}),
            ('update_memory', {'working_memory': 'Cannot resolve', 'hypotheses': [], 'lessons': [], 'evidence_ids': ['E1']}),
            ('stop', {'reason': 'Unresolved'}),
        ]
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            client = ScriptedClient(actions)
            agent = ScientistTeam(store, client)
            with patch.object(agent, 'apply', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    agent.run({'goal': 'Interrupt before tool execution'})
            key = next(agent.root.iterdir()).name
            reloaded = ScientistTeam(store, client)
            with patch('tetra.orchestration.workflow.resolve_request', return_value={'status': 'input_error', 'reason': 'Ambiguous'}):
                state = reloaded.resume(key)
            self.assertEqual(state['team']['lead']['version'], 1)
            history = state['team']['tasks']['T1']['messages']
            self.assertEqual(history[0]['role'], 'assistant')
            self.assertEqual(history[1]['role'], 'tool')
            self.assertEqual(state['llm_calls'], 5)

    def test_completed_action_checkpoint_includes_role_feedback(self):
        client = ScriptedClient([
            ('delegate', {'scientist': 'evidence', 'objective': 'Resolve', 'tools': ['resolve_input'], 'max_steps': 3}),
            ('resolve_input', {}),
        ])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientistTeam(Store(root), client)
            save = agent.save
            def interrupt_after_observation(state):
                save(state)
                if state['observations'] and not state.get('pending_call'):
                    raise KeyboardInterrupt()
            with patch.object(agent, 'save', side_effect=interrupt_after_observation), \
                 patch('tetra.orchestration.workflow.resolve_request', return_value={'status': 'input_error'}):
                with self.assertRaises(KeyboardInterrupt):
                    agent.run({'goal': 'Atomic completion test'})
            path = next(agent.root.iterdir()) / 'session.json'
            state = json.loads(path.read_text())
            self.assertEqual(state['status'], 'running')
            self.assertEqual(len(state['team']['evidence']), 1)
            self.assertEqual(state['team']['tasks']['T1']['messages'][-1]['role'], 'tool')
