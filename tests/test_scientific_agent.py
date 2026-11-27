"""Single scientist feedback and persistent memory behavior."""
import json
import tempfile
import unittest
from unittest.mock import patch

from tetra.orchestration.scientific_agent import ScientificAgent, AgentConfig
from tetra.runtime.store import Store
from test_scientist_team import ScriptedClient


class ScientificAgentTests(unittest.TestCase):
    def test_feedback_must_be_integrated_before_next_action(self):
        client = ScriptedClient([
            ('resolve_input', {}),
            ('predict_complex', {}),
            ('update_memory', {'working_memory': 'Identity unresolved; no prediction justified.',
                               'hypotheses': [], 'lessons': [], 'evidence_ids': ['E1']}),
            ('stop', {'reason': 'Unresolved identity'}),
        ])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientificAgent(Store(root), client)
            with patch('tetra.orchestration.workflow.resolve_request', return_value={'status': 'input_error', 'reason': 'Ambiguous'}):
                state = agent.run({'goal': 'Synthetic identity test'})
            self.assertEqual(state['prediction_count'], 0)
            self.assertEqual(state['research']['lead']['version'], 1)
            self.assertEqual(state['status'], 'stopped')
            names = [tool['function']['name'] for tool in client.calls[1][1]]
            self.assertEqual(names, ['update_memory', 'stop'])
            self.assertNotIn('delegate', [tool['function']['name'] for tool in client.calls[0][1]])
            report = json.loads((agent.folder(state) / 'report.json').read_text())
            self.assertIn('research', report)
            self.assertNotIn('team', report)

    def test_pending_action_resume_keeps_protocol_and_memory(self):
        client = ScriptedClient([
            ('resolve_input', {}),
            ('update_memory', {'working_memory': 'Identity missing', 'hypotheses': [], 'lessons': [], 'evidence_ids': ['E1']}),
            ('stop', {'reason': 'Missing identity'}),
        ])
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            agent = ScientificAgent(store, client)
            with patch.object(agent, 'apply', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    agent.run({'goal': 'Synthetic recovery test'})
            key = next(agent.root.iterdir()).name
            resumed = ScientificAgent(store, client)
            with patch('tetra.orchestration.workflow.resolve_request', return_value={'status': 'input_error'}):
                state = resumed.resume(key)
            self.assertEqual(state['research']['lead']['version'], 1)
            self.assertEqual(state['messages'][3]['role'], 'tool')

    def test_invalid_memory_does_not_change_board(self):
        client = ScriptedClient([
            ('update_memory', {'working_memory': 'Invented', 'hypotheses': [],
                               'lessons': [{'claim': 'Fake', 'scope': 'session', 'evidence_ids': ['fake']}], 'evidence_ids': []}),
            ('stop', {'reason': 'End test'}),
        ])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientificAgent(Store(root), client, AgentConfig(max_steps=2))
            state = agent.run({'goal': 'Test evidence integrity'})
            self.assertEqual(state['research']['lead']['version'], 0)
            self.assertEqual(state['research']['lead']['lessons'], [])

    def test_provenance_is_visible_without_large_sequence(self):
        client = ScriptedClient([
            ('resolve_input', {}),
            ('stop', {'reason': 'Source visibility test'}),
        ])
        with tempfile.TemporaryDirectory() as root:
            agent = ScientificAgent(Store(root), client)
            result = {'status': 'resolved', 'input_id': 'test', 'provenance': {
                'protein': {'accession': 'P00533', 'source_url': 'https://rest.uniprot.org/uniprotkb/P00533',
                            'sequence': 'SENTINEL_LARGE_SEQUENCE'}}}
            with patch('tetra.orchestration.workflow.resolve_request', return_value=result):
                agent.run({'goal': 'Test source visibility'})
            current_context = client.calls[1][0][-1]['content']
            self.assertIn('P00533', current_context)
            self.assertIn('https://rest.uniprot.org', current_context)
            self.assertNotIn('SENTINEL_LARGE_SEQUENCE', current_context)

    def test_artifact_failure_cannot_regain_policy_acceptance(self):
        from tetra.evaluation.assessment import CHECKS
        from tetra.orchestration.workflow import Controller
        with tempfile.TemporaryDirectory() as root:
            agent=ScientificAgent(Store(root),ScriptedClient([('stop',{'reason':'test'})]))
            initialize=agent._initialize
            def injected(state):
                initialize(state)
                state['candidates']={'P':{'candidate_id':'P','sha256':'hash'}}
                state['assessments']={'P':{'status':'assessed','candidate_sha256':'hash',
                    'checks':{key:'pass' for key in ('identity',*CHECKS)}}}
            with patch.object(agent,'_initialize',side_effect=injected), \
                 patch('tetra.orchestration.workflow.verify_candidate',side_effect=ValueError('changed')):
                state=agent.run({'goal':'Artifact mutation test'})
            report=json.loads((agent.folder(state)/'report.json').read_text())
            self.assertEqual(state['status'],'artifact_error')
            self.assertIsNone(state['selected'])
            self.assertEqual(report['computational_acceptance'],'NO_ACCEPTED_CANDIDATE')
