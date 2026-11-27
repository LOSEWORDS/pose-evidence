import tempfile, unittest
from pathlib import Path
from tetra.research.scientist_memory import initial_team,record_observation
from tetra.research.evidence_files import persist,read_evidence,require_reads

class EvidenceFilesTests(unittest.TestCase):
    def state(self):
        state={'research':initial_team('Predict'),'evidence_access_mode':'files-v1'}
        record_observation(state['research'],{'action':'measure_contacts','result':{'minimum_distance_A':2.7}})
        return state

    def test_separate_files_read_receipts_and_no_recursive_evidence(self):
        state=self.state()
        with tempfile.TemporaryDirectory() as root:
            persist(state,root)
            self.assertTrue((Path(root)/'evidence/E1.json').exists())
            self.assertTrue((Path(root)/'memory/current.json').exists())
            with self.assertRaises(ValueError):require_reads(state,['E1'])
            data=read_evidence(state,root,['E1'])
            require_reads(state,['E1'])
            self.assertEqual(data['evidence']['E1']['observation']['result']['minimum_distance_A'],2.7)
            self.assertEqual(len(state['research']['evidence']),1)
            with self.assertRaises(ValueError):read_evidence(state,root,['../../secret'])

    def test_tampered_file_is_not_silently_rewritten(self):
        state=self.state()
        with tempfile.TemporaryDirectory() as root:
            persist(state,root)
            (Path(root)/'evidence/E1.json').write_text('{}')
            with self.assertRaises(ValueError):read_evidence(state,root,['E1'])

class FileModeIntegrationTests(unittest.TestCase):
    def test_full_result_is_available_only_after_read_and_no_recursive_read_evidence(self):
        from unittest.mock import patch
        from tetra.orchestration.scientific_agent import ScientificAgent,AgentConfig
        from tetra.runtime.store import Store
        from test_scientist_team import ScriptedClient
        client=ScriptedClient([
            ('resolve_input',{}),('read_evidence',{'evidence_ids':['E1']}),
            ('update_memory',{'working_memory':'Identity unresolved.','hypotheses':[],'lessons':[],'evidence_ids':['E1']}),
            ('stop',{'reason':'Input unavailable'})])
        with tempfile.TemporaryDirectory() as root:
            agent=ScientificAgent(Store(root),client,AgentConfig(max_steps=4,evidence_files=True))
            with patch('tetra.orchestration.workflow.resolve_request',return_value={'status':'input_error','payload':'SENTINEL_FULL_RESULT'}):
                state=agent.run({'goal':'Test explicit evidence reading'})
            self.assertNotIn('SENTINEL_FULL_RESULT',str(client.calls[1][0]))
            self.assertIn('SENTINEL_FULL_RESULT',str(client.calls[2][0]))
            self.assertEqual(state['research']['lead']['version'],1)
            self.assertEqual(state['evidence_read_ids'],['E1'])
            self.assertNotIn('read_evidence',[r['observation']['action'] for r in state['research']['evidence'].values()])
