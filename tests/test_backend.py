import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from tetra.runtime.store import Store
from tetra.prediction.backend import execute, inspect, run_job, status

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'runs/smoke-20261116T110620Z/boltz_results_smoke_complex/predictions/smoke_complex'
class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.s=Store(self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_process_error_and_timeout_are_explicit(self):
        r=execute([sys.executable,'-c','raise SystemExit(7)'],self.root,10,{})
        self.assertEqual(r['exit_code'],7)
        r=execute([sys.executable,'-c','import time; time.sleep(20)'],self.root,0.1,{})
        self.assertTrue(r['timed_out']);self.assertNotEqual(r['exit_code'],0)
    def test_exit_zero_without_artifacts_is_failed(self):
        x=self.s.register_input('ACDEFGHIK','CCO',source={'kind':'user'},confirmed=True)
        j=self.s.submit(x['input_id'])
        with patch('tetra.prediction.backend.command_for',return_value=[sys.executable,'-c','pass']),patch('tetra.prediction.backend.ensure_gpu_available'):
            run_job(self.s,j['job_id'])
        self.assertEqual(self.s.job(j['job_id'])['status'],'failed')
        self.assertIn('artifact',self.s.job(j['job_id'])['error'].lower())
    @unittest.skipUnless(FIXTURE.exists(),'Real smoke fixture is local only')
    def test_real_artifact_passes_and_changed_identity_fails(self):
        import yaml
        data=yaml.safe_load((ROOT/'inputs/smoke_complex.yaml').read_text())
        record={'sequence':data['sequences'][0]['protein']['sequence'],'smiles':data['sequences'][1]['ligand']['smiles']}
        shutil.copytree(FIXTURE,self.root/'predictions')
        report=inspect(self.root,record)
        self.assertTrue(report['checks']['protein_sequence_matches'])
        record['sequence']='AAAA'
        with self.assertRaises(ValueError):inspect(self.root,record)
    def test_dead_worker_becomes_interrupted_without_resubmit(self):
        x=self.s.register_input('ACDEFGHIK','CCO',source={'kind':'user'},confirmed=True)
        j=self.s.submit(x['input_id']);self.s.update_job(j['job_id'],'running',worker_pid=99999999,worker_started=0)
        self.assertEqual(status(self.s,j['job_id'])['status'],'interrupted')
        self.assertEqual(self.s.submit(x['input_id'])['job_id'],j['job_id'])
    @unittest.skipUnless(FIXTURE.exists(),'Real smoke fixture is local only')
    def test_missing_protein_sidechain_atoms_rejected(self):
        import gemmi,yaml
        data=yaml.safe_load((ROOT/'inputs/smoke_complex.yaml').read_text())
        record={'sequence':data['sequences'][0]['protein']['sequence'],'smiles':data['sequences'][1]['ligand']['smiles']}
        target=self.root/'predictions';shutil.copytree(FIXTURE,target)
        path=next(target.glob('*_model_0.cif'));st=gemmi.read_structure(str(path))
        for residue in st[0]['A']:
            for index in reversed(range(len(residue))):
                if residue[index].name not in ['N','CA','C','O']:del residue[index]
        st.make_mmcif_document().write_file(str(path))
        with self.assertRaises(ValueError):inspect(self.root,record)
