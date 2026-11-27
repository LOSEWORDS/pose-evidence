import json
from pathlib import Path
import tempfile
import unittest
from tetra.runtime.store import Store

SEQ='ACDEFGHIKLMNPQRSTVWY'
class InputsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.s=Store(self.root)
    def tearDown(self):self.tmp.cleanup()
    def add(self,**kw):
        return self.s.register_input(SEQ,'CCO',source={'kind':'user'},**kw)
    def test_invalid_sequence_and_chemistry_rejected(self):
        for seq,smi in [('ACDX','CCO'),(SEQ,'bad-smiles'),('', 'CCO'),(SEQ,'CC.O')]:
            with self.subTest(seq=seq,smi=smi), self.assertRaises(ValueError):
                self.s.register_input(seq,smi,source={'kind':'user'},confirmed=True)
    def test_undefined_stereochemistry_requires_resolution(self):
        with self.assertRaises(ValueError):self.s.register_input(SEQ,'CC(O)C(=O)O',source={'kind':'user'})
    def test_draft_cannot_submit_and_confirm_is_persistent(self):
        x=self.add()
        with self.assertRaises(ValueError):self.s.submit(x['input_id'])
        self.s.confirm_input(x['input_id'])
        j=self.s.submit(x['input_id'])
        self.assertEqual(Store(self.root).job(j['job_id'])['status'],'queued')
    def test_duplicate_submission_is_same_job(self):
        x=self.add(confirmed=True)
        a=self.s.submit(x['input_id']);b=Store(self.root).submit(x['input_id'])
        self.assertEqual(a['job_id'],b['job_id'])
        self.assertEqual(len(list((self.root/'jobs').iterdir())),1)
    def test_msa_query_mismatch_rejected(self):
        p=self.root/'wrong.a3m';p.write_text('>query\nAAAA\n')
        with self.assertRaises(ValueError):self.add(msa=str(p))
    def test_msa_is_snapshotted_and_input_mutation_detected(self):
        p=self.root/'ok.a3m';p.write_text('>query\n'+SEQ+'\n')
        x=self.add(msa=str(p),confirmed=True);p.write_text('modified')
        frozen=self.s.input(x['input_id']);self.assertIn(SEQ,Path(frozen['msa']).read_text())
        Path(frozen['msa']).write_text('tampered')
        with self.assertRaises(ValueError):self.s.submit(x['input_id'])
    def test_config_bounds_and_unknown_keys_rejected(self):
        x=self.add(confirmed=True)
        for config in [{'diffusion_samples':0},{'gpu':0},{'timeout_seconds':0},{'arbitrary':'x'}]:
            with self.subTest(config=config),self.assertRaises(ValueError):self.s.submit(x['input_id'],config)
    def test_unsupported_isotopes_radicals_and_wildcards_rejected(self):
        for smiles in ['[13CH3]CO','[CH2]CO','*CC']:
            with self.subTest(smiles=smiles),self.assertRaises(ValueError):
                self.s.register_input(SEQ,smiles,source={'kind':'user'},confirmed=True)
    def test_official_csv_msa_preserved(self):
        p=self.root/'official.csv';p.write_text('key,sequence\n0,'+SEQ+'\n-1,'+SEQ+'\n')
        x=self.add(msa=p,confirmed=True)
        self.assertEqual(Path(x['msa']).suffix,'.csv')
        self.assertEqual(Path(x['msa']).read_bytes(),p.read_bytes())
        self.assertEqual(x['msa_format'],'csv')
    def test_official_sampling_budget(self):
        x=self.add(confirmed=True)
        j=self.s.submit(x['input_id'],{'diffusion_samples':5,'recycling_steps':10})
        self.assertEqual(j['config']['diffusion_samples'],5)
    def test_benchmark_preserves_explicit_unspecified_stereo(self):
        from tetra.runtime.inputs import validate
        with self.assertRaises(ValueError):validate('ACDE','CC(O)C(=O)O')
        result,_=validate('ACDE','CC(O)C(=O)O',allow_unspecified_stereo=True)
        self.assertEqual(result['smiles'],'CC(O)C(=O)O')
