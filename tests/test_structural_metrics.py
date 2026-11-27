import tempfile
import unittest
from pathlib import Path
import numpy as np
from tetra.evaluation.structural_metrics import align_points, sequence_residues
from tetra.prediction.constraints import validate_pockets
from tetra.prediction import backend
from tetra.runtime.inputs import validate_config


class StructuralMetricTests(unittest.TestCase):
    def test_rigid_alignment(self):
        reference = np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
        rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        moved = reference @ rotation + 7
        aligned, _, _ = align_points(moved, reference)
        np.testing.assert_allclose(aligned, reference, atol=1e-10)

    def test_sequence_numbering_is_explicit(self):
        import gemmi
        structure = gemmi.Structure()
        model = gemmi.Model('1')
        chain = gemmi.Chain('X')
        for name, index, insertion in [('ALA',40,' '),('GLY',40,'A')]:
            residue = gemmi.Residue();residue.name=name;residue.seqid=gemmi.SeqId(index,insertion)
            chain.add_residue(residue)
        model.add_chain(chain);structure.add_model(model)
        rows = sequence_residues(structure, 'AG')
        self.assertEqual(rows[1]['input_position'], 2)
        self.assertEqual(rows[1]['pdb_insertion_code'], 'A')
        with self.assertRaises(ValueError):sequence_residues(structure, 'GA')

    def test_constraints_validate_and_change_payload(self):
        constraints = [{'binder':'B','contacts':[['A',2]],'max_distance':6.0,'force':True}]
        config = validate_config({'pocket_constraints':constraints})
        record={'sequence':'AG','smiles':'CC','msa':None}
        payload=backend.input_payload(record, config.get('pocket_constraints'))
        self.assertEqual(payload['constraints'][0]['pocket'],constraints[0])
        self.assertNotIn('pocket_constraints',validate_config())
        with self.assertRaises(ValueError):
            backend.input_payload({'sequence':'A','smiles':'CC'},constraints)
        for invalid in [float('nan'),3,21]:
            with self.assertRaises(ValueError):
                validate_pockets([dict(constraints[0],max_distance=invalid)])
        with self.assertRaises(ValueError):
            validate_pockets([dict(constraints[0],contacts=[['A',True]])])

    def test_constraint_changes_have_distinct_job_fingerprints(self):
        from tetra.runtime.store import Store
        with tempfile.TemporaryDirectory() as root:
            store=Store(root)
            item=store.register_input('AG','CC',source={'kind':'unit-test'},confirmed=True)
            first=[{'binder':'B','contacts':[['A',1]],'max_distance':6.0,'force':True}]
            second=[{'binder':'B','contacts':[['A',2]],'max_distance':6.0,'force':True}]
            a=store.submit(item['input_id'],{'pocket_constraints':first})
            b=store.submit(item['input_id'],{'pocket_constraints':second})
            c=store.submit(item['input_id'],{'pocket_constraints':first})
            self.assertNotEqual(a['job_id'],b['job_id'])
            self.assertEqual(a['job_id'],c['job_id'])
            self.assertTrue(c['reused'])
