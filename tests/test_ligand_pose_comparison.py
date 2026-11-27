import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tetra.evaluation.structural_metrics import ligand_pose_comparison

class PoseComparisonTests(unittest.TestCase):
    def molecule(self):
        m = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
        AllChem.EmbedMolecule(m, randomSeed=42)
        return Chem.RemoveHs(m)

    def test_atom_order_symmetry(self):
        m = self.molecule()
        n = Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms()))))
        result = ligand_pose_comparison(m, n, np.eye(3), np.zeros(3))
        self.assertAlmostEqual(result['symmetry_corrected_pose_rmsd_A'], 0, places=6)

    def test_translation_is_pose_not_internal_change(self):
        m = self.molecule()
        result = ligand_pose_comparison(m, m, np.eye(3), np.array([4., 0., 0.]))
        self.assertAlmostEqual(result['symmetry_corrected_pose_rmsd_A'], 4, places=6)
        self.assertAlmostEqual(result['internal_conformation_rmsd_A'], 0, places=6)

    def test_truncated_symmetry_is_unavailable(self):
        m = self.molecule()
        self.assertEqual(ligand_pose_comparison(m, m, np.eye(3), np.zeros(3), max_matches=1)['status'], 'unavailable')
