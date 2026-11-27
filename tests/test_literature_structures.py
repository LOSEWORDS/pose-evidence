import unittest
from unittest.mock import patch

from tetra.research.literature_structures import _residue_mapping, reference_components


class LiteratureStructureTests(unittest.TestCase):
    def test_only_stable_residue_positions_are_mapped(self):
        mapping, identity, coverage = _residue_mapping('ACCD', 'ACCCD')
        self.assertEqual(mapping, {0: 1, 3: 5})
        self.assertEqual(identity, 1.0)
        self.assertEqual(coverage, 0.8)

    def test_any_matching_target_component_excludes_entry(self):
        prefix = 'ABCDEFGHIJKLMN'
        data = [
            {'rcsb_entry_container_identifiers': {'non_polymer_entity_ids': ['2', '3']}},
            {'pdbx_entity_nonpoly': {'comp_id': 'A01'}},
            {'pdbx_entity_nonpoly': {'comp_id': 'A02'}},
            {'rcsb_chem_comp_descriptor': {'InChIKey': 'ZZZZZZZZZZZZZZ-ABCDEFGHIJ-A'},
             'rcsb_chem_comp_info': {'atom_count_heavy': 20}},
            {'rcsb_chem_comp_descriptor': {'InChIKey': prefix + '-ABCDEFGHIJ-A'}},
        ]
        with patch('tetra.research.literature_structures._metadata', side_effect=data):
            with self.assertRaisesRegex(ValueError, 'target ligand connectivity'):
                reference_components('1ABC', prefix)
