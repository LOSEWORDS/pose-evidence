import unittest

from tetra.research.literature import search_literature, structural_followup
from tetra.orchestration.scientific_agent import ScientificAgent


class RetrievalReadinessTests(unittest.TestCase):
    def test_rank_structural_family_hit_above_disease_and_other_family(self):
        def fetch(_parameters):
            return {'resultList': {'result': [
                {'source': 'MED', 'id': '1', 'title': 'Cathepsin and disease risk'},
                {'source': 'MED', 'id': '2', 'title': 'Coronavirus protease crystal structure inhibitor complex'},
                {'source': 'MED', 'id': '3', 'title': 'Cathepsin crystal structure with bound inhibitor'},
            ]}}
        result = search_literature('cathepsin inhibitors', family_terms=['cathepsin'], fetch=fetch)
        self.assertEqual(result['results'][0]['source_id'], 'MED:3')

    def test_fallback_keeps_family(self):
        queries = []
        def fetch(parameters):
            queries.append(parameters['query'])
            return {'resultList': {'result': []}}
        search_literature('cathepsin peptide crystal structure', broader_query='inhibitor complex',
                          family_terms=['cathepsin'], fetch=fetch)
        self.assertIn('cathepsin', queries[-1])

    def test_failed_details_offer_only_current_unread_sources(self):
        rows = [
            {'action': 'search_literature', 'result': {'results': [{'source_id': 'MED:old'}]}},
            {'action': 'read_literature', 'arguments': {'source_id': 'MED:old'},
             'result': {'source_id': 'MED:old', 'status': 'retrieved'}},
            {'action': 'read_structure_details', 'result': {'source_id': 'MED:old', 'status': 'unavailable'}},
            {'action': 'search_literature', 'result': {'results': [{'source_id': 'MED:new'}]}},
        ]
        schema = ScientificAgent.literature_read_schema({'observations': rows})
        self.assertEqual(schema['function']['parameters']['properties']['source_id']['enum'], ['MED:new'])
        self.assertEqual(structural_followup(rows)[0], 'read_literature')
        rows.append({'action': 'search_literature', 'result': {'results': []}})
        self.assertEqual(structural_followup(rows)[0], 'search_literature')

    def test_reference_pocket_preserves_mapping_and_omits_unmapped_residues(self):
        from tetra.research.literature_structures import pocket_drafts
        references = {'entries': [{'pdb_id': '1ABC', 'coordinate_sha256': 'test',
                                   'measurements': [{'chain': 'C', 'reference_ligand': 'XYZ',
                                                     'contacts': [{'target_position': 7},
                                                                  {'target_position': None}]}]}]}
        draft, = pocket_drafts(references)
        self.assertEqual(draft['target_positions'], [7])
        self.assertEqual(draft['pdb_id'], '1ABC')
        self.assertNotIn('contact', draft)

    def test_sequence_search_screens_entries_before_exposing_citations(self):
        from tetra.research.literature_structures import search_structure_citations

        class Response:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {'result_set': [{'identifier': '1AAA_1'}, {'identifier': '2AAA_1'}]}

        def screen(pdb_id, _target_connectivity):
            if pdb_id == '1AAA':
                raise ValueError('Target ligand')
            return ['XYZ']

        result = search_structure_citations(
            {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'smiles': 'CCO'},
            fetch=lambda *_args, **_kwargs: Response(), screen=screen,
            metadata=lambda _path: {'rcsb_primary_citation': {'pdbx_database_id_PubMed': 123}})
        self.assertEqual(result['results'][0]['source_id'], 'MED:123')
        self.assertEqual(result['results'][0]['pdb_ids'], ['2AAA'])
        self.assertEqual(result['excluded'][0]['pdb_id'], '1AAA')

    def test_failed_reads_do_not_stall_the_current_batch(self):
        rows = [
            {'action': 'read_literature', 'result': {'source_id': 'MED:old', 'status': 'retrieved'}},
            {'action': 'read_structure_details', 'result': {'source_id': 'MED:old', 'status': 'unavailable'}},
            {'action': 'search_literature', 'result': {'results': [{'source_id': 'MED:new'}]}},
            {'action': 'read_literature', 'arguments': {'source_id': 'MED:new'},
             'result': {'source_id': 'MED:new', 'status': 'unavailable'}},
        ]
        self.assertEqual(structural_followup(rows)[0], 'search_literature')

    def test_reference_distances_cannot_alone_support_an_atom_contact(self):
        from tetra.prediction.prediction_experiments import require_mapped_reference_support

        cited = [{'pocket_drafts': [{'target_positions': [7, 8]}]}]
        require_mapped_reference_support([{'pocket': {'contacts': [['A', 7]]}}], cited)
        with self.assertRaisesRegex(ValueError, 'input residue'):
            require_mapped_reference_support([{'pocket': {'contacts': [['A', 2]]}}], cited)
        with self.assertRaisesRegex(ValueError, 'ligand atom'):
            require_mapped_reference_support([{'contact': {}}], cited)

    def test_v14_search_arguments_match_runtime_schema(self):
        from tetra.orchestration.tool_registry import validate_arguments

        args = {
            'query': 'cathepsin cysteine protease CGSCW active site motif inhibitor complex structure',
            'broader_query': 'cysteine protease papain family inhibitor binding site structure',
            'family_terms': ['cathepsin', 'papain-like cysteine protease', 'C1 family'],
        }
        validate_arguments('search_literature', args)
        validate_arguments('search_literature', {'query': 'comparable structures', 'route': 'sequence'})
        for invalid in ('cathepsin', [''], [1], ['a'] * 6):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'family_terms'):
                validate_arguments('search_literature', {**args, 'family_terms': invalid})
        with self.assertRaisesRegex(ValueError, 'route'):
            validate_arguments('search_literature', {**args, 'route': 'unknown'})
