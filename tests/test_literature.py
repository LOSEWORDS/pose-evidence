import unittest

from tetra.research.literature import read_literature, search_literature


class LiteratureTests(unittest.TestCase):
    def test_search_does_not_claim_abstract_was_read(self):
        article = {'source': 'MED', 'id': '123', 'title': 'Example',
                   'abstractText': 'Original abstract'}
        def fetch(parameters):
            return {'resultList': {'result': [article]}}
        hit, = search_literature('binding', fetch=fetch)['results']
        self.assertEqual(hit['source_id'], 'MED:123')
        self.assertNotIn('text', hit)
        self.assertEqual(read_literature(hit['source_id'], fetch=fetch)['text'],
                         'Original abstract')

    def test_missing_abstract_is_unavailable(self):
        def fetch(parameters):
            return {'resultList': {'result': [{'source': 'MED', 'id': '123'}]}}
        result = read_literature('MED:123', fetch=fetch)
        self.assertEqual(result['status'], 'unavailable')
        self.assertNotIn('text', result)

    def test_fetch_failure_propagates(self):
        def fetch(parameters):
            raise TimeoutError('timeout')
        with self.assertRaises(TimeoutError):
            search_literature('binding', fetch=fetch)

    def test_initial_stall_uses_consecutive_empty_searches(self):
        from tetra.research.literature import initial_search_stalled

        hit = {'action': 'search_literature', 'result': {'results': [{'source_id': 'MED:1'}]}}
        empty = {'action': 'search_literature', 'result': {'results': []}}
        failure = {'action': 'search_literature', 'result': {'error': 'timeout'}}
        self.assertFalse(initial_search_stalled([hit, empty]))
        self.assertTrue(initial_search_stalled([hit, empty, failure]))
        self.assertFalse(initial_search_stalled([empty, failure, hit]))
    def test_zero_hits_tries_and_records_the_broader_query(self):
        queries = []

        def fetch(parameters):
            queries.append(parameters['query'])
            rows = [{'source': 'MED', 'id': '123'}] if parameters['query'] == 'family binding' else []
            return {'resultList': {'result': rows}}

        result = search_literature('specific compound family binding',
                                   broader_query='family binding', fetch=fetch)
        self.assertEqual(queries, ['specific compound family binding', 'family binding'])
        self.assertEqual(result['matched_query'], 'family binding')
        self.assertEqual(result['results'][0]['source_id'], 'MED:123')
        self.assertEqual([row['hit_count'] for row in result['attempts']], [0, 1])

    def test_structural_followup_retrieves_matching_figure_with_provenance(self):
        from unittest.mock import patch
        from tetra.research.literature import read_structure_details

        sections = [
            {'section': 'Methods', 'text': 'Samples were prepared at room temperature.'},
            {'section': 'Figure 2', 'text': 'The ligand contacts the S2 pocket at 4 Å.'},
        ]
        with patch('tetra.research.literature._pmc_sections', return_value=sections):
            result = read_structure_details({'source_id': 'MED:123', 'pmcid': 'PMC123'},
                                            ['S2 pocket'])
        self.assertEqual(result['status'], 'retrieved')
        self.assertEqual(result['passages'], [sections[1]])
        self.assertEqual(result['url'], 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/')

    def test_abstract_only_cannot_be_used_to_skip_structural_followup(self):
        from tetra.research.literature import structural_followup_blocker

        def read(source_id):
            return {'action': 'read_literature', 'arguments': {'source_id': source_id},
                    'result': {'source_id': source_id, 'status': 'retrieved'}}
        def details(source_id, status):
            return {'action': 'read_structure_details',
                    'result': {'source_id': source_id, 'status': status}}

        history = [read('MED:1')]
        self.assertIn('read_structure_details', structural_followup_blocker(history))
        history.append(details('MED:1', 'unavailable'))
        self.assertIn('comparable structural study', structural_followup_blocker(history))
        history.append({'action': 'search_literature',
                        'result': {'results': [{'source_id': 'MED:2'}]}})
        self.assertIn('Read a new hit', structural_followup_blocker(history))
        history.append(read('MED:2'))
        self.assertIn('newly screened', structural_followup_blocker(history))
        history.append(details('MED:2', 'retrieved'))
        self.assertIsNone(structural_followup_blocker(history))
