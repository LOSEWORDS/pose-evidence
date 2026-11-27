import unittest
from tetra.evaluation.external_science import parse_gnina, parse_fpocket_info, rank_scored_candidates

class ExternalScienceTests(unittest.TestCase):
    def test_score_rejects_missing_nonfinite_or_duplicate(self):
        for text in ('Affinity: -7', 'CNNscore: nan', 'CNNscore: 0.3\nCNNscore: 0.4', 'CNNscore: 1.2'):
            with self.assertRaises(ValueError):parse_gnina(text)
        self.assertEqual(parse_gnina('Affinity: -7.0\nCNNscore: 0.75\nCNNaffinity: 5.2')['cnn_score'],.75)

    def test_pocket_descriptors_preserve_identity(self):
        result=parse_fpocket_info('Pocket 1 :\n\tScore : 0.42\n\tDruggability Score : 0.19\nPocket 2 :\n\tScore : 0.1\n')
        self.assertEqual(result[1]['Score'],.42)
        self.assertEqual(result[2]['Score'],.1)

    def test_rank_requires_complete_comparable_fresh_scores(self):
        candidates={'a':{'sha256':'A'},'b':{'sha256':'B'}}
        scores={'a':{'candidate_sha256':'A','profile':'fixed','cnn_score':.1},
                'b':{'candidate_sha256':'B','profile':'fixed','cnn_score':.8}}
        self.assertEqual(rank_scored_candidates(['a','b'],candidates,scores),['b','a'])
        scores['b']['candidate_sha256']='stale'
        self.assertIsNone(rank_scored_candidates(['a','b'],candidates,scores))
        scores['b']['candidate_sha256']='B';scores['b']['profile']='other-model'
        self.assertIsNone(rank_scored_candidates(['a','b'],candidates,scores))

class ScientificToolIntegrationTests(unittest.TestCase):
    def test_score_ranking_does_not_bypass_physical_gates(self):
        from tetra.evaluation.assessment import CHECKS
        from tetra.evaluation.revision_policy import select_with_policy
        state={'selection_policy_version':'selection-policy-v3','revisions':{},'candidates':{
            'a':{'sha256':'A','original_confidence':{'confidence_score':.9}},
            'b':{'sha256':'B','original_confidence':{'confidence_score':.3}}},
            'assessments':{},'pose_scores':{
                'a':{'candidate_sha256':'A','profile':'fixed','cnn_score':.1},
                'b':{'candidate_sha256':'B','profile':'fixed','cnn_score':.8}}}
        for key in state['candidates']:
            state['assessments'][key]={'status':'assessed','candidate_sha256':key.upper(),
                'checks':dict.fromkeys(('identity',*CHECKS),'pass')}
        self.assertEqual(select_with_policy(state)['selected_ids'],['b','a'])
        state['assessments']['b']['checks']['identity']='fail'
        self.assertEqual(select_with_policy(state)['selected_ids'],['a'])
        state['assessments']['b']['checks']['identity']='pass'
        del state['pose_scores']['b']
        self.assertEqual(select_with_policy(state)['selected_ids'],['a','b'])
        self.assertIn('incomplete', select_with_policy(state)['ranking_basis'])

class FrozenToolTests(unittest.TestCase):
    def test_frozen_tool_availability_cannot_change_mid_experiment(self):
        from unittest.mock import patch
        from tetra.evaluation.external_science import validate_tool_snapshot
        original={'gnina':{'available':False},'fpocket':{'available':False}}
        changed={'gnina':{'available':True,'binary_sha256':'new'},'fpocket':{'available':False}}
        with patch('tetra.evaluation.external_science.tool_snapshot',return_value=changed):
            with self.assertRaises(ValueError):validate_tool_snapshot(original)
