import unittest
from tetra.evaluation.selection import select_candidates

class SelectionTests(unittest.TestCase):
    def test_deterioration_keeps_parent(self):
        cs=[{'candidate_id':'a','sha256':'a'},{'candidate_id':'b','sha256':'b','parent_id':'a'}]
        aa={'a':{'candidate_sha256':'a','status':'assessed','checks':{'identity':'pass','clash':'fail','geometry':'pass'}},'b':{'candidate_sha256':'b','status':'assessed','checks':{'identity':'pass','clash':'pass','geometry':'fail'}}}
        self.assertEqual(select_candidates(cs,aa)['selected_ids'],['a'])
    def test_only_nonregressing_improvement_replaces(self):
        cs=[{'candidate_id':'a','sha256':'a'},{'candidate_id':'b','sha256':'b','parent_id':'a'}]
        aa={'a':{'candidate_sha256':'a','status':'assessed','checks':{'identity':'pass','clash':'fail'}},'b':{'candidate_sha256':'b','status':'assessed','checks':{'identity':'pass','clash':'pass'}}}
        self.assertEqual(select_candidates(cs,aa)['selected_ids'],['b'])
        aa['b']['checks']['clash']='unavailable'
        self.assertEqual(select_candidates(cs,aa)['selected_ids'],['a'])
    def test_stale_assessment_cannot_select(self):
        with self.assertRaises(ValueError):select_candidates([{'candidate_id':'a','sha256':'new'}],{'a':{'candidate_sha256':'old','checks':{}}})
