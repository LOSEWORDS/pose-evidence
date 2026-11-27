import unittest
from copy import deepcopy
from tetra.evaluation.assessment import CHECKS
from tetra.evaluation.revision_policy import select_with_policy


class QualifiedSelectionTests(unittest.TestCase):
    def state(self):
        return {'selection_policy_version':'selection-policy-v4','candidates':{
            'P':{'candidate_id':'P','sha256':'hash','original_confidence':{'confidence_score':.9}}},
            'assessments':{'P':{'status':'assessed','candidate_sha256':'hash','checks':{
                **{name:'pass' for name in ('identity',*CHECKS)},'minimum_distance_to_protein':'fail'}}},
            'revisions':{},'pose_scores':{},'research':{'evidence':{}}}

    def test_clash_allows_explicitly_qualified_original_delivery(self):
        result=select_with_policy(self.state())
        self.assertEqual(result['selected_ids'],['P'])
        self.assertEqual(result['delivery_qualification'],'prediction_with_limitations')
        detail=result['candidate_decisions']['P']
        self.assertEqual(detail['physical_quality'],'FAIL')
        self.assertEqual(detail['failed_checks'],['minimum_distance_to_protein'])
        self.assertEqual(detail['computational_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(result['ranking_basis'],'single_deliverable_candidate; no ranking required')

    def test_identity_failure_or_stale_assessment_never_qualifies(self):
        for change in ('identity','stale','missing'):
            state=self.state()
            if change=='identity': state['assessments']['P']['checks']['identity']='fail'
            elif change=='stale': state['assessments']['P']['candidate_sha256']='other'
            else: del state['assessments']['P']['checks']['bond_angles']
            result=select_with_policy(state)
            self.assertEqual(result['selected_ids'],[])
            self.assertTrue(result['candidate_decisions']['P']['blocking_reasons'])
            self.assertNotIn('GNINA',result['ranking_basis'])

    def test_failed_revision_cannot_be_relabelled_qualified_success(self):
        from test_revision_policy import PolicyTests
        fixture=PolicyTests();fixture.setUp();fixture.add_contacts('C','child',[2])
        state=fixture.state;state['selection_policy_version']='selection-policy-v4'
        state['candidates']['C']['parent_id']='P'
        result=select_with_policy(state)
        self.assertEqual(result['selected_ids'],['P'])
        self.assertEqual(result['revision_verdicts']['C']['decision'],'REJECT')
        self.assertIn('revision_contract_not_met',result['candidate_decisions']['C']['blocking_reasons'])

    def test_v3_remains_unchanged(self):
        state=self.state();state['selection_policy_version']='selection-policy-v3'
        self.assertEqual(select_with_policy(state)['selected_ids'],[])

    def test_qualified_delivery_exposes_defects_in_manifest_and_readable_record(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tetra.evaluation.assessment import fingerprint
        from tetra.evaluation.delivery import build_delivery
        from tetra.research.scientist_memory import initial_team
        state = self.state()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receptor, ligand = root / 'source.pdb', root / 'source.sdf'
            receptor.write_text('test receptor')
            ligand.write_text('test ligand')
            candidate = state['candidates']['P']
            candidate.update(receptor=str(receptor), ligand=str(ligand), sha256=fingerprint(receptor, ligand))
            state['assessments']['P']['candidate_sha256'] = candidate['sha256']
            state.update(goal='Predict', status='finished', research=initial_team('Predict'))
            state['selected'] = select_with_policy(state)
            with patch('tetra.evaluation.delivery.selected_complex', side_effect=lambda c, p: p.write_text('test CIF')):
                result = build_delivery(state, root)
            self.assertEqual(result['delivery_status'], 'prediction_with_limitations')
            self.assertEqual(result['goal_satisfaction'], 'UNRESOLVED')
            self.assertEqual(result['artifacts'][0]['assessment']['computational_acceptance'], 'NOT_ACCEPTED')
            self.assertIn('minimum_distance_to_protein', (root / 'delivery/rounds.md').read_text())
            self.assertTrue(Path(result['artifacts'][0]['files']['complex']['path']).exists())
