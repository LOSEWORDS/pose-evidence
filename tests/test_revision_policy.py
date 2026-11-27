"""Controlled policy tests, not scientific performance evidence."""
import copy
import unittest
from tetra.evaluation.assessment import CHECKS
from tetra.evaluation.revision_policy import freeze_contract, evaluate_revision, verify_contract, select_with_policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.state={'candidates':{'P':{'candidate_id':'P','sha256':'parent'},'C':{'candidate_id':'C','sha256':'child'}},
                    'assessments':{},'research':{'evidence':{}},'revisions':{}}
        for key,sha in [('P','parent'),('C','child')]:
            self.state['assessments'][key]={'status':'assessed','candidate_sha256':sha,
                                          'checks':{name:'pass' for name in ('identity',*CHECKS)}}
        self.state['assessments']['P']['checks']['minimum_distance_to_protein']='fail'
        self.add_contacts('P','parent',[2,4])
        self.add_contacts('C','child',[2,4,7])
        self.plan={'revision_id':'R1','candidate_id':'P','candidate_sha256':'parent','method':'local_refinement',
                   'pocket_constraints':None,'status':'succeeded','result':{'candidate':{'candidate_id':'C'}}}
        self.plan['contract']=freeze_contract(self.state,self.plan)
        self.state['revisions']['R1']=self.plan

    def add_contacts(self,key,sha,contacts):
        self.state['research']['evidence'][key]={'observation':{'action':'measure_contacts','result':{
            'candidate_id':key,'candidate_sha256':sha,'cutoff_A':4.0,'contact_positions':contacts,
            'residues':[{'input_position':i,'minimum_distance_A':3.5 if i in contacts else 5.0} for i in range(1,9)]}}}

    def test_geometry_repair_and_preservation_accept(self):
        result=evaluate_revision(self.state,self.plan,'C')
        self.assertEqual(result['decision'],'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE')
        self.assertEqual(result['biological_validation'],'NOT_ESTABLISHED')

    def test_lost_contact_rejects_even_when_geometry_passes(self):
        self.add_contacts('C','child',[2])
        result=evaluate_revision(self.state,self.plan,'C')
        self.assertEqual(result['decision'],'REJECT')
        selected=select_with_policy(self.state)
        self.assertNotIn('C',selected['selected_ids'])

    def test_required_missing_and_stale_are_unknown(self):
        del self.state['assessments']['C']['checks']['bond_angles']
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['decision'],'UNRESOLVED')
        self.state['assessments']['C']['candidate_sha256']='stale'
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['hard_gates']['identity'],'UNKNOWN')

    def test_tamper_is_rejected(self):
        self.plan['contract']['preserve_contacts']['positions']=[]
        with self.assertRaises(ValueError):verify_contract(self.plan)

    def test_geometry_failure_can_be_repaired_but_identity_failure_cannot(self):
        self.state['assessments']['C']['checks']['minimum_distance_to_protein']='fail'
        result=evaluate_revision(self.state,self.plan,'C')
        self.assertEqual(result['decision'],'REJECT')
        self.assertTrue(result['repair_eligible'])
        self.state['assessments']['C']['checks']['identity']='fail'
        self.assertFalse(evaluate_revision(self.state,self.plan,'C')['repair_eligible'])

    def test_parent_missing_measurement_blocks_freeze(self):
        del self.state['research']['evidence']['P']
        with self.assertRaises(ValueError):freeze_contract(self.state,self.plan)

    def test_result_change_updates_selection_without_stale_acceptance(self):
        self.assertIn('C',select_with_policy(self.state)['selected_ids'])
        self.add_contacts('C','child',[])
        self.assertNotIn('C',select_with_policy(self.state)['selected_ids'])

    def test_missing_child_never_accepted(self):
        del self.state['candidates']['C']
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['decision'],'UNRESOLVED')

    def test_nonfinite_contact_measurement_is_unknown(self):
        self.state['research']['evidence']['C']['observation']['result']['residues'][0]['minimum_distance_A']=float('nan')
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['preserve_contacts'],'UNKNOWN')

    def test_evidence_origin_cannot_be_promoted_by_model_text(self):
        from tetra.research.scientist_memory import initial_team,record_observation
        memory=initial_team('test')
        key=record_observation(memory,{'action':'measure_contacts','result':{
            'origin':'independent_experiment','claim':'biologically validated'}})
        self.assertEqual(memory['evidence'][key]['origin'],'model_derived')
        self.assertFalse(memory['evidence'][key]['independent_biological_evidence'])

    def test_mutating_action_after_freeze_is_rejected(self):
        self.plan['method']='pocket_resampling'
        with self.assertRaises(ValueError):verify_contract(self.plan)

    def test_repair_cannot_relax_rejected_ancestor_preservation(self):
        self.add_contacts('C','child',[2])
        self.state['assessments']['C']['checks']['minimum_distance_to_protein']='fail'
        self.state['candidates']['D']={'candidate_id':'D','sha256':'grandchild'}
        self.state['assessments']['D']={'status':'assessed','candidate_sha256':'grandchild',
            'checks':{name:'pass' for name in ('identity',*CHECKS)}}
        self.add_contacts('D','grandchild',[2])
        second={'revision_id':'R2','candidate_id':'C','candidate_sha256':'child','method':'local_refinement',
                'pocket_constraints':None,'status':'succeeded','result':{'candidate':{'candidate_id':'D'}}}
        second['contract']=freeze_contract(self.state,second)
        self.state['revisions']['R2']=second
        self.assertEqual(evaluate_revision(self.state,second,'D')['decision'],'REJECT')
        self.assertNotIn('D',select_with_policy(self.state)['selected_ids'])

    def test_measurement_version_change_is_unknown(self):
        self.state['assessments']['P']['versions']={'posebusters':'v1'}
        self.plan['contract']=freeze_contract(self.state,self.plan)
        self.state['assessments']['C']['versions']={'posebusters':'v2'}
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['decision'],'UNRESOLVED')

    def test_frozen_pose_bound_rejects_excessive_movement(self):
        self.plan['pose_preservation'] = {'contact_positions': [2], 'max_ligand_pose_rmsd_A': 0.5}
        self.plan['contract'] = freeze_contract(self.state, self.plan)
        self.assertEqual(self.plan['contract']['preserve_contacts']['positions'], [2])
        self.add_contacts('C', 'child', [2])
        missing = evaluate_revision(self.state, self.plan, 'C')
        self.assertEqual(missing['decision'], 'UNRESOLVED')
        self.assertEqual(missing['preserve_pose'], 'UNKNOWN')

        comparison = {
            'original_id': 'P', 'original_sha256': 'parent',
            'candidate_id': 'C', 'candidate_sha256': 'child',
            'ligand_pose_comparison': {
                'status': 'measured', 'alignment': 'whole_chain_CA; no ligand fitting for pose RMSD',
                'symmetry_corrected_pose_rmsd_A': 0.7,
            },
        }
        self.state['research']['evidence']['CMP'] = {'observation': {
            'action': 'compare_structures', 'result': comparison,
        }}
        too_far = evaluate_revision(self.state, self.plan, 'C')
        self.assertEqual(too_far['decision'], 'REJECT')
        self.assertEqual(too_far['preserve_pose'], 'FAIL')
        comparison['ligand_pose_comparison']['symmetry_corrected_pose_rmsd_A'] = 0.4
        self.assertEqual(evaluate_revision(self.state, self.plan, 'C')['decision'],
                         'ACCEPTED_AS_COMPUTATIONALLY_FEASIBLE')
        comparison['candidate_sha256'] = 'stale'
        self.assertEqual(evaluate_revision(self.state, self.plan, 'C')['decision'], 'UNRESOLVED')

    def test_inherited_pose_bound_keeps_ancestor_comparison_required(self):
        from tetra.evaluation.revision_tools import required_comparisons

        self.plan['pose_preservation'] = {'contact_positions': [2], 'max_ligand_pose_rmsd_A': 0.5}
        self.plan['contract'] = freeze_contract(self.state, self.plan)
        self.state['assessments']['C']['checks']['minimum_distance_to_protein'] = 'fail'
        next_plan = {'revision_id': 'R2', 'candidate_id': 'C', 'candidate_sha256': 'child',
                     'method': 'local_refinement', 'pocket_constraints': None}
        next_plan['contract'] = freeze_contract(self.state, next_plan)
        self.assertEqual(required_comparisons(next_plan, 'D'), [('C', 'D'), ('P', 'D')])
        self.assertEqual(next_plan['contract']['inherited_plans'][0]['pose_preservation'],
                         self.plan['pose_preservation'])

    def test_pose_profile_cannot_protect_an_unmeasured_parent_contact(self):
        self.plan['pose_preservation'] = {'contact_positions': [7], 'max_ligand_pose_rmsd_A': 0.5}
        with self.assertRaisesRegex(ValueError, 'measured parent contacts'):
            freeze_contract(self.state, self.plan)

    def test_malformed_measurement_is_unknown(self):
        self.state['research']['evidence']['C']['observation']['result']['residues']=[None]
        self.assertEqual(evaluate_revision(self.state,self.plan,'C')['preserve_contacts'],'UNKNOWN')

class ConservativeRankingTests(unittest.TestCase):
    def setUp(self):
        PolicyTests.setUp(self)

    add_contacts = PolicyTests.add_contacts

    def test_v2_retains_valid_parent_and_uses_original_confidence(self):
        self.state['selection_policy_version'] = 'selection-policy-v2'
        self.state['assessments']['P']['checks']['minimum_distance_to_protein'] = 'pass'
        self.state['candidates']['P']['original_confidence'] = {'confidence_score': 0.9}
        self.state['candidates']['C']['original_confidence'] = {'confidence_score': 0.1}
        selection = select_with_policy(self.state)
        self.assertEqual(selection['selected_ids'], ['P', 'C'])
        self.assertEqual(selection['policy_version'], 'selection-policy-v2')
