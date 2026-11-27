"""Evidence-bound scientific revision plans and real structural actions."""
import json
from tetra.prediction import backend
from tetra.evaluation import revision_policy
from tetra.evaluation.assessment import verify_candidate
from tetra.prediction.constraints import validate_pockets
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest, stable
from tetra.evaluation.structural_metrics import measure_contacts, compare_structures
from tetra.orchestration.tool_registry import current_assessment

TEXT = {'type': 'string'}
PARAMETERS = {
    'read_evidence': {'evidence_ids': {'type': 'array', 'items': TEXT, 'minItems': 1, 'maxItems': 4}},
    'measure_contacts': {'candidate_id': TEXT},
    'inspect_contact_chemistry': {
        'candidate_id': TEXT, 'residue_positions': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 16},
    },
    'measure_atom_geometry': {
        'candidate_id': TEXT, 'atom_ids': {'type': 'array', 'items': TEXT, 'minItems': 2, 'maxItems': 4},
    },
    'detect_pockets': {'candidate_id': TEXT},
    'score_candidates': {},
    'compare_structures': {'original_id': TEXT, 'candidate_id': TEXT},
    'plan_revision': {
        'candidate_id': TEXT,
        'method': {'type': 'string', 'enum': ['local_refinement', 'pocket_resampling', 'site_exploration']},
        'hypothesis': TEXT, 'evidence_ids': {'type': 'array', 'items': TEXT},
        'expected_change': TEXT, 'preserve': TEXT,
        'residues': {'type': 'array', 'items': {'type': 'integer'}},
        'max_distance': {'type': 'number'},
    },
    'execute_revision': {'revision_id': TEXT},
    'evaluate_revision': {'revision_id': TEXT},
}
OPTIONAL_PLAN_FIELDS = {
    'pose_preservation': {
        'type': 'object', 'additionalProperties': False,
        'properties': {'contact_positions': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1},
                       'max_ligand_pose_rmsd_A': {'type': 'number', 'exclusiveMinimum': 0}},
        'required': ['contact_positions', 'max_ligand_pose_rmsd_A'],
        'description': 'Optional local refinement contract. Freeze a nonempty subset of measured 4 Å parent '
                       'contacts and a finite symmetry-corrected ligand pose RMSD limit in the receptor CA frame. '
                       'Explain the preservation choices in preserve.',
    },
}
PARAMETERS['register_trigger'] = {
    'candidate_id': TEXT,
    'trigger_type': {'type': 'string', 'enum': ['physical_issue', 'candidate_disagreement', 'user_objective']},
    'evidence_ids': {'type': 'array', 'items': TEXT}, 'goal_quote': TEXT, 'question': TEXT}
PARAMETERS['audit_delivery'] = {
    'evidence_ids': {'type': 'array', 'items': TEXT}, 'reason': TEXT,
    'unresolved_questions': {'type': 'array', 'items': TEXT},
    'remaining_actions': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
        'properties': {'action': {'type': 'string', 'enum': ['local_refinement', 'pocket_resampling', 'site_exploration', 'gather_measurements']},
                       'disposition': {'type': 'string', 'enum': ['not_needed', 'unsupported_by_evidence', 'unavailable', 'deferred']},
                       'reason': TEXT, 'expected_information': TEXT, 'evidence_ids': {'type': 'array', 'items': TEXT}},
        'required': ['action', 'disposition', 'reason', 'expected_information', 'evidence_ids']}}}
PARAMETERS['audit_delivery']['measured_limitations'] = {'type': 'object', 'additionalProperties': {'type': 'array', 'items': TEXT}, 'description': 'Copy current measured_limitations from context exactly. Scientific interpretations cannot erase failed checks. Empty object only when there are no measured limitations.'}
PARAMETERS['audit_delivery']['remaining_actions']['items']['properties']['site_option_status'] = {
    'type': 'string', 'enum': ['not_measured', 'no_eligible_options', 'eligible_options_present'],
    'description': 'Required for site_challenge in request-v2. Copy the measured inventory status; explain utility separately in reason.'}
DESCRIPTIONS = {
    'register_trigger': 'Before planning any revision, register a read, hash-bound physical issue, measured candidate disagreement, or exact quote from the user goal. Cavity detection alone is not justification. Independent literature ingestion is not implemented. Scientific benefit remains a hypothesis.',
    'audit_delivery': 'Before finish/stop, account for every enabled family listed in required_stop_actions, what each could reveal, and why it is not being pursued. Cite read evidence. A deferred option is not proof that all search is exhausted.',
    'read_evidence': 'Read one to four immutable evidence files by ID from the evidence index. Required before citing evidence in memory. Reading is not a new scientific observation.',
    'detect_pockets': 'Detect geometric cavities using fpocket on receptor atoms only. Input positions map exactly to the protein sequence. Does not establish biological binding.',
    'score_candidates': 'GNINA score-only on current assessed identity-valid candidates. No coordinate optimization. Reuses hash-bound scores; final ranking uses CNN score only when every eligible candidate has comparable scores.',
    'inspect_contact_chemistry': 'Identify the nearest heavy-atom pair and its local chemistry for each requested one-based protein residue position, or use [] for the global pair. Reports stable atom IDs, ligand bond neighbors, graph features and group matches. Does not establish reaction chemistry.',
    'measure_atom_geometry': 'Measure distance (2), angle centered on the middle atom (3), or signed dihedral (4) for explicitly ordered atom IDs on the candidate. Use protein:<input position>:<atom name> or ligand:<zero-based heavy atom index>. Cite separately why this geometry is relevant to a proposed hypothesis.',
    'measure_contacts': 'Measure actual residue-to-ligand distances with exact sequence/PDB mapping. Geometric proximity is not binding evidence.',
    'compare_structures': 'Compare predicted candidates: whole-chain CA-aligned symmetry-corrected ligand pose RMSD, centroid displacement, internally fitted ligand RMSD, protein CA changes and contacts. Not core alignment or experimental RMSD.',
    'plan_revision': 'Register a testable revision with actual candidate evidence, expected change and preservation conditions. local_refinement uses residues=[]; pocket_resampling uses one-based input positions. pocket_resampling may test measured pose disagreement even when all physical checks pass; it tests candidate-derived contact feasibility, not pose accuracy. State competing predictions and expected discriminating measurements. No arbitrary repeat or forced revision. pocket_resampling tests old candidate contacts; site_exploration requires cited receptor-only detect_pockets evidence, at least three pocket residues each farther than max(8, max_distance+2) Angstrom from current ligand, and an original candidate. Exploration permits old contact loss but fixes CA backbone RMS<=2 A and maximum displacement<=5 A. Neither action establishes biological truth.',
    'evaluate_revision': 'Apply frozen acceptance contract to the real child measurements. Returns PASS/FAIL/UNKNOWN and computational acceptance; never biological validation.',
    'execute_revision': 'Execute a registered, unused revision within global limits. Creates new artifacts and preserves original. Reassess and compare before selection.',
}

PARAMETERS['read_evidence_bundle'] = {'evidence_ids': {'type': 'array', 'items': TEXT, 'minItems': 1, 'maxItems': 64}}
PARAMETERS['plan_docking'] = {'candidate_id': TEXT, 'question': TEXT, 'hypothesis': TEXT,
                              'expected_observation': TEXT, 'evidence_ids': {'type': 'array', 'items': TEXT}}
PARAMETERS['execute_docking'] = {'experiment_id': TEXT}
DESCRIPTIONS['read_evidence_bundle'] = 'Read up to 64 existing measurements in one summarized bundle. Full residue tables remain in raw evidence and can be read separately. Does not create measurements.'
DESCRIPTIONS['plan_docking'] = 'Freeze one bounded fixed-receptor normal GNINA docking probe using predicted ligand autobox. Cite read comparison evidence, scientific question, hypothesis and expected observations. Not proof of accuracy; original structures retained.'
DESCRIPTIONS['execute_docking'] = 'Execute the single frozen docking experiment. Produces at most five candidates; requires subsequent physical checks, comparisons and uniform scoring before selection.'


PARAMETERS['measure_fragment_variability'] = {'candidate_ids': {'type':'array','items':TEXT,'minItems':2,'maxItems':16}, 'parent_id': TEXT}
PARAMETERS['plan_fragment_experiment'] = {name: TEXT for name in ('candidate_id','torsion_id','question','hypothesis','alternative','expected_observation')}
PARAMETERS['plan_fragment_experiment']['evidence_ids'] = {'type':'array','items':TEXT}
PARAMETERS['execute_fragment_experiment'] = {'experiment_id':TEXT}
DESCRIPTIONS['measure_fragment_variability'] = 'Measure conservative rotatable-bond dispersion and fragment displacement in a shared whole-chain CA frame. Returns explicit mappings and eligible experiments; not real flexibility or error probability.'
DESCRIPTIONS['plan_fragment_experiment'] = 'Freeze one question and competing explanation around a read, eligible torsion measurement. Requires physically valid parent. Four fixed offsets; fixed receptor and restrained core; no target reference. Not the old all-contact preservation repair contract.'
DESCRIPTIONS['execute_fragment_experiment'] = 'Execute the frozen four-start CPU torsion experiment once. Record parent/edit/relaxed checks. Only identity- and manipulation-valid children enter the pool; physical checks and uniform scoring still required.'
PARAMETERS['audit_delivery']['remaining_actions']['items']['properties']['action']['enum'] += ['fragment_exploration','docking_exploration']


PARAMETERS['measure_site_options'] = {'candidate_id': TEXT, 'evidence_ids': {'type':'array','items':TEXT}}
PARAMETERS['plan_site_challenge'] = {k: TEXT for k in ('candidate_id','question','hypothesis_a','hypothesis_b')}
PARAMETERS['plan_site_challenge'].update(pocket_id={'type':'integer'}, evidence_ids={'type':'array','items':TEXT},
    branches={'type':'object','additionalProperties':False,'properties':{k:TEXT for k in ('ALTERNATIVE_FEASIBLE','A_ONLY_IN_TEST','INCONCLUSIVE')},'required':['ALTERNATIVE_FEASIBLE','A_ONLY_IN_TEST','INCONCLUSIVE']})
PARAMETERS['execute_site_challenge'] = {'experiment_id': TEXT}
PARAMETERS['interpret_site_challenge'] = {'experiment_id':TEXT,'evidence_ids':{'type':'array','items':TEXT},'explanation':TEXT,'unresolved':{'type':'array','items':TEXT}}
DESCRIPTIONS.update(measure_site_options='Measure separate equal-sized search boxes from read receptor-only cavity evidence. No valid B means skip.',
 plan_site_challenge='Register one matched A/B computational feasibility experiment. Cite read site options; provide distinct hypotheses and concrete outcome-to-action branches. Neither branch proves biological site uniqueness.',
 execute_site_challenge='Run two explicit-box CPU docking arms once, 180 seconds per search, then assess output poses. Timeout/failure is inconclusive. Artifacts are diagnostic and not automatically admitted to selection.',
 interpret_site_challenge='After reading execution evidence, record scoped interpretation and unresolved alternatives. Follow-up action is bound to the preregistered result branch. Then update Scientist memory.')
PARAMETERS['audit_delivery']['remaining_actions']['items']['properties']['action']['enum'].append('site_challenge')


def schema(name: str, state=None) -> dict:
    from copy import deepcopy
    from tetra.research.question_board import is_open
    result = {'type': 'function', 'function': {'name': name, 'description': DESCRIPTIONS[name],
            'parameters': {'type': 'object', 'properties': PARAMETERS[name],
                           'required': list(PARAMETERS[name]), 'additionalProperties': False}}}
    if name == 'plan_revision':
        result = deepcopy(result)
        result['function']['parameters']['properties'].update(deepcopy(OPTIONAL_PLAN_FIELDS))
    if name == 'audit_delivery':
        result = deepcopy(result)
        if state is None or not is_open(state.get('research', {})):
            result['function']['parameters']['properties'].pop('measured_limitations', None)
            result['function']['parameters']['required'].remove('measured_limitations')
    if name == 'audit_delivery' and state is not None and is_open(state.get('research', {})):
        result = deepcopy(result)
        result['function']['description'] = 'Record the request-grounded stop reason and remaining uncertainty with read evidence. remaining_actions contains only actions actually considered and may be empty; no tool-family checklist or proof of search exhaustion.'
        row = result['function']['parameters']['properties']['remaining_actions']['items']
        row['properties']['action'].pop('enum', None)
        row['properties'].pop('site_option_status', None)
    if name == 'audit_delivery' and state is not None:
        from tetra.research.decision_policy import required_actions
        families = required_actions(state)
        if families:
            remaining = result['function']['parameters']['properties']['remaining_actions']
            remaining.update(minItems=len(families), maxItems=len(families),
                             description='Provide one entry for each: ' + ', '.join(families))
            remaining['items']['properties']['action']['enum'] = families
        from tetra.research.problem_followup import enabled as followup_enabled, extend_audit_schema
        if followup_enabled(state):
            extend_audit_schema(result['function']['parameters'])
    return result


def scoring_readiness(state):
    """Distinguish missing prerequisites from a failed or unavailable scorer."""
    pending, eligible, excluded = [], [], []
    for key in state['candidates']:
        assessment = current_assessment(state, key)
        if not assessment:
            pending.append(key)
        elif assessment.get('checks', {}).get('identity') == 'pass':
            eligible.append(key)
        else:
            excluded.append(key)
    return {'pending_assessment_ids': pending, 'eligible_ids': eligible,
            'identity_excluded_ids': excluded,
            'ready': not pending and bool(eligible)}


def revision_trigger_candidates(state: dict) -> list[str]:
    return [candidate_id for candidate_id, trigger in state.get('revision_triggers', {}).items()
            if candidate_id in state['candidates']
            and trigger.get('candidate_sha256') == state['candidates'][candidate_id]['sha256']]


def available_revision_tools(state: dict) -> list[str]:
    names = ['read_evidence', 'read_evidence_bundle'] if state.get('evidence_access_mode') == 'files-v1' else []
    if state.get('decision_policy_version') == 'decision-v1':
        names += ['register_trigger', 'audit_delivery']
    if state.get('allow_docking_exploration') and state['candidates']:
        names.append('plan_docking')
        if any(v['status'] == 'planned' for v in state.get('docking_experiments', {}).values()):
            names.append('execute_docking')
    if state.get('allow_site_challenge') and state['candidates']:
        names += ['measure_site_options', 'plan_site_challenge']
        if any(x['status'] in ('planned','running') for x in state.get('site_challenges', {}).values()):
            names.append('execute_site_challenge')
        if any(x['status'] == 'completed' for x in state.get('site_challenges', {}).values()):
            names.append('interpret_site_challenge')
    if state.get('allow_fragment_exploration'):
        if len(state['candidates']) > 1:
            names.append('measure_fragment_variability')
        if state.get('fragment_reports') and not state.get('fragment_experiments'):
            names.append('plan_fragment_experiment')
        if any(r['status'] == 'planned' for r in state.get('fragment_experiments', {}).values()):
            names.append('execute_fragment_experiment')
    if state['candidates']:
        names += ['measure_contacts', 'inspect_contact_chemistry', 'measure_atom_geometry']
        if state.get('decision_policy_version') != 'decision-v1' or revision_trigger_candidates(state):
            names.append('plan_revision')
        from tetra.evaluation.external_science import available
        if available('fpocket'):
            names.append('detect_pockets')
        if available('gnina') and scoring_readiness(state)['ready']:
            names.append('score_candidates')
    if len(state['candidates']) > 1:
        names.append('compare_structures')
    if any(plan['status'] == 'planned' for plan in state.get('revisions', {}).values()):
        names.append('execute_revision')
    if any(plan['status'] == 'succeeded' for plan in state.get('revisions', {}).values()):
        names.append('evaluate_revision')
    return names


def _capacity(state: dict, method: str) -> None:
    if method == 'local_refinement':
        if not state.get('refinement_enabled') or (not state.get('uncapped_experiment') and state.get('refinement_count', 0) >= 2):
            raise ValueError('Refinement unavailable or budget exhausted.')
    elif method in ('pocket_resampling', 'site_exploration'):
        if not state.get('resolved', {}).get('input_id') or (not state.get('uncapped_experiment') and state.get('prediction_count', 0) >= 3):
            raise ValueError('Validated molecular input required; at most three total prediction jobs.')
    else:
        raise ValueError('Unsupported revision method.')


def plan_revision(state: dict, candidate_id: str, method: str, hypothesis: str,
                  evidence_ids: list[str], expected_change: str, preserve: str,
                  residues: list[int], max_distance: float, pose_preservation: dict | None = None) -> dict:
    if not isinstance(candidate_id, str) or candidate_id not in state['candidates']:
        raise ValueError('Unknown candidate.')
    if any(not isinstance(text, str) or not text.strip() for text in (hypothesis, expected_change, preserve)):
        raise ValueError('Hypothesis, expected change and preservation conditions are required.')
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError('A revision needs real observed evidence.')
    evidence = state['research']['evidence']
    candidate = state['candidates'][candidate_id]
    relevant = False
    for key in evidence_ids:
        if not isinstance(key, str) or key not in evidence:
            raise ValueError('Unknown evidence ID.')
        observation = evidence[key]['observation']
        result = observation.get('result', {})
        if (observation['action'] in ('assess_structure', 'measure_contacts', 'compare_structures', 'detect_pockets')
                and ((result.get('candidate_id') == candidate_id and result.get('candidate_sha256') == candidate['sha256'])
                     or (observation['action'] == 'compare_structures' and result.get('original_id') == candidate_id
                         and result.get('original_sha256') == candidate['sha256'])) and 'error' not in result):
            relevant = True
    if not relevant:
        raise ValueError('Evidence must include an actual measurement of this immutable candidate.')
    if state.get('decision_policy_version') == 'decision-v1':
        trigger = state.get('revision_triggers', {}).get(candidate_id)
        if not trigger or trigger['candidate_sha256'] != candidate['sha256']:
            raise ValueError('Register a measured revision trigger before planning.')
        if trigger['trigger_type'] == 'physical_issue' and method != 'local_refinement':
            raise ValueError('A physical defect alone supports local repair, not changing the binding site.')
        if not set(trigger['evidence_ids']).issubset(evidence_ids):
            raise ValueError('Plan must cite its registered trigger evidence.')
    _capacity(state, method)
    pockets = None
    if method == 'local_refinement':
        assessment = current_assessment(state, candidate_id)
        if not assessment or not assessment.get('refinement_applicable'):
            raise ValueError('Local refinement needs an assessed geometry/clash problem.')
        if residues != []:
            raise ValueError('Local refinement does not accept pocket residues.')
    else:
        pockets = validate_pockets([{'binder': 'B', 'contacts': [['A', position] for position in residues],
                                     'max_distance': max_distance, 'force': True}], len(candidate['sequence']))
    experiment_input = state['resolved']['input_id'] if method in ('pocket_resampling', 'site_exploration') else candidate['sha256']
    signature = digest(stable({'experiment_input': experiment_input, 'method': method, 'pockets': pockets}))
    revisions = state.setdefault('revisions', {})
    if any(plan['signature'] == signature for plan in revisions.values()):
        raise ValueError('This exact experiment has already been planned; revise conditions or stop.')
    if not state.get('uncapped_experiment') and len(revisions) >= 6:
        raise ValueError('Revision-plan budget exhausted.')
    plan = {'revision_id': f'R{len(revisions) + 1}', 'candidate_id': candidate_id,
            'candidate_sha256': candidate['sha256'], 'method': method,
            'hypothesis': hypothesis, 'evidence_ids': list(evidence_ids),
            'intent': 'REPAIR' if method == 'local_refinement' else 'HYPOTHESIS_TEST',
            'hypothesis_outcome': 'NOT_YET_TESTED',
            'expected_change': expected_change, 'preserve': preserve,
            'pocket_constraints': pockets, 'signature': signature, 'status': 'planned',
            'limitations': ['Hypothesis and expected change are model interpretations.',
                           'Candidate-derived constraints are not independent experimental evidence.']}
    if state.get('decision_policy_version') == 'decision-v1':
        plan['trigger'] = dict(state['revision_triggers'][candidate_id])
    if pose_preservation is not None:
        plan['pose_preservation'] = pose_preservation
    plan['contract'] = revision_policy.freeze_contract(state, plan)
    revisions[plan['revision_id']] = plan
    return plan.copy()


def dispatch(agent, state: dict, name: str, args: dict, attempt_id: str) -> dict:
    if name == 'register_trigger':
        from tetra.research.decision_policy import validate_trigger
        result = validate_trigger(state, **args)
        state.setdefault('revision_triggers', {})[args['candidate_id']] = result
        return result
    if name == 'audit_delivery':
        from tetra.research.decision_policy import validate_stop
        result = validate_stop(state, **args)
        state['delivery_audit'] = result
        return result
    required = set(PARAMETERS[name])
    optional = set(OPTIONAL_PLAN_FIELDS) if name == 'plan_revision' else set()
    if not required <= set(args) <= required | optional:
        raise ValueError('Invalid scientific action fields.')
    folder = agent.folder(state)
    if name in ('measure_site_options','plan_site_challenge','execute_site_challenge','interpret_site_challenge'):
        if not state.get('allow_site_challenge'):
            raise ValueError('Site challenge requires opt-in.')
        from tetra.evaluation import site_challenge
        if name == 'execute_site_challenge':
            return site_challenge.execute(agent, state, **args)
        return {'measure_site_options':site_challenge.measure,'plan_site_challenge':site_challenge.plan,'interpret_site_challenge':site_challenge.interpret}[name](state, **args)
    if name in ('measure_fragment_variability','plan_fragment_experiment','execute_fragment_experiment'):
        if not state.get('allow_fragment_exploration'):
            raise ValueError('Fragment tools require opt-in.')
        from tetra.evaluation import fragment_tools
        if name == 'measure_fragment_variability':
            return fragment_tools.measure(state, **args)
        if name == 'plan_fragment_experiment':
            return fragment_tools.plan(state, **args)
        return fragment_tools.execute(agent, state, **args)
    if name == 'read_evidence_bundle':
        from tetra.research.evidence_bundle import read_bundle
        return read_bundle(state, folder, args['evidence_ids'])
    if name == 'plan_docking':
        from tetra.evaluation.docking_exploration import plan_experiment
        return plan_experiment(state, **args)
    if name == 'execute_docking':
        from tetra.evaluation.docking_exploration import execute_experiment
        return execute_experiment(agent, state, **args)
    if name == 'read_evidence':
        from tetra.research.evidence_files import EvidenceReadTooLarge, read_evidence
        try:
            return read_evidence(state, folder, args['evidence_ids'])
        except EvidenceReadTooLarge:
            from tetra.research.evidence_bundle import read_bundle
            result = read_bundle(state, folder, args['evidence_ids'])
            result['detail'] = 'The full read exceeded its limit. These are summaries; read individual IDs for detail.'
            return result
    if name == 'score_candidates':
        readiness = scoring_readiness(state)
        if not readiness['ready']:
            raise ValueError('Scoring prerequisites unmet; assess every candidate first. This is not a GNINA failure. ' + json.dumps(readiness))
        from tetra.evaluation.external_science import score_candidate, tool_identity, available
        if not available('gnina'):
            raise ValueError('GNINA has not passed local tool validation.')
        if len(state['candidates']) > 16:
            raise ValueError('At most sixteen candidates may be scored per bounded request.')
        scores = state.setdefault('pose_scores', {})
        errors = {}
        import time
        deadline = time.monotonic() + 300
        identity = tool_identity('gnina')
        for key, candidate in state['candidates'].items():
            assessment = current_assessment(state, key)
            if not assessment or assessment.get('checks', {}).get('identity') != 'pass':
                continue
            verify_candidate(candidate)
            previous = scores.get(key, {})
            if (previous.get('candidate_sha256') == candidate['sha256'] and
                    previous.get('provenance', {}).get('binary_sha256') == identity['binary_sha256'] and
                    previous.get('provenance', {}).get('runtime_sha256') == identity.get('runtime_sha256')):
                continue
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Total GNINA action budget exhausted; partial pool is not reranked.')
                scores[key] = score_candidate(candidate, folder / 'scores' / attempt_id / key, timeout=min(180, remaining))
                agent.save(state)
            except Exception as exc:
                errors[key] = type(exc).__name__ + ': ' + str(exc)
        return {'scores': scores, 'errors': errors, 'readiness': readiness,
                'ranking_requires_complete_comparable_pool': True}
    if name == 'detect_pockets':
        from tetra.evaluation.external_science import detect_pockets, available
        if not available('fpocket'):
            raise ValueError('fpocket has not passed local tool validation.')
        candidate = state['candidates'].get(args.get('candidate_id'))
        if candidate is None:
            raise ValueError('Unknown candidate.')
        # A replay after interruption gets its own directory; old artifacts remain intact.
        import uuid
        return detect_pockets(candidate, folder / 'pockets' / (attempt_id + '-' + uuid.uuid4().hex[:8]))
    if name == 'plan_revision':
        candidate = state['candidates'].get(args.get('candidate_id'))
        if candidate is None:
            raise ValueError('Unknown revision candidate.')
        verify_candidate(candidate)
        return plan_revision(state, **args)
    if name in ('inspect_contact_chemistry', 'measure_atom_geometry'):
        from tetra.evaluation.contact_chemistry import inspect_contact_chemistry, measure_atom_geometry
        candidate_id = args['candidate_id']
        if not isinstance(candidate_id, str) or candidate_id not in state['candidates']:
            raise ValueError('Unknown candidate.')
        candidate = state['candidates'][candidate_id]
        result = (inspect_contact_chemistry(candidate, args['residue_positions'])
                  if name == 'inspect_contact_chemistry' else measure_atom_geometry(candidate, args['atom_ids']))
        path = folder / 'measurements' / (attempt_id + '.json')
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, result)
        return {**result, 'artifact_path': str(path)}
    if name in ('measure_contacts', 'compare_structures'):
        for key in args.values():
            if not isinstance(key, str) or key not in state['candidates']:
                raise ValueError('Unknown candidate.')
        candidate = state['candidates'][args['candidate_id']]
        if name == 'measure_contacts':
            result = measure_contacts(candidate)
        else:
            result = compare_structures(state['candidates'][args['original_id']], candidate)
        path = folder / 'measurements' / (attempt_id + '.json')
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, result)
        return {**result, 'artifact_path': str(path)}
    key = args['revision_id']
    if not isinstance(key, str) or key not in state.get('revisions', {}):
        raise ValueError('Unknown revision.')
    plan = state['revisions'][key]
    revision_policy.verify_contract(plan)
    if name == 'evaluate_revision':
        verify_candidate(state['candidates'][plan['candidate_id']])
        for child in revision_policy.children(plan):
            verify_candidate(state['candidates'][child])
        results = [revision_policy.evaluate_revision(state, plan, child) for child in revision_policy.children(plan)]
        if not results:
            raise ValueError('Revision has no completed child structure to evaluate.')
        plan['acceptance_results'] = results
        return {'revision_id': key, 'acceptance_results': results}
    # A pending replay is permitted only for its own persisted attempt.
    if plan['status'] != 'planned' and plan.get('attempt_id') != attempt_id:
        raise ValueError('Revision already executed; no automatic repeat.')
    candidate = state['candidates'][plan['candidate_id']]
    verify_candidate(candidate)
    if candidate['sha256'] != plan['candidate_sha256']:
        raise ValueError('Revision candidate changed.')
    if plan['status'] == 'planned':
        _capacity(state, plan['method'])
        plan.update(status='running', attempt_id=attempt_id)
        agent.save(state)
    if plan['method'] == 'local_refinement':
        result = agent.dispatch(state, 'refine_structure',
                                {'candidate_id': plan['candidate_id'], 'reason': plan['hypothesis']}, attempt_id)
        plan['status'] = result.get('status', 'failed')
        plan['result'] = result
        return {'revision_id': key, **result}
    record = agent.store.input(state['resolved']['input_id'])
    if record.get('sequence') != candidate['sequence'] or record.get('smiles') != candidate['smiles']:
        raise ValueError('Prediction input differs from the revision candidate molecular identity.')
    config = {'seed': 101, 'diffusion_samples': 1, 'recycling_steps': 3, 'sampling_steps': 200,
              'pocket_constraints': plan['pocket_constraints']}
    job = agent.store.submit(record['input_id'], config)
    if job.get('reused') and plan.get('job_id') != job['job_id']:
        plan.update(status='cache_reused', job_id=job['job_id'])
        raise ValueError('This exact prediction already exists; no new structural experiment was executed.')
    plan['job_id'] = job['job_id']
    if not state['pending_call'].get('reserved'):
        state['prediction_count'] += 1
        state['pending_call']['reserved'] = True
    state['job_id'] = job['job_id']
    state['active_prediction_revision'] = key
    agent.save(state)
    result = backend.launch(agent.store, job['job_id'])
    state['status'] = 'waiting_prediction'
    return {'revision_id': key, **result}


def required_comparisons(plan: dict, child_id: str) -> list[tuple[str, str]]:
    pairs = [(plan['candidate_id'], child_id)]
    def ancestors(current):
        for ancestor in current.get('contract', {}).get('inherited_plans', []):
            if (ancestor['contract']['policy_version'] == 'site-exploration-v1'
                    or ancestor['contract'].get('pose_preservation')):
                pair = (ancestor['candidate_id'], child_id)
                if pair not in pairs:
                    pairs.append(pair)
            ancestors(ancestor)
    ancestors(plan)
    return pairs


def outstanding_comparisons(state: dict) -> list[dict]:
    pending = []
    for key, plan in state.get('revisions', {}).items():
        if plan['status'] != 'succeeded':
            continue
        for child_id in revision_policy.children(plan):
            for parent_id, child_id in required_comparisons(plan, child_id):
                parent = state['candidates'].get(parent_id, {})
                child = state['candidates'].get(child_id, {})
                found = False
                for item in state['research']['evidence'].values():
                    observation = item['observation']
                    result = observation.get('result', {})
                    if (observation['action'] == 'compare_structures'
                            and result.get('original_id') == parent_id
                            and result.get('candidate_id') == child_id
                            and result.get('original_sha256') == parent.get('sha256')
                            and result.get('candidate_sha256') == child.get('sha256') and 'error' not in result):
                        found = True
                        break
                if not found:
                    pending.append({'revision_id': key, 'original_id': parent_id, 'candidate_id': child_id})
    return pending


def unreviewed_revisions(state: dict) -> list[str]:
    return list(dict.fromkeys(row['revision_id'] for row in outstanding_comparisons(state)))
