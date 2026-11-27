"""Shared capabilities with artifact-local prerequisites, not task dispatch branches."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Capability:
    description: str
    parameters: dict
    cost: str
    output: str
    optional: tuple = ()
    def schema(self,name):
        return {'type':'function','function':{'name':name,'description':self.description+' Cost: '+self.cost,
            'parameters':{'type':'object','properties':self.parameters,'required':[key for key in self.parameters if key not in self.optional],'additionalProperties':False}}}

ID={'type':'string'}


def _closed_object(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': False}


def _token(selector):
    return {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {
        'anyOf': [ID, selector],
    }}


_POSITION = {'type': 'integer', 'minimum': 1}
_SELECTOR = _closed_object({'smarts': ID, 'atom': {'type': 'integer', 'minimum': 0}, 'element': ID})
_TOKEN = _token({'anyOf': [_POSITION, _SELECTOR]})
_DISTANCE = {'type': 'number', 'minimum': 4, 'maximum': 20}
_FORCE = {'type': 'boolean'}
_CONDITION = {'anyOf': [
    _closed_object({'pocket': _closed_object({'binder': ID,
        'contacts': {'type': 'array', 'minItems': 1, 'items': _token(_POSITION)},
        'max_distance': _DISTANCE, 'force': _FORCE})}),
    _closed_object({'contact': _closed_object({'token1': _TOKEN, 'token2': _TOKEN,
        'max_distance': _DISTANCE, 'force': _FORCE})}),
]}
_SAMPLING = _closed_object({
    'diffusion_samples': {'type': 'integer', 'minimum': 1, 'maximum': 5},
    'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2147483647},
    'recycling_steps': {'type': 'integer', 'minimum': 1, 'maximum': 10},
    'sampling_steps': {'type': 'integer', 'minimum': 1, 'maximum': 400},
}, required=[])

TOOLS={
 'record_delivery_requirement': Capability(
     'Record the exact final structure count explicitly requested by the user. Quote the request verbatim.',
     {'count': {'type': 'integer', 'minimum': 1, 'maximum': 25}, 'source_quote': ID},
     'CPU', 'delivery requirement'),
 'find_reference_complexes': Capability(
     'Search experimental reference complexes by the resolved input sequence. Returns primary '
     'citations and screened PDB identifiers for subsequent structural measurement.',
     {}, 'network', 'reference structure citations'),
 'search_literature': Capability(
     'Search PubMed for structural evidence usable as Boltz conditions: binding-site residues, '
     'ligand fragment-to-residue contacts, and measured distances in comparable complexes. '
     'Use the protein or family and one structural concept per query. Supply a broader_query '
     'that removes uncertain ligand names or exact motifs and retains the same question. '
     'Pass family_terms to retain the protein family when broadening. The tool tries broader_query '
     'on zero hits. Search hits are discovery metadata. If no hit is relevant, submit a better query. '
     'Use find_reference_complexes for sequence-based discovery.',
     {'query': ID, 'broader_query': ID, 'family_terms': {'type': 'array', 'items': ID, 'maxItems': 5},
      'route': {'type': 'string', 'enum': ['pubmed', 'sequence']}},
     'network', 'source IDs', optional=('broader_query', 'family_terms', 'route')),
 'read_literature': Capability(
     'Read available PMC full-text sections, falling back to a labeled abstract, for a search result.',
     {'source_id': ID}, 'network', 'provenanced abstract'),
 'measure_reference_pocket': Capability(
     'Measure screened experimental PDBs from a sequence-search citation group. '
     'Return observed reference contacts plus mapped input positions. No paper read prerequisite.',
     {'source_id': ID}, 'network', 'measured reference pocket'),
 'read_structure_details': Capability(
     'After screening a relevant article, locate structural passages in its full text, tables and '
     'figure captions. If no passage is available, measure DOI-linked public reference structures. '
     'Use mapped residues for pocket hypotheses. Reference ligand atoms require independent mapping '
     'to support a target contact.',
     {'source_id': ID, 'terms': {'type': 'array', 'items': ID, 'minItems': 1, 'maxItems': 6}},
     'network', 'attributable structural passages'),
 'resolve_input':Capability('Resolve supplied identities using UniProt/PubChem; ambiguous input terminates. Never invent identities.',{},'network','validated input'),
 'plan_prediction_experiment': Capability(
     'Register a Boltz test. Encode each applicable, mapped, evidence-supported pocket or contact in conditions. For an unconstrained control or when no condition can be mapped, use conditions=[] and explain why in decision_implications. Sampling config uses diffusion_samples, seed, recycling_steps and sampling_steps. The input MSA and molecules are inherited. For a ligand contact use [B, {smarts, atom, element}] with a zero-based fragment atom.',
     {'question': ID, 'hypothesis': ID, 'evidence_ids': {'type': 'array', 'items': ID},
      'conditions': {'type': 'array', 'items': _CONDITION},
      'condition_purpose': {'type': 'string', 'enum': ['proximity', 'unconstrained_sampling'],
                            'description': 'Actual intervention: proximity is an upper distance bound. '
                                           'If the question needs a lower bound, named protein atom, '
                                           'bond or angle, choose a different tool.'},
      'expected_observations': ID, 'decision_implications': ID,
      'literature_condition_basis': {'type': 'string',
          'description': 'Required after a detailed literature read. Explain the source passage, '
                         'residue/fragment correspondence, and how it sets the condition or why it cannot.'},
      'config': _SAMPLING}, 'CPU', 'frozen plan', optional=('literature_condition_basis',)),
 'execute_prediction_experiment': Capability(
     'Execute a registered condition group once.', {'experiment_id': ID}, 'GPU', 'candidate structures'),
 'predict_complex':Capability(
     'Predict validated molecules with Boltz-2. Optional diffusion_samples (default 1) and seed (default 42). '
     'Additional sampling can test a scientific uncertainty without a repair trigger. '
     'Identical configuration may reuse a cached job; repeated samples are not independent biological evidence.',
     {'diffusion_samples':{'type':'integer','minimum':1,'maximum':5},
      'seed':{'type':'integer','minimum':0,'maximum':2147483647}},
     'GPU','candidate structures',optional=('diffusion_samples','seed')),
 'analyze_prediction_batch': Capability(
     'Assess each candidate in a completed prediction group and report recorded condition measurements. Measure contacts and pairwise poses for identity-valid candidates, with individual failures reported separately.',
     {'experiment_id': ID}, 'CPU', 'batch measurements'),
 'assess_structure':Capability('Run reference-free physical checks on an immutable candidate.',{'candidate_id':ID},'CPU','hash-bound assessment'),
 'assess_structure_batch': Capability(
     'Assess 2 to 5 unassessed candidates in one tool call. Pass candidate_ids together; '
     'each result is recorded separately. Prefer this for multiple assessments in the same question.',
     {'candidate_ids': {'type': 'array', 'items': ID, 'minItems': 2, 'maxItems': 5, 'uniqueItems': True}},
     'CPU', 'separate hash-bound assessments'),
 'refine_structure':Capability('Constrained coordinate optimization for assessed geometry/clash issues. Preserve identity; reassess output.',{'candidate_id':ID,'reason':ID},'CPU, bounded','new child candidate'),
 'select_candidates':Capability('Compare assessed original and refined candidates using the documented conservative rule.',{},'small','selection evidence'),
 'finish':Capability('Deliver selected structures and limitations. Cannot claim measured pose accuracy.',{},'small','report'),
 'stop':Capability('Stop when no justified action remains; preserve artifacts and report incomplete evidence.',{'reason':ID},'small','partial report')}

def current_assessment(state,key):
    c=state['candidates'][key];a=state['assessments'].get(key)
    return a if a and a.get('candidate_sha256')==c['sha256'] else None

def missing_candidate_assessments(state):
    return [key for key in state['candidates'] if not current_assessment(state, key)]


def eligible_tools(state):
    names=['stop']
    if not state.get('resolved') and not state.get('candidates'):names.append('resolve_input')
    prediction_limit = 3 if 'research' in state else 1
    if (state.get('resolved', {}).get('input_id') and state.get('status') != 'waiting_prediction'
            and (state.get('uncapped_experiment') or state.get('prediction_count', 0) < prediction_limit)):
        names.append('predict_complex')
    missing = missing_candidate_assessments(state)
    if missing:names.append('assess_structure')
    if state['candidates'] and not missing:names.append('select_candidates')
    if state.get('refinement_enabled') and state.get('refinement_count',0)<state.get('max_refinements',2):
        for k in state['candidates']:
            a=current_assessment(state,k)
            if a and a.get('refinement_applicable') and k not in state.get('attempted_parents',[]):
                names.append('refine_structure');break
    if state.get('selected') and not missing:names.append('finish')
    return names


def validate_arguments(name, args):
    capability = TOOLS[name]
    required = set(capability.parameters) - set(capability.optional)
    if not isinstance(args, dict) or not required.issubset(args) or set(args) - set(capability.parameters):
        raise ValueError('Tool parameters do not match schema.')
    if name == 'predict_complex':
        from tetra.runtime.inputs import validate_config
        validate_config(args)
    elif name == 'record_delivery_requirement':
        if type(args['count']) is not int or not isinstance(args['source_quote'], str):
            raise ValueError('Provide integer count and verbatim source quote.')
    elif name == 'search_literature':
        for key in ('query', 'broader_query'):
            if key in args and (not isinstance(args[key], str) or not args[key].strip()):
                raise ValueError(f'{key} must be nonempty text.')
        terms = args.get('family_terms', [])
        if (not isinstance(terms, list) or len(terms) > 5
                or any(not isinstance(term, str) or not term.strip() for term in terms)):
            raise ValueError('family_terms must be an array of up to five nonempty strings.')
        if args.get('route', 'pubmed') not in ('pubmed', 'sequence'):
            raise ValueError('route must be pubmed or sequence.')
    elif name == 'read_structure_details':
        if not isinstance(args['source_id'], str) or not isinstance(args['terms'], list):
            raise ValueError('Provide a screened source ID and structural search terms.')
    elif name == 'plan_prediction_experiment':
        for key in ('question', 'hypothesis', 'expected_observations', 'decision_implications'):
            if not isinstance(args[key], str) or not args[key].strip():
                raise ValueError(f'{key} must be nonempty text.')
    elif name == 'assess_structure_batch':
        ids = args['candidate_ids']
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 5
                or any(not isinstance(cid, str) or not cid.strip() for cid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Pass 2 to 5 distinct candidate IDs in candidate_ids.')
    elif any(not isinstance(value, str) or not value.strip() for value in args.values()):
        raise ValueError('Arguments must be nonempty strings.')
