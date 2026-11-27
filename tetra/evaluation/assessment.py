"""Immutable receptor/ligand candidates and reference-free physical assessments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tetra.runtime.download import atomic_json

PROJECT=Path(__file__).resolve().parents[2]
CHECKS=('sanitization','all_atoms_connected','bond_lengths','bond_angles','internal_steric_clash',
        'aromatic_ring_flatness','double_bond_flatness','internal_energy','minimum_distance_to_protein','volume_overlap_with_protein')

def fingerprint(receptor,ligand):
    return hashlib.sha256(Path(receptor).read_bytes()+b'\0'+Path(ligand).read_bytes()).hexdigest()

def verify_candidate(candidate):
    if fingerprint(candidate['receptor'],candidate['ligand'])!=candidate['sha256']:raise ValueError('Candidate files changed after registration.')

def import_candidate(source,folder):
    if set(source)-{'receptor','ligand','smiles','sequence','parent_id','original_confidence','confidence_provenance'}:
        raise ValueError('Unsupported candidate fields; reference data is not an agent input.')
    if not all(source.get(k) for k in ('receptor','ligand','smiles','sequence')):raise ValueError('Candidate requires receptor, ligand, sequence and SMILES.')
    from tetra.runtime.inputs import validate
    validate(source['sequence'],source['smiles'])
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Snapshot the bytes; never optimize caller-owned files.
    data={k:v for k,v in source.items() if k not in ('receptor','ligand')}
    data['sha256']=fingerprint(source['receptor'],source['ligand']);data['candidate_id']=data['sha256'][:24]
    dest=folder/data['candidate_id'];dest.mkdir(exist_ok=True)
    for key,suffix in [('receptor','.pdb'),('ligand','.sdf')]:
        path=dest/(key+suffix);raw=Path(source[key]).read_bytes()
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Immutable candidate collision.')
        if not path.exists():path.write_bytes(raw)
        data[key]=str(path.resolve())
    if source.get('parent_id'):
        data.pop('original_confidence',None);data.pop('confidence_provenance',None)
    manifest=dest/'candidate.json'
    if manifest.exists() and json.loads(manifest.read_text())!=data:raise ValueError('Candidate metadata collision.')
    atomic_json(manifest,data)
    return data

def ligand_identity_diagnostics(mol, smiles):
    """Separate graph, isotope and stereo mismatches without weakening identity."""
    from rdkit import Chem
    expected = Chem.MolFromSmiles(smiles)
    if mol is None or expected is None:
        return {'matches': False, 'reason': 'unreadable_molecule'}
    observed = Chem.RemoveHs(Chem.Mol(mol))
    expected = Chem.RemoveHs(expected)
    before = Chem.MolToSmiles(observed, isomericSmiles=True)
    if observed.GetNumConformers() == 1:
        Chem.AssignStereochemistryFrom3D(observed)
    actual = Chem.MolToSmiles(observed, isomericSmiles=True)
    target = Chem.MolToSmiles(expected, isomericSmiles=True)
    def graph(molecule, keep_isotopes):
        copy = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(copy)
        if not keep_isotopes:
            for atom in copy.GetAtoms():
                atom.SetIsotope(0)
        return Chem.MolToSmiles(copy, isomericSmiles=True)
    return {'matches': actual == target,
            'connectivity_matches': graph(observed, False) == graph(expected, False),
            'isotope_identity_matches': graph(observed, True) == graph(expected, True),
            'stereochemical_identity_matches': actual == target,
            'expected_isomeric_smiles': target, 'observed_isomeric_smiles': actual,
            'stored_isomeric_smiles': before, 'stereo_changed_by_coordinates': before != actual,
            'stereo_centers_observed': Chem.FindMolChiralCenters(observed, includeUnassigned=True),
            'stereo_centers_expected': Chem.FindMolChiralCenters(expected, includeUnassigned=True),
            'note': 'Atom indices are local to each molecule; not an atom correspondence.'}


def identity_diagnostics(candidate):
    import gemmi
    import numpy as np
    from rdkit import Chem
    from tetra.prediction.backend import complete_protein
    verify_candidate(candidate)
    ligand = Chem.SDMolSupplier(candidate['ligand'], removeHs=False)[0]
    details = {'ligand': ligand_identity_diagnostics(ligand, candidate['smiles'])}
    details['single_ligand_conformer'] = ligand is not None and ligand.GetNumConformers() == 1
    structure = gemmi.read_structure(candidate['receptor'])
    details['single_protein_chain'] = len(structure) == 1 and len(structure[0]) == 1
    details.update(protein_sequence_matches=False, protein_heavy_atoms_complete=False, coordinates_finite=False)
    if details['single_protein_chain']:
        chain = structure[0][0]
        sequence = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in chain)
        details['protein_sequence_matches'] = sequence == candidate['sequence']
        details['protein_heavy_atoms_complete'] = bool(complete_protein(chain))
        coords = [(a.pos.x, a.pos.y, a.pos.z) for r in chain for a in r]
        details['coordinates_finite'] = bool(details['single_ligand_conformer'] and
            np.isfinite(coords).all() and np.isfinite(ligand.GetConformer().GetPositions()).all())
    details['matches'] = details['ligand']['matches'] and all(
        details[key] for key in ('single_ligand_conformer', 'single_protein_chain',
                                'protein_sequence_matches', 'protein_heavy_atoms_complete', 'coordinates_finite'))
    return details


def identity_check(candidate):
    return identity_diagnostics(candidate)['matches']

def measurement_semantics(raw):
    """Expose metric domains without guessing atom mapping or biological meaning."""
    return {
        'clash_domains': {
            'ligand_internal': {
                'count': raw.get('number_clashes'),
                'source_metric': 'number_clashes',
                'scope': 'Intramolecular ligand geometry only; does not count protein-ligand clashes.',
            },
            'protein_ligand': {
                'count': raw.get('num_pairwise_clashes_protein'),
                'source_metric': 'num_pairwise_clashes_protein',
                'clash_detected': raw.get('most_extreme_clash_protein') if type(raw.get('most_extreme_clash_protein')) is bool else None,
                'flag_semantics': 'most_extreme_clash_protein=true means the atom pair IS a clash, not a passed check.',
                'scope': 'Intermolecular ligand-protein atom pairs; independent of ligand internal clashes and volume-overlap checks.',
            },
        },
        'protein_atom_identifier': {
            'value': raw.get('most_extreme_protein_atom_id_protein'),
            'domain': 'PoseBusters protein atom identifier, NOT residue number or sequence position',
            'residue_assignment': None,
            'mapping_status': 'not_established_by_this_measurement',
        },
        'interpretation_limits': [
            'A short distance does not identify a catalytic residue or establish a chemical mechanism.',
            'A failed physical check remains a measured defect even if qualified delivery is permitted.',
            'A mechanistic explanation is untested unless supported by separate appropriate evidence.',
        ],
    }


def assess_candidate(candidate,output_dir,python=None):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    result={'candidate_id':candidate['candidate_id'],'candidate_sha256':candidate['sha256'],
            'status':'unavailable','checks':{},'refinement_applicable':False,
            'limitations':['Physical validity does not establish pose accuracy. No target reference was used.']}
    try:
        verify_candidate(candidate)
        atomic_json(output_dir/'request.json',candidate)
        executable=python or os.environ.get('TETRA_ASSESS_PYTHON',sys.executable)
        command=[executable,str(PROJECT/'scripts/assess_structure.py'),str(output_dir/'request.json'),str(output_dir/'raw.json')]
        with (output_dir/'worker.log').open('wb') as log:
            run=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5','180',*command],stdout=log,stderr=subprocess.STDOUT,timeout=190,env={**os.environ,'OMP_NUM_THREADS':'1'})
        if run.returncode:raise ValueError('Assessment worker failed; see worker.log (exit '+str(run.returncode)+').')
        raw=json.loads((output_dir/'raw.json').read_text())
        raw['identity_diagnostics'] = identity_diagnostics(candidate)
        raw['checks']['identity'] = 'pass' if raw['identity_diagnostics']['matches'] else 'fail'
        result.update(raw)
        result['measurement_semantics'] = measurement_semantics(raw.get('raw_metrics', {}))
        result['checks']={key:raw.get('checks',{}).get(key,'unavailable') for key in ('identity',*CHECKS)}
        result['refinement_applicable']=(result['checks']['identity']=='pass' and
            any(result['checks'][k]=='fail' for k in ('bond_lengths','bond_angles','internal_steric_clash','minimum_distance_to_protein','volume_overlap_with_protein')))
        verify_candidate(candidate)
        result['status']='assessed'
    except Exception as e:result.update(status='unavailable',refinement_applicable=False,error=str(e))
    atomic_json(output_dir/'assessment.json',result)
    return result
