"""Reference-free residue contacts and aligned changes between predicted candidates."""
from typing import Any
import gemmi
import numpy as np
from rdkit import Chem
from tetra.evaluation.assessment import identity_check, verify_candidate


def sequence_residues(structure: gemmi.Structure, sequence: str) -> list[dict]:
    """Map exact complete sequence order to PDB numbering, including insertion codes."""
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError('Metrics require exactly one protein chain/model.')
    chain = structure[0][0]
    observed = ''.join(gemmi.find_tabulated_residue(residue.name).one_letter_code for residue in chain)
    if observed != sequence:
        raise ValueError('Complete structure sequence does not match the registered input.')
    return [{'input_position': index, 'chain': chain.name, 'residue_name': residue.name,
             'pdb_number': residue.seqid.num, 'pdb_insertion_code': residue.seqid.icode.strip()}
            for index, residue in enumerate(chain, 1)]


def align_points(moving: np.ndarray, reference: np.ndarray) -> tuple:
    """Least-squares proper rigid alignment; reflections are forbidden."""
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3 or len(moving) < 3:
        raise ValueError('Alignment requires matched arrays of at least three points.')
    if not np.isfinite(moving).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite coordinates.')
    left = moving.mean(axis=0)
    right = reference.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - left).T @ (reference - right))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = right - left @ rotation
    return moving @ rotation + translation, rotation, translation


def _load(candidate: dict) -> tuple:
    verify_candidate(candidate)
    if not identity_check(candidate):
        raise ValueError('Candidate identity/completeness check failed.')
    structure = gemmi.read_structure(candidate['receptor'])
    mapping = sequence_residues(structure, candidate['sequence'])
    molecule = Chem.RemoveHs(Chem.SDMolSupplier(candidate['ligand'], removeHs=False)[0])
    coordinates = molecule.GetConformer().GetPositions()
    return structure, mapping, molecule, coordinates


def measure_contacts(candidate: dict, cutoff: float = 4.0) -> dict[str, Any]:
    """Measure nearest heavy-atom distances per input residue; not interaction typing."""
    structure, mapping, _, ligand = _load(candidate)
    rows = []
    for residue, row in zip(structure[0][0], mapping):
        protein = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue
                            if not atom.element.is_hydrogen])
        distance = np.linalg.norm(protein[:, None, :] - ligand[None, :, :], axis=-1)
        rows.append({**row, 'minimum_distance_A': float(distance.min()),
                     'atom_pairs_within_cutoff': int((distance <= cutoff).sum())})
    return {'candidate_id': candidate['candidate_id'], 'candidate_sha256': candidate['sha256'],
            'cutoff_A': cutoff, 'residue_mapping': 'exact complete sequence, one-based input positions',
            'minimum_distance_A': min(row['minimum_distance_A'] for row in rows),
            'contact_positions': [row['input_position'] for row in rows if row['minimum_distance_A'] <= cutoff],
            'residues': rows,
            'limitations': ['Geometric proximity is not a chemical interaction or biological binding-site annotation.']}


def ligand_pose_comparison(reference_mol, moving_mol, rotation, translation, max_matches=4096):
    """Exact graph-symmetry comparisons; receptor frame is fixed for pose RMSD."""
    reference = reference_mol.GetConformer().GetPositions()
    moving = moving_mol.GetConformer().GetPositions() @ rotation + translation
    if reference.shape != moving.shape or not np.isfinite(moving).all() or not np.isfinite(reference).all():
        raise ValueError('Ligand coordinates must be finite and atom counts equal.')
    matches = moving_mol.GetSubstructMatches(reference_mol, uniquify=False,
                                            useChirality=True, maxMatches=max_matches + 1)
    if not matches or len(matches) > max_matches:
        return {'status': 'unavailable', 'reason': 'No complete stereochemical mapping or symmetry enumeration limit exceeded.'}
    pose, internal = [], []
    for match in matches:
        mapped = moving[list(match)]
        pose.append(float(np.sqrt(np.mean(np.sum((mapped-reference)**2, axis=1)))))
        if len(mapped) >= 3:
            fitted, _, _ = align_points(mapped, reference)
            internal.append(float(np.sqrt(np.mean(np.sum((fitted-reference)**2, axis=1)))))
    return {'status': 'measured', 'atom_mapping_count': len(matches),
            'symmetry_corrected_pose_rmsd_A': min(pose),
            'centroid_displacement_A': float(np.linalg.norm(moving.mean(0)-reference.mean(0))),
            'internal_conformation_rmsd_A': min(internal) if internal else None,
            'alignment': 'whole_chain_CA; no ligand fitting for pose RMSD',
            'symmetry': 'complete chirality-preserving molecular graph matches',
            'limitations': ['Prediction-to-prediction comparison, not experimental accuracy.',
                           'Internal RMSD uses independent ligand fitting; metrics are not additive.',
                           'No unique orientation angle is inferred for symmetric or flexible ligands.',
                           'Whole-chain alignment is not core alignment.']}


def compare_structures(original: dict, candidate: dict) -> dict:
    if original['sequence'] != candidate['sequence']:
        raise ValueError('Cannot compare different protein identities.')
    first, mapping, reference_mol, _ = _load(original)
    second, _, moving_mol, _ = _load(candidate)
    if Chem.MolToSmiles(Chem.MolFromSmiles(original['smiles'])) != Chem.MolToSmiles(Chem.MolFromSmiles(candidate['smiles'])):
        raise ValueError('Cannot compare different ligands.')
    def ca_coordinates(structure):
        atoms = []
        for residue in structure[0][0]:
            matching = [atom for atom in residue if atom.name == 'CA']
            if len(matching) != 1:
                raise ValueError('Ambiguous alpha-carbon mapping.')
            atoms.append([matching[0].pos.x, matching[0].pos.y, matching[0].pos.z])
        return np.array(atoms)
    reference = ca_coordinates(first)
    aligned, rotation, translation = align_points(ca_coordinates(second), reference)
    displacements = np.linalg.norm(aligned - reference, axis=1)
    before = measure_contacts(original)
    after = measure_contacts(candidate)
    old = set(before['contact_positions'])
    new = set(after['contact_positions'])
    return {'original_id': original['candidate_id'], 'candidate_id': candidate['candidate_id'],
            'original_sha256': original['sha256'], 'candidate_sha256': candidate['sha256'],
            'protein_sequence_length': len(original['sequence']),
            'ligand_pose_comparison': ligand_pose_comparison(reference_mol, moving_mol, rotation, translation),
            'protein_CA_aligned_change_rms_A': float(np.sqrt(np.mean(displacements ** 2))),
            'maximum_CA_displacement_A': float(displacements.max()),
            'per_residue_CA_displacements': [{**row, 'displacement_A': float(value)} for row, value in zip(mapping, displacements)],
            'lost_contact_positions': sorted(old - new), 'gained_contact_positions': sorted(new - old),
            'retained_contact_fraction': len(old & new) / len(old) if old else None,
            'minimum_distance_before_A': before['minimum_distance_A'],
            'minimum_distance_after_A': after['minimum_distance_A'],
            'limitations': ['Comparison is between predictions, not experimental RMSD.',
                           'Contact loss at a fixed cutoff is not proof of lost binding.',
                           'Whole-chain alignment can conceal local domain movement; inspect local structure.']}


def compare_binding_site(original, candidate, positions):
    """Use a caller-frozen set of exact sequence positions; never fit the ligand."""
    first, _, reference_mol, _ = _load(original)
    second, _, moving_mol, _ = _load(candidate)
    if original['sequence'] != candidate['sequence'] or original['smiles'] != candidate['smiles']:
        raise ValueError('Matched molecular identities required.')
    if len(set(positions)) < 3 or any(type(i) is not int or not 1 <= i <= len(original['sequence']) for i in positions):
        return {'status': 'unavailable', 'reason': 'At least three mapped pocket residues required.'}
    def points(structure):
        rows = []
        for i in sorted(set(positions)):
            atoms = [a for a in structure[0][0][i-1] if a.name == 'CA']
            if len(atoms) != 1:
                raise ValueError('Ambiguous pocket CA.')
            a = atoms[0]
            rows.append([a.pos.x, a.pos.y, a.pos.z])
        return np.array(rows)
    reference, moving = points(first), points(second)
    if min(np.linalg.matrix_rank(reference-reference.mean(0)), np.linalg.matrix_rank(moving-moving.mean(0))) < 2:
        return {'status': 'unavailable', 'reason': 'Degenerate pocket alignment.'}
    _, rotation, translation = align_points(moving, reference)
    result = ligand_pose_comparison(reference_mol, moving_mol, rotation, translation)
    result['alignment'] = 'caller-frozen binding-site CA positions; not stable-core alignment'
    result['positions'] = sorted(set(positions))
    return result
