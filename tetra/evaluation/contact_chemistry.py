"""Candidate-bound atom identity, ligand graph features and explicit geometry."""

import numpy as np
from rdkit import Chem, rdBase

from tetra.evaluation.structural_metrics import _load

GROUP_PATTERNS = {
    'epoxide': '[O;X2;r3]1[C;r3][C;r3]1',
    'carbonyl': '[CX3]=[OX1]',
    'amide': '[CX3](=[OX1])[NX3]',
    'carboxylic_acid': '[CX3](=[OX1])[OX2H]',
    'carboxylate': '[CX3](=[OX1])[O-]',
    'alcohol': '[C;X4][O;X2H]',
    'guanidine_or_guanidinium': '[NX3][CX3](=[NX2,NX3+])[NX3]',
}


def _atom_table(structure, mapping, ligand, coordinates):
    protein = {}
    for residue, row in zip(structure[0][0], mapping):
        for atom in residue:
            if atom.element.is_hydrogen:
                continue
            atom_id = f"protein:{row['input_position']}:{atom.name.strip()}"
            if atom_id in protein:
                raise ValueError(f'Ambiguous protein atom ID {atom_id}; resolve alternate locations first.')
            point = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            protein[atom_id] = ({**row, 'atom_id': atom_id, 'atom_name': atom.name.strip(),
                                 'element': atom.element.name}, point)
    ligand_atoms = {}
    for atom, point in zip(ligand.GetAtoms(), coordinates):
        index = atom.GetIdx()
        atom_id = f'ligand:{index}'
        ligand_atoms[atom_id] = ({
            'atom_id': atom_id, 'heavy_atom_index': index, 'element': atom.GetSymbol(),
            'hybridization': str(atom.GetHybridization()), 'formal_charge': atom.GetFormalCharge(),
            'neighbors': [
                {'atom_id': f'ligand:{neighbor.GetIdx()}', 'element': neighbor.GetSymbol(),
                 'bond_type': str(ligand.GetBondBetweenAtoms(index, neighbor.GetIdx()).GetBondType())}
                for neighbor in sorted(atom.GetNeighbors(), key=lambda neighbor: neighbor.GetIdx())
            ],
        }, point)
    return protein, ligand_atoms


def _metadata(candidate):
    return {
        'candidate_id': candidate['candidate_id'],
        'candidate_sha256': candidate['sha256'],
        'status': 'measured',
        'atom_id_convention': 'protein:<one-based input residue position>:<atom name>; ligand:<zero-based hydrogen-stripped SDF order>',
    }


def inspect_contact_chemistry(candidate: dict, residue_positions: list[int]) -> dict:
    """Describe the closest heavy-atom pair globally or for each requested residue."""
    if (not isinstance(residue_positions, list) or len(residue_positions) > 16
            or any(type(position) is not int or position < 1 for position in residue_positions)
            or len(set(residue_positions)) != len(residue_positions)):
        raise ValueError('Provide up to 16 distinct one-based residue positions or [] for the global nearest pair.')
    structure, mapping, molecule, coordinates = _load(candidate)
    protein, ligand = _atom_table(structure, mapping, molecule, coordinates)
    if set(residue_positions) - {row['input_position'] for row in mapping}:
        raise ValueError('Unknown protein residue position.')
    group_matches = {
        name: [list(match) for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern))]
        for name, pattern in GROUP_PATTERNS.items()
    }
    ligand_rows = list(ligand.values())
    contacts = []
    for position in residue_positions or [None]:
        rows = [(meta, point) for meta, point in protein.values() if position in (None, meta['input_position'])]
        if not rows or not ligand_rows:
            raise ValueError('No heavy atoms for the requested contact scope.')
        protein_points = np.array([point for _, point in rows])
        distances = np.linalg.norm(protein_points[:, None, :] - coordinates[None, :, :], axis=-1)
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite contact coordinates.')
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        protein_atom = rows[i][0]
        ligand_atom = ligand_rows[j][0]
        memberships = [name for name, matches in group_matches.items() if any(j in match for match in matches)]
        contacts.append({'distance_A': float(distances[i, j]), 'protein': protein_atom,
                         'ligand': {**ligand_atom, 'tracked_groups': memberships}})
    return {
        **_metadata(candidate), 'residue_positions': residue_positions, 'contacts': contacts,
        'contact_scope': 'one nearest heavy-atom pair per requested residue, or one global pair when no residues are supplied',
        'ligand_groups': {name: matches for name, matches in group_matches.items() if matches},
        'group_presence': {name: bool(matches) for name, matches in group_matches.items()},
        'group_panel': dict(GROUP_PATTERNS), 'rdkit_version': rdBase.rdkitVersion,
        'protein_protonation_status': 'not_measured',
        'limitations': [
            'Group matches describe the supplied ligand graph under the listed SMARTS rules.',
            'The panel does not exhaust all chemistry. These observations do not establish reaction mechanism, energetic favorability or biological binding.',
        ],
    }


def measure_atom_geometry(candidate: dict, atom_ids: list[str]) -> dict:
    """Measure distance, angle or signed dihedral for two, three or four ordered atoms."""
    if (not isinstance(atom_ids, list) or len(atom_ids) not in (2, 3, 4)
            or any(not isinstance(atom_id, str) for atom_id in atom_ids)
            or len(set(atom_ids)) != len(atom_ids)):
        raise ValueError('Provide two, three or four distinct ordered atom IDs.')
    protein, ligand = _atom_table(*_load(candidate))
    atoms = {**protein, **ligand}
    for atom_id in atom_ids:
        if atom_id not in atoms:
            raise ValueError(f'Unknown atom ID {atom_id}.')
    points = np.array([atoms[atom_id][1] for atom_id in atom_ids])
    vectors = np.diff(points, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Geometry is undefined for coincident or nonfinite points.')
    if len(atom_ids) == 2:
        kind, unit, value = 'distance', 'angstrom', norms[0]
    elif len(atom_ids) == 3:
        cosine = np.dot(-vectors[0], vectors[1]) / (norms[0] * norms[1])
        kind, unit, value = 'angle', 'degree', np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    else:
        axis = vectors[1] / norms[1]
        left = -vectors[0] + np.dot(vectors[0], axis) * axis
        right = vectors[2] - np.dot(vectors[2], axis) * axis
        if min(np.linalg.norm(left), np.linalg.norm(right)) <= 1e-12:
            raise ValueError('Dihedral is undefined when either defining plane is degenerate.')
        kind, unit, value = 'dihedral', 'degree', np.degrees(np.arctan2(
            np.dot(np.cross(axis, left), right), np.dot(left, right)))
    return {**_metadata(candidate), 'atom_ids': atom_ids, 'geometry': kind, 'value': float(value), 'unit': unit,
            'limitations': ['An explicit coordinate measurement. Reaction compatibility requires a separately justified criterion.']}
