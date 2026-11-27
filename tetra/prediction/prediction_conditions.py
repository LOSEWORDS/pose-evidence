"""Validated Boltz pocket/contact constraints with parser-backed token checks."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path


def _distance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 4 <= value <= 20:
        raise ValueError('Constraint max_distance must be finite and between 4 and 20 Angstrom.')
    return value


def validate_pockets(pockets, sequence_length=None):
    if not isinstance(pockets, list) or not 1 <= len(pockets) <= 4:
        raise ValueError('Provide one to four pocket constraints.')
    result = []
    for pocket in pockets:
        if not isinstance(pocket, dict) or set(pocket) != {'binder', 'contacts', 'max_distance', 'force'}:
            raise ValueError('Pocket fields must be binder, contacts, max_distance and force.')
        if not isinstance(pocket['binder'], str) or not pocket['binder']:
            raise ValueError('Pocket binder must name an input chain.')
        if type(pocket['force']) is not bool:
            raise ValueError('Pocket force must be boolean.')
        contacts = pocket['contacts']
        if not isinstance(contacts, list) or not 1 <= len(contacts) <= 20:
            raise ValueError('Pocket requires one to twenty contacts.')
        seen = set()
        for contact in contacts:
            if (not isinstance(contact, list) or len(contact) != 2 or
                    not isinstance(contact[0], str) or type(contact[1]) is not int or contact[1] < 1):
                raise ValueError('Pocket contacts use [protein_chain, one_based_input_position].')
            if sequence_length is not None and contact[0] == 'A' and contact[1] > sequence_length:
                raise ValueError('Pocket residue is outside the input sequence.')
            key = tuple(contact)
            if key in seen:
                raise ValueError('Duplicate pocket contact.')
            seen.add(key)
        result.append({'binder': pocket['binder'], 'contacts': contacts,
                       'max_distance': _distance(pocket['max_distance']), 'force': pocket['force']})
    return result


def validate_contacts(contacts):
    if not isinstance(contacts, list) or not 1 <= len(contacts) <= 20:
        raise ValueError('Provide one to twenty contact constraints.')
    result = []
    for contact in contacts:
        if not isinstance(contact, dict) or set(contact) != {'token1', 'token2', 'max_distance', 'force'}:
            raise ValueError('Contact fields must be token1, token2, max_distance and force.')
        for token in (contact['token1'], contact['token2']):
            if (not isinstance(token, list) or len(token) != 2 or not isinstance(token[0], str)
                    or not token[0] or not (type(token[1]) is int and token[1] > 0 or
                                           isinstance(token[1], str) and token[1])):
                raise ValueError('Contact tokens use [chain, one_based_position_or_ligand_atom_name].')
        if type(contact['force']) is not bool:
            raise ValueError('Contact force must be boolean.')
        result.append({'token1': contact['token1'], 'token2': contact['token2'],
                       'max_distance': _distance(contact['max_distance']), 'force': contact['force']})
    return result


def validate_boltz_payload(payload, mol_dir):
    """Parse emitted YAML through the installed Boltz parser without model execution."""
    import yaml
    from boltz.data.parse.yaml import parse_yaml

    with tempfile.TemporaryDirectory(prefix='tetra-boltz-parse-') as directory:
        root = Path(directory)
        path = root / 'input.yaml'
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        # Boltz constructs a named RDKit molecule from SMILES; this call validates
        # atom tokens against those parser-generated names rather than RDKit indices.
        parse_yaml(path, {}, Path(mol_dir), boltz2=True)
    return True


def map_ligand_atom(smiles: str, fragment_smarts: str, fragment_atom: int) -> dict:
    """Resolve an unambiguous SMARTS atom to the Boltz SMILES atom name."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    fragment = Chem.MolFromSmarts(fragment_smarts)
    if molecule is None or fragment is None:
        raise ValueError('Invalid ligand SMILES or fragment SMARTS.')
    if type(fragment_atom) is not int or not 0 <= fragment_atom < fragment.GetNumAtoms():
        raise ValueError('Fragment atom is outside the SMARTS graph.')
    matches = molecule.GetSubstructMatches(fragment, uniquify=False, useChirality=True)
    indices = {match[fragment_atom] for match in matches}
    if len(indices) != 1:
        raise ValueError('The fragment selector must resolve to exactly one ligand atom.')
    index, = indices
    hydrogenated = Chem.AddHs(molecule)
    rank = Chem.CanonicalRankAtoms(hydrogenated)[index]
    name = hydrogenated.GetAtomWithIdx(index).GetSymbol().upper() + str(rank + 1)
    if len(name) > 4:
        raise ValueError('Boltz atom name exceeds four characters.')
    return {'fragment_smarts': fragment_smarts, 'fragment_atom': fragment_atom,
            'ligand_atom_index': index, 'boltz_atom_name': name,
            'element': hydrogenated.GetAtomWithIdx(index).GetSymbol()}
