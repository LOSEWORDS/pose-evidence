"""Validate the deliberately narrow single-protein/single-ligand pocket interface."""
import math
from copy import deepcopy


def validate_pockets(pockets: list[dict], sequence_length: int | None = None) -> list[dict]:
    if not isinstance(pockets, list) or not 1 <= len(pockets) <= 2:
        raise ValueError('One or two pocket constraints are supported.')
    for pocket in pockets:
        if not isinstance(pocket, dict) or set(pocket) != {'binder', 'contacts', 'max_distance', 'force'}:
            raise ValueError('Pocket fields must be explicit.')
        if pocket['binder'] != 'B' or pocket['force'] is not True:
            raise ValueError('This adapter supports ligand B and force=true only.')
        distance = pocket['max_distance']
        if type(distance) not in (int, float) or not math.isfinite(distance) or not 4 <= distance <= 20:
            raise ValueError('Pocket max_distance must be finite and 4..20 Angstrom.')
        contacts = pocket['contacts']
        if not isinstance(contacts, list) or not 1 <= len(contacts) <= 20:
            raise ValueError('One to twenty pocket residues are supported.')
        seen = set()
        for contact in contacts:
            if (not isinstance(contact, list) or len(contact) != 2 or contact[0] != 'A'
                    or type(contact[1]) is not int or contact[1] < 1):
                raise ValueError('Contacts use chain A and one-based input sequence indices.')
            if sequence_length is not None and contact[1] > sequence_length:
                raise ValueError('Pocket residue is outside the exact input sequence.')
            if contact[1] in seen:
                raise ValueError('Duplicate pocket residue.')
            seen.add(contact[1])
    return deepcopy(pockets)
