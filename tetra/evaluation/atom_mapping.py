"""Lossless named-atom coordinate mapping, including residue insertion codes."""

def apply_protein_coordinates(structure,atoms,positions):
    import gemmi
    lookup={}
    for atom in atoms:
        residue=atom.residue
        key=(residue.chain.id,str(residue.id),residue.insertionCode.strip(),residue.name,atom.name)
        if key in lookup:raise ValueError('Duplicate prepared protein atom identity.')
        lookup[key]=positions[atom.index]
    seen=set()
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                key=(chain.name,str(residue.seqid.num),residue.seqid.icode.strip(),residue.name,atom.name)
                if key in seen or key not in lookup:raise ValueError('Original protein atom mapping is not one-to-one.')
                seen.add(key);atom.pos=gemmi.Position(*lookup[key])
