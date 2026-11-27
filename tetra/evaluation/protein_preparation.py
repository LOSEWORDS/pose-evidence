"""Recorded internal free-terminal topology preparation; never edits input atoms."""
import hashlib,json,math
from pathlib import Path
STANDARD=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())

def atom_map(structure):
    result={}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                key=(chain.name,residue.seqid.num,residue.seqid.icode.strip(),residue.name,atom.name)
                if key in result:raise ValueError('Duplicate protein atom identity')
                result[key]=(atom.element.name,(atom.pos.x,atom.pos.y,atom.pos.z))
    return result

def prepare_terminal_topology(source,output):
    import gemmi
    import openmm as mm
    from openmm import app
    from pdbfixer import PDBFixer
    source=Path(source);output=Path(output);raw=source.read_bytes();st=gemmi.read_structure(str(source))
    if len(st)!=1 or len(st[0])!=1 or len(st[0][0])==0:raise ValueError('Only a single nonempty protein chain is supported')
    if any(r.name not in STANDARD for r in st[0][0]):raise ValueError('Only standard peptide residues supported; no automatic chemistry substitution')
    before=atom_map(st);last=st[0][0][-1];terminal=(st[0][0].name,last.seqid.num,last.seqid.icode.strip(),last.name,'OXT')
    fixer=PDBFixer(filename=str(source),platform=mm.Platform.getPlatformByName('CPU'))
    fixer.findMissingResidues()
    if fixer.missingResidues:raise ValueError('Missing residues are not automatically reconstructed')
    fixer.findMissingAtoms()
    if fixer.missingAtoms:raise ValueError('Missing nonterminal atoms are not supported')
    for residue,names in fixer.missingTerminals.items():
        key=(residue.chain.id,int(residue.id),residue.insertionCode.strip(),residue.name,'OXT')
        if names!=['OXT'] or key!=terminal:raise ValueError('Only the final residue terminal OXT may be added')
    if fixer.missingTerminals:fixer.addMissingAtoms(seed=0)
    output.mkdir(parents=True,exist_ok=True);prepared=output/'prepared-protein.pdb'
    with prepared.open('w') as f:app.PDBFile.writeFile(fixer.topology,fixer.positions,f,keepIds=True)
    after=atom_map(gemmi.read_structure(str(prepared)))
    if set(before)-set(after) or set(after)-set(before) not in (set(),{terminal}):raise ValueError('Unexpected topology change')
    shift=0.0
    for key,(element,xyz) in before.items():
        if after[key][0]!=element:raise ValueError('Protein element changed')
        distance=math.dist(xyz,after[key][1])
        if not math.isfinite(distance) or distance>.0011:raise ValueError('Preparation moved an existing atom')
        shift=max(shift,distance)
    if source.read_bytes()!=raw:raise ValueError('Original protein changed')
    record={'source_sha256':hashlib.sha256(raw).hexdigest(),'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'added_atoms':[{'chain':k[0],'residue':k[1],'insertion':k[2],'residue_name':k[3],'atom':k[4]} for k in sorted(set(after)-set(before))],
        'max_existing_atom_shift_A':shift,'assumption':'Single standard peptide with uncapped free termini; OXT is internal force-field topology only.',
        'output_policy':'Minimized candidate is projected back to the original input atom set; added terminal atoms and hydrogens are not emitted.'}
    (output/'preparation.json').write_text(json.dumps(record,indent=2));return prepared,record
