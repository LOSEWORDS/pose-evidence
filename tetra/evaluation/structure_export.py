"""Coordinate export for locally produced Boltz single-ligand artifacts."""
import math
from pathlib import Path
from rdkit import Chem
import gemmi

def with_coordinates(molecule,atoms):
    molecule=Chem.RemoveHs(Chem.Mol(molecule))
    names=[a.GetProp('name') for a in molecule.GetAtoms()]
    if len(names)!=len(set(names)) or set(names)!=set(atoms):
        raise ValueError('Predicted atom names do not exactly match the saved Boltz molecular graph.')
    conformer=Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for atom,name in zip(molecule.GetAtoms(),names):
        element,xyz=atoms[name]
        if atom.GetSymbol()!=element or len(xyz)!=3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError('Predicted element or coordinates are invalid.')
        conformer.SetAtomPosition(atom.GetIdx(),xyz)
    molecule.RemoveAllConformers();molecule.AddConformer(conformer)
    return molecule

def export_single_ligand(cif,molecule,output):
    output=Path(output)
    if output.exists():raise ValueError('Output directory already exists.')
    structure=gemmi.read_structure(str(cif))
    if len(structure)!=1 or {c.name for c in structure[0]}!={'A','B'}:raise ValueError('Only verified A protein / B ligand outputs supported.')
    atoms={}
    for residue in structure[0]['B']:
        for atom in residue:
            if atom.element.is_hydrogen:continue
            if atom.name in atoms:raise ValueError('Duplicate ligand atom name.')
            atoms[atom.name]=(atom.element.name,(atom.pos.x,atom.pos.y,atom.pos.z))
    ligand=with_coordinates(molecule,atoms)
    output.mkdir(parents=True)
    with Chem.SDWriter(str(output/'ligand.sdf')) as writer:writer.write(ligand)
    receptor=structure.clone();del receptor[0]['B'];receptor.write_pdb(str(output/'receptor.pdb'))
    return output/'ligand.sdf',output/'receptor.pdb'


def selected_complex(candidate,output):
    from tetra.evaluation.assessment import verify_candidate
    verify_candidate(candidate)
    st=gemmi.read_structure(candidate['receptor']);mol=Chem.SDMolSupplier(candidate['ligand'],removeHs=False)[0]
    if mol is None:raise ValueError('Selected ligand unreadable')
    if len(st)!=1 or len(st[0])!=1:raise ValueError('Expected one receptor chain')
    st[0][0].name='A';chain=gemmi.Chain('B');res=gemmi.Residue();res.name='LIG';res.seqid=gemmi.SeqId(1,' ');res.het_flag='H'
    conf=mol.GetConformer()
    for i,atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum()==1:continue
        a=gemmi.Atom();a.name=f'{atom.GetSymbol()}{i+1}';a.element=gemmi.Element(atom.GetSymbol());p=conf.GetAtomPosition(i);a.pos=gemmi.Position(p.x,p.y,p.z);res.add_atom(a)
    chain.add_residue(res);st[0].add_chain(chain);st.setup_entities();st.make_mmcif_document().write_file(str(output))
