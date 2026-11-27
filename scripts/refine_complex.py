"""CPU-only restrained complex minimization; fails closed on incomplete topology."""
import sys,json,time,importlib.metadata,random
from pathlib import Path
import numpy as np
import gemmi
from rdkit import Chem
import openmm as mm
from openmm import app,unit
from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

start=time.time();request=json.loads(Path(sys.argv[1]).read_text());c=request['candidate'];cfg=request['config'];out=Path(sys.argv[2])
random.seed(0);np.random.seed(0)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tetra.evaluation.protein_preparation import prepare_terminal_topology
prepared,preparation=prepare_terminal_topology(c['receptor'],out/'preparation')
protein=app.PDBFile(str(prepared))
original=list(protein.topology.atoms())
if any(a.element is None for a in original):raise ValueError('Unsupported protein atom.')
# Explicit terminal oxygen is necessary for an uncapped standard peptide topology.
if 'OXT' not in {a.name for a in list(protein.topology.residues())[-1].atoms()}:
    raise ValueError('Protein lacks terminal OXT; supply an explicitly prepared complete topology. No automatic heavy-atom repair.')
ligand=Chem.SDMolSupplier(c['ligand'],removeHs=False)[0]
if ligand is None:raise ValueError('Invalid ligand.')
ligand=Chem.AddHs(ligand,addCoords=True)
Chem.AssignStereochemistryFrom3D(ligand)
off=Molecule.from_rdkit(ligand,allow_undefined_stereo=False,hydrogens_are_explicit=True)
ff=app.ForceField(cfg['protein_forcefield'])
generator=SMIRNOFFTemplateGenerator(molecules=[off],forcefield=cfg['ligand_forcefield'])
ff.registerTemplateGenerator(generator.generator)
model=app.Modeller(protein.topology,protein.positions)
variants=model.addHydrogens(ff,pH=7.0,platform=mm.Platform.getPlatformByName('CPU'))
nprotein=model.topology.getNumAtoms()
model.add(off.to_topology().to_openmm(),np.asarray(ligand.GetConformer().GetPositions())*unit.angstrom)
# Vacuum minimization is a local geometry experiment, not solvent binding-energy prediction.
system=ff.createSystem(model.topology,nonbondedMethod=app.NoCutoff,constraints=None,rigidWater=False)
restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
restraint.addGlobalParameter('k',cfg['restraint_k_kj_mol_nm2'])
for name in ('x0','y0','z0'):restraint.addPerParticleParameter(name)
xyz=np.asarray(model.positions.value_in_unit(unit.nanometer))
for atom in model.topology.atoms():
    if atom.element!=app.element.hydrogen:restraint.addParticle(atom.index,xyz[atom.index].tolist())
system.addForce(restraint)
integrator=mm.VerletIntegrator(0.001*unit.picoseconds)
context=mm.Context(system,integrator,mm.Platform.getPlatformByName('CPU'),{'Threads':str(cfg['threads'])})
context.setPositions(model.positions)
before=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
mm.LocalEnergyMinimizer.minimize(context,10*unit.kilojoule_per_mole/unit.nanometer,cfg['max_iterations'])
state=context.getState(getEnergy=True,getPositions=True)
after=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
positions=np.asarray(state.getPositions().value_in_unit(unit.angstrom))
if not np.isfinite(positions).all() or not np.isfinite([before,after]).all():raise ValueError('Nonfinite minimization output.')
# Map named original atoms; do not emit added hydrogens or change original heavy-atom identity.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tetra.evaluation.atom_mapping import apply_protein_coordinates
st=gemmi.read_structure(c['receptor'])
apply_protein_coordinates(st,[a for a in model.topology.atoms() if a.index<nprotein],positions)
st.write_pdb(str(out/'refined.pdb'))
conf=ligand.GetConformer()
for i in range(ligand.GetNumAtoms()):conf.SetAtomPosition(i,positions[nprotein+i])
with Chem.SDWriter(str(out/'refined.sdf')) as writer:writer.write(Chem.RemoveHs(ligand))
with (out/'prepared-complex.pdb').open('w') as stream:app.PDBFile.writeFile(model.topology,model.positions,stream,keepIds=True)
(out/'metrics.json').write_text(json.dumps({'initial_energy_kj_mol':before,'final_energy_kj_mol':after,
 'max_displacement_A':float(np.linalg.norm(positions-xyz*10,axis=1).max()),'elapsed_seconds':time.time()-start,
 'protein_hydrogen_variants':variants,'preparation':preparation,
 'hydrogen_preparation':'Protein hydrogens at pH 7; ligand protonation and stereo retained.',
 'method':'Restrained vacuum minimization; energy includes restraints; not a binding affinity or pose-accuracy metric.',
 'config':cfg,'versions':{k:importlib.metadata.version(k) for k in ('openmm','openmmforcefields','openff-toolkit','rdkit')}},indent=2,allow_nan=False))
