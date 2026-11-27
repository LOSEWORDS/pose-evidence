"""Experimental torsion relaxation: fixed receptor, restrained ligand core, mobile fragment."""
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
roundtrip = off.to_rdkit()
# Normalize equivalent aromatic/Kekule representations before indexed graph comparison.
Chem.SanitizeMol(ligand)
Chem.SanitizeMol(roundtrip)
if [(a.GetAtomicNum(), a.GetFormalCharge()) for a in ligand.GetAtoms()] != [(a.GetAtomicNum(), a.GetFormalCharge()) for a in roundtrip.GetAtoms()]:
    raise ValueError('Parameterization changed atom order or formal charge.')
if sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),b.GetBondTypeAsDouble()) for b in ligand.GetBonds()) != sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),b.GetBondTypeAsDouble()) for b in roundtrip.GetBonds()):
    raise ValueError('Parameterization changed indexed bond graph.')
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
for index in range(nprotein):
    system.setParticleMass(index, 0 * unit.dalton)
core_mode = request.get('core_mode', 'harmonic')
if core_mode not in ('harmonic', 'exact_fixed'):
    raise ValueError('Unsupported core mode.')
core_indices = request['core_atoms']
if not core_indices or len(set(core_indices)) != len(core_indices) or any(type(i) is not int or i < 0 or i >= ligand.GetNumHeavyAtoms() for i in core_indices):
    raise ValueError('Invalid heavy-atom core mapping.')
for index in core_indices:
    if core_mode == 'exact_fixed':
        system.setParticleMass(nprotein + index, 0 * unit.dalton)
    else:
        restraint.addParticle(nprotein + index, xyz[nprotein + index].tolist())
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
import shutil
if np.max(np.abs(positions[:nprotein] - xyz[:nprotein] * 10)) > 1e-6:
    raise ValueError('Fixed receptor moved.')
shutil.copyfile(c['receptor'],out/'refined.pdb')
conf=ligand.GetConformer()
for i in range(ligand.GetNumAtoms()):conf.SetAtomPosition(i,positions[nprotein+i])
with Chem.SDWriter(str(out/'refined.sdf')) as writer:writer.write(Chem.RemoveHs(ligand))
with (out/'prepared-complex.pdb').open('w') as stream:app.PDBFile.writeFile(model.topology,model.positions,stream,keepIds=True)
(out/'metrics.json').write_text(json.dumps({'initial_energy_kj_mol':before,'final_energy_kj_mol':after,
 'max_displacement_A':float(np.linalg.norm(positions-xyz*10,axis=1).max()),'elapsed_seconds':time.time()-start,
 'protein_hydrogen_variants':variants,'preparation':preparation,
 'hydrogen_preparation':'Protein hydrogens at pH 7; ligand protonation and stereo retained.',
 'method':'Fixed receptor, core-restrained ligand vacuum minimization. Energy is not binding affinity or pose accuracy.', 'core_atoms':request['core_atoms'], 'receptor_max_displacement_A':float(np.max(np.abs(positions[:nprotein]-xyz[:nprotein]*10))),
 'core_mode':core_mode, 'rdkit_to_openmm':{str(i):nprotein+i for i in range(ligand.GetNumAtoms())},
 'core_max_displacement_A':float(np.linalg.norm(positions[nprotein+np.array(core_indices)]-xyz[nprotein+np.array(core_indices)]*10,axis=1).max()),
 'minimization_status':'returned_normally_convergence_not_established',
 'config':cfg,'versions':{k:importlib.metadata.version(k) for k in ('openmm','openmmforcefields','openff-toolkit','rdkit')}},indent=2,allow_nan=False))
