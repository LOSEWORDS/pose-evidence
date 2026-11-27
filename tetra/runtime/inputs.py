"""Validate explicit molecular inputs without changing their identity."""
import csv
import io
import hashlib
import json
from pathlib import Path
from rdkit import Chem

AMINO_ACIDS=set('ACDEFGHIKLMNPQRSTVWY')
DEFAULT_CONFIG={'seed':42,'diffusion_samples':1,'recycling_steps':3,'sampling_steps':200,'timeout_seconds':1800}

def digest(data):
    return hashlib.sha256(data if isinstance(data,bytes) else data.encode()).hexdigest()

def stable(data):
    return json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def validate(sequence,smiles,msa=None,*,allow_unspecified_stereo=False,max_residues=1500):
    if not isinstance(sequence,str) or not sequence or not set(sequence)<=AMINO_ACIDS:
        raise ValueError('Sequence must contain only the 20 standard uppercase amino acids; no automatic trimming or substitutions.')
    if len(sequence)>max_residues:raise ValueError(f'Sequence exceeds {max_residues}-residue resource limit; do not truncate it.')
    if not isinstance(smiles,str) or not smiles:raise ValueError('SMILES is required.')
    molecule=Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumHeavyAtoms()==0:raise ValueError('SMILES is invalid.')
    if any(a.GetAtomicNum()==0 or a.GetIsotope()!=0 or a.GetNumRadicalElectrons()!=0 or a.HasQuery() for a in molecule.GetAtoms()):
        raise ValueError('Wildcard/query atoms, isotope labels and radical species are not preserved by this backend adapter.')
    if len(Chem.GetMolFrags(molecule))!=1:raise ValueError('Disconnected ligand/salt requires explicit molecular-form clarification.')
    if molecule.GetNumHeavyAtoms()>150:raise ValueError('Ligand exceeds initial 150-heavy-atom resource limit.')
    if not allow_unspecified_stereo and any(x.specified==Chem.StereoSpecified.Unspecified for x in Chem.FindPotentialStereo(molecule)):
        raise ValueError('Unspecified stereochemistry: provide the intended stereoisomer before prediction.')
    msa_bytes=None
    if msa:
        msa_bytes=Path(msa).read_bytes()
        if Path(msa).suffix.lower()=='.csv':
            reader=csv.DictReader(io.StringIO(msa_bytes.decode()))
            if reader.fieldnames!=['key','sequence']:raise ValueError('CSV MSA must have key,sequence columns.')
            rows=list(reader)
            if not rows or any(not r.get('sequence') or r.get(None) for r in rows):raise ValueError('Malformed CSV MSA.')
            try:
                for row in rows:int(row['key'])
            except (ValueError,TypeError):raise ValueError('Invalid CSV MSA pairing key.')
            first=rows[0]['sequence']
        else:
            lines=msa_bytes.decode().splitlines();query=[];started=False
            for line in lines:
                if line.startswith('>'):
                    if started:break
                    started=True
                elif started:query.append(line.strip())
            first=''.join(query)
        first=''.join(c for c in first if not c.islower()).replace('-','').replace('.','')
        if first!=sequence:raise ValueError('MSA query does not match the complete input sequence.')
    return {'sequence':sequence,'smiles':smiles,'canonical_smiles':Chem.MolToSmiles(molecule,isomericSmiles=True),
            'protein_length':len(sequence),'ligand_heavy_atoms':molecule.GetNumHeavyAtoms(),
            'msa_sha256':digest(msa_bytes) if msa_bytes else None},msa_bytes

def validate_config(config=None):
    result=DEFAULT_CONFIG.copy()
    if config:
        if set(config)-set(result)-{'pocket_constraints','contact_constraints','use_potentials'}:raise ValueError('Unknown prediction configuration keys.')
        result.update(config)
    limits={'seed':(0,2147483647),'diffusion_samples':(1,5),'recycling_steps':(1,10),'sampling_steps':(1,400),'timeout_seconds':(1,3600)}
    for key,(lo,hi) in limits.items():
        if type(result[key]) is not int or not lo<=result[key]<=hi:raise ValueError(f'Invalid {key}: permitted range {lo}..{hi}')
    if 'use_potentials' in result and type(result['use_potentials']) is not bool:
        raise ValueError('use_potentials must be an explicit boolean.')
    if 'pocket_constraints' in result:
        from tetra.prediction.prediction_conditions import validate_pockets
        result['pocket_constraints']=validate_pockets(result['pocket_constraints'])
    if 'contact_constraints' in result:
        from tetra.prediction.prediction_conditions import validate_contacts
        result['contact_constraints']=validate_contacts(result['contact_constraints'])
    return result
