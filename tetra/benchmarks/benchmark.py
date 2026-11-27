"""Runs N' Poses manifest and offline accuracy extraction; no LLM dependencies."""
from collections import Counter
import argparse
import copy
import csv
from datetime import date
import json
import math
from pathlib import Path
import subprocess
from tetra.runtime.inputs import digest, validate

CUTOFF='2023-06-01'

def build_manifest(inputs, annotations, cutoff=CUTOFF):
    cutoff_date=date.fromisoformat(cutoff)
    grouped={}
    for row in annotations:
        grouped.setdefault(row['system_id'],[]).append(row)
    if set(grouped)-set(inputs):
        raise ValueError('Inputs and annotations must cover the same system IDs.')
    predictions=[];evaluation=[];excluded=[]
    for system_id,molecules in sorted(inputs.items()):
        if system_id not in grouped:
            excluded.append({'system_id':system_id,'reason':'missing_annotation'})
            continue
        rows=grouped[system_id]
        dates={r['release_date'] for r in rows}
        chains=[r['ligand_instance_chain'] for r in rows]
        if len(dates)!=1 or len(chains)!=len(set(chains)):
            raise ValueError(f'Inconsistent or duplicate annotation: {system_id}')
        released=date.fromisoformat(next(iter(dates)))
        if released<=cutoff_date:
            excluded.append({'system_id':system_id,'reason':'not_after_training_cutoff','release_date':str(released)})
            continue
        if not molecules.get('sequences') or not molecules.get('smiles') or len(molecules['smiles'])!=len(molecules.get('ccd_codes',[])):
            raise ValueError(f'Incomplete molecular specification: {system_id}')
        reasons=[]
        if Counter(molecules['ccd_codes'])!=Counter(r['ligand_ccd_code'] for r in rows):
            reasons.append('annotation_component_mapping_requires_review')
        if len(molecules['sequences'])!=1 or len(molecules['smiles'])!=1:
            reasons.append('current_backend_requires_one_protein_and_one_ligand')
        else:
            try:validate(next(iter(molecules['sequences'].values())),molecules['smiles'][0])
            except ValueError as exc:reasons.append(str(exc))
        predictions.append({'system_id':system_id,'molecules':copy.deepcopy(molecules),
                            'development_compatible':not reasons,'unsupported_reasons':reasons})
        ligands=[]
        for row in rows:
            if row['ligand_is_proper'] not in ('True','False'):raise ValueError('Invalid proper-ligand flag.')
            value=row.get('sucos_shape_pocket_qcov_2023','')
            ligands.append({'chain':row['ligand_instance_chain'],'ccd':row['ligand_ccd_code'],
                            'proper':row['ligand_is_proper']=='True',
                            'published_similarity_2023':float(value) if value else None})
        evaluation.append({'system_id':system_id,'release_date':str(released),'ligands':ligands})
    return {'schema_version':1,'cutoff':cutoff,'prediction_inputs':predictions,'evaluation_metadata':evaluation,
            'excluded':excluded,'counts':{'all_systems':len(inputs),'eligible_systems':len(predictions),
            'development_compatible_systems':sum(p['development_compatible'] for p in predictions),
            'eligible_proper_ligands':sum(l['proper'] for e in evaluation for l in e['ligands'])}}

def is_success(rmsd,lddt_pli):
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (rmsd,lddt_pli)):return None
    if rmsd<0 or not 0<=lddt_pli<=1:return None
    return rmsd<2 and lddt_pli>.8

def extract_scores(raw,ligand_chains):
    indexed={}
    for metric in ('rmsd','lddt_pli'):
        values={}
        for entry in raw.get(metric,{}).get('assigned_scores',[]):
            chain=Path(entry['reference_ligand']).stem
            if chain in values:raise ValueError(f'Duplicate reference ligand assignment: {chain}')
            values[chain]=entry
        indexed[metric]=values
    results=[]
    for chain in ligand_chains:
        rmsd=indexed['rmsd'].get(chain,{}).get('score')
        lddt=indexed['lddt_pli'].get(chain,{}).get('score')
        valid=is_success(rmsd,lddt)
        results.append({'reference_ligand':chain,'rmsd':rmsd if isinstance(rmsd,(int,float)) and math.isfinite(rmsd) else None,
                        'lddt_pli':lddt if isinstance(lddt,(int,float)) and math.isfinite(lddt) else None,
                        'success':valid,'status':'scored' if valid is not None else 'unavailable'})
    return results

def score_model(ost,model,receptor,ligands,output,timeout=600,model_ligands=None):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if any(p.exists() for p in (output,output.with_suffix('.log'),output.with_suffix('.execution.json'))):
        raise ValueError('Refusing to overwrite an existing scoring attempt.')
    paths=[Path(p).resolve() for p in [model,receptor,*ligands]]
    if len(paths)<3 or any(not p.is_file() for p in paths):raise ValueError('Missing model or reference files.')
    command=[str(Path(ost).resolve()),'compare-ligand-structures','-m',str(paths[0]),'-r',str(paths[1]),
             '-rl',*[str(p) for p in paths[2:]],'-o',str(output),'--lddt-pli','--rmsd','--lddt-pli-amc']
    reference_paths=paths[2:]
    if model_ligands is not None:
        explicit=[Path(p).resolve() for p in model_ligands]
        if not explicit or any(not p.is_file() for p in explicit):raise ValueError('Missing explicit model ligand files.')
        command.extend(['-ml',*[str(p) for p in explicit]]);paths.extend(explicit)
    record={'command':command,'input_sha256':{str(p):digest(p.read_bytes()) for p in paths},'timeout_seconds':timeout}
    try:
        with output.with_suffix('.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        record['returncode']=result.returncode
        if result.returncode:raise RuntimeError(f'OpenStructure exited {result.returncode}; see {output.with_suffix(".log")}')
        raw=json.loads(output.read_text())
        return extract_scores(raw,[p.stem for p in reference_paths])
    except Exception as exc:
        record['error']=str(exc);raise
    finally:
        output.with_suffix('.execution.json').write_text(json.dumps(record,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    m=sub.add_parser('manifest');m.add_argument('--inputs',required=True);m.add_argument('--annotations',required=True);m.add_argument('--out',required=True)
    s=sub.add_parser('score');s.add_argument('--ost',required=True);s.add_argument('--model',required=True);s.add_argument('--receptor',required=True);s.add_argument('--ligands',nargs='+',required=True);s.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.action=='manifest':
        inputs=Path(args.inputs);annotations=Path(args.annotations)
        result=build_manifest(json.loads(inputs.read_text()),list(csv.DictReader(annotations.open())))
        folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
        for key in ('prediction_inputs','evaluation_metadata','excluded'):
            path=folder/(key+'.json')
            if path.exists():raise ValueError(f'Refusing to overwrite {path}')
        for key in ('prediction_inputs','evaluation_metadata','excluded'):
            (folder/(key+'.json')).write_text(json.dumps(result.pop(key),indent=2))
        result['sources']={str(p.resolve()):digest(p.read_bytes()) for p in (inputs,annotations)}
        (folder/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['counts'],indent=2))
    else:
        result=score_model(args.ost,args.model,args.receptor,args.ligands,args.out)
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
