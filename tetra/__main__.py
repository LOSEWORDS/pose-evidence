"""Command-line entry point; explicit human input/confirmation is separate from LLM tools."""
import argparse
import json
from pathlib import Path
import sys
from tetra.runtime.store import Store
from tetra.prediction import backend
from tetra.research.retrieval import Retrieval
from tetra.orchestration.agent import Agent
from tetra.runtime.llm import create_client


def main(argv=None):
    p=argparse.ArgumentParser(description='Training-free protein–ligand agent (CLI milestone)')
    p.add_argument('--state',default=str(backend.PROJECT/'state'))
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register',help='Register a JSON sequence/SMILES explicitly supplied by the user');r.add_argument('file')
    r=sub.add_parser('register-complex',help='Register explicit proteins/ligands with local MSAs');r.add_argument('file')
    for name in ['input','confirm','submit','status','inspect','session']:
        r=sub.add_parser(name);r.add_argument('id')
        if name=='submit':r.add_argument('--config',help='JSON configuration file')
    r=sub.add_parser('search-protein');r.add_argument('name');r.add_argument('--taxon-id',type=int)
    r=sub.add_parser('get-protein');r.add_argument('accession')
    r=sub.add_parser('search-ligand');r.add_argument('name')
    r=sub.add_parser('get-ligand');r.add_argument('cid',type=int)
    r=sub.add_parser('prepare');r.add_argument('protein_id');r.add_argument('ligand_id')
    r=sub.add_parser('chat');r.add_argument('prompt');r.add_argument('--input-id');r.add_argument('--max-calls',type=int,default=8)
    r=sub.add_parser('resume');r.add_argument('session_id');r.add_argument('--message')
    a=p.parse_args(argv);s=Store(a.state)
    try:
        if a.command=='register':
            path=Path(a.file).resolve();data=json.loads(path.read_text())
            if set(data)-{'sequence','smiles','msa'}:raise ValueError('Input file supports only sequence, smiles and optional msa.')
            msa=data.get('msa')
            if msa:msa=str((path.parent/msa).resolve())
            result=s.register_input(data['sequence'],data['smiles'],msa,source={'kind':'user','file':str(path)},confirmed=True)
        elif a.command=='register-complex':
            path=Path(a.file).resolve();data=json.loads(path.read_text())
            if set(data)!={'proteins','ligands'}:raise ValueError('Complex input requires only proteins and ligands.')
            for protein in data['proteins']:
                if protein.get('msa'):protein['msa']=str((path.parent/protein['msa']).resolve())
            result=s.register_complex(data['proteins'],data['ligands'],source={'kind':'user','file':str(path)},confirmed=True)
        elif a.command=='input':result=s.input(a.id)
        elif a.command=='confirm':result=s.confirm_input(a.id)
        elif a.command=='submit':
            cfg=json.loads(Path(a.config).read_text()) if a.config else None
            j=s.submit(a.id,cfg);result=backend.launch(s,j['job_id'])
        elif a.command=='status':result=backend.status(s,a.id)
        elif a.command=='inspect':
            j=backend.status(s,a.id)
            if j['status']!='succeeded':raise ValueError('Prediction is not successful: '+j['status'])
            result=backend.inspect(s.root/'jobs'/a.id,s.input(j['input_id']))
        elif a.command=='session':result=s.session(a.id)
        elif a.command=='search-protein':result=Retrieval(s).search_protein(a.name,a.taxon_id)
        elif a.command=='get-protein':result=Retrieval(s).get_protein(a.accession)
        elif a.command=='search-ligand':result=Retrieval(s).search_ligand(a.name)
        elif a.command=='get-ligand':result=Retrieval(s).get_ligand(a.cid)
        elif a.command=='prepare':
            from tetra.orchestration.agent import Tools
            result=Tools(s).dispatch('prepare_input',{'protein_id':a.protein_id,'ligand_id':a.ligand_id})
        elif a.command=='chat':
            if not 1<=a.max_calls<=20:raise ValueError('max-calls must be 1..20')
            result=Agent(s,create_client(),a.max_calls).start(a.prompt,a.input_id)
        else:result=Agent(s,create_client()).resume(a.session_id,a.message)
        if a.command in ['chat','resume']:
            result={k:v for k,v in result.items() if k!='messages'}
            result['next_step']='Use resume after the job finishes; use input/confirm before resuming an identity confirmation.'
        print(json.dumps(result,indent=2,ensure_ascii=False));return 0
    except Exception as e:
        print(json.dumps({'error':str(e)},ensure_ascii=False));return 1

if __name__=='__main__':sys.exit(main())
