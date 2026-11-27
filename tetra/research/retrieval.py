"""Database retrieval with immutable raw responses and explicit candidate identities."""
import json
from urllib.parse import quote
import requests
from tetra.runtime.inputs import digest
from tetra.runtime.store import now

class Retrieval:
    def __init__(self,store):self.store=store
    def fetch(self,url,params=None):
        response=requests.get(url,params=params,timeout=(10,40))
        if response.status_code!=200:raise ValueError(f'Database HTTP {response.status_code}: {response.url}')
        raw=response.content;key=digest(raw)
        path=self.store.root/'sources'/(key+'.json');path.write_bytes(raw)
        return response.json(),{'source_url':response.url,'retrieved_utc':now(),'raw_sha256':key,'raw_path':str(path)}
    def search_protein(self,name,taxon_id=None):
        if not isinstance(name,str) or not 1<=len(name)<=200:raise ValueError('Protein search name must be 1..200 characters.')
        query=name if taxon_id is None else f'({name}) AND (organism_id:{taxon_id})'
        data,source=self.fetch('https://rest.uniprot.org/uniprotkb/search',{'query':query,'format':'json','size':5})
        candidates=[self._protein(r,source) for r in data.get('results',[])]
        return {'candidates':candidates,'limit':5,'may_be_truncated':len(candidates)==5,'note':'Canonical entries only; isoform/construct must be explicitly resolved before prediction.'}
    def get_protein(self,accession):
        import re
        if not re.fullmatch(r'[A-Z0-9]{6,10}',accession):raise ValueError('Use a canonical UniProt accession; isoforms need explicit sequence input in this version.')
        data,source=self.fetch('https://rest.uniprot.org/uniprotkb/'+accession+'.json')
        return self._protein(data,source)
    def _protein(self,data,source):
        description=data.get('proteinDescription',{})
        name=description.get('recommendedName',{}).get('fullName',{}).get('value')
        if not name:name=description.get('submissionNames',[{}])[0].get('fullName',{}).get('value','')
        isoforms=[x for x in data.get('comments',[]) if x.get('commentType')=='ALTERNATIVE PRODUCTS']
        record={'kind':'protein','accession':data['primaryAccession'],'name':name,'organism':data.get('organism',{}),
                'sequence':data['sequence']['value'],'length':data['sequence']['length'],'isoform_annotations':isoforms,**source}
        result=self.store.save_entity(record)
        return {k:v for k,v in result.items() if k!='sequence'}
    def search_ligand(self,name):
        if not isinstance(name,str) or not 1<=len(name)<=200:raise ValueError('Ligand name must be 1..200 characters.')
        return self._ligands('name/'+quote(name,safe=''))
    def get_ligand(self,cid):
        if type(cid) is not int or cid<=0:raise ValueError('CID must be a positive integer.')
        return self._ligands('cid/'+str(cid))
    def _ligands(self,identifier):
        data,source=self.fetch('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/'+identifier+'/property/IsomericSMILES,InChIKey,MolecularFormula/JSON')
        rows=data['PropertyTable']['Properties'];results=[]
        for row in rows[:5]:
            smiles=row.get('SMILES') or row.get('IsomericSMILES')
            if not smiles:raise ValueError('Database response has no stereochemical SMILES.')
            results.append(self.store.save_entity({'kind':'ligand','cid':row['CID'],'smiles':smiles,'inchikey':row.get('InChIKey'),
                                                  'formula':row.get('MolecularFormula'),**source}))
        return {'candidates':results,'may_be_truncated':len(rows)>5,'note':'Verify stereochemistry, charge and salt form; no standardization is applied.'}
