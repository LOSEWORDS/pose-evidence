"""Single-input resolution. Database ambiguity is an error, never confirmation by LLM."""
from tetra.research.retrieval import Retrieval


def unique(result):
    rows=result.get('candidates',[])
    if result.get('may_be_truncated') or len(rows)!=1:
        raise ValueError('Molecular identity is unresolved: provide an explicit accession/CID or sequence/SMILES.')
    return rows[0]


def resolve_request(request,store,retrieval=None):
    allowed={'sequence','smiles','protein_name','taxon_id','accession','cid','ligand_name','msa','canonical'}
    try:
        if not isinstance(request,dict) or set(request)-allowed:raise ValueError('Unsupported input fields.')
        r=retrieval or Retrieval(store);sources={'kind':'single_input','request':request}
        if sum(bool(request.get(k)) for k in ('sequence','accession','protein_name'))!=1:raise ValueError('Specify exactly one protein identity.')
        if sum(bool(request.get(k)) for k in ('smiles','cid','ligand_name'))!=1:raise ValueError('Specify exactly one ligand identity.')
        sequence=request.get('sequence')
        if not sequence:
            if request.get('accession'):
                p=r.get_protein(request['accession'])
            else:
                if type(request.get('taxon_id')) is not int:raise ValueError('A protein name requires an explicit species taxon_id.')
                p=unique(r.search_protein(name=request['protein_name'],taxon_id=request['taxon_id']))
            p=store.entity(p['entity_id'])
            if request.get('taxon_id') and p.get('organism',{}).get('taxonId')!=request['taxon_id']:raise ValueError('Retrieved species does not match input.')
            if p.get('isoform_annotations') and request.get('canonical') is not True and not request.get('accession'):raise ValueError('Isoform unresolved; specify accession or sequence.')
            sequence=p['sequence'];sources['protein']=p
        smiles=request.get('smiles')
        if not smiles:
            l=unique(r.get_ligand(request['cid']) if request.get('cid') else r.search_ligand(request['ligand_name']))
            smiles=l['smiles'];sources['ligand']=l
        record=store.register_input(sequence,smiles,msa=request.get('msa'),source=sources,confirmed=True)
        return {'status':'resolved','input_id':record['input_id'],'protein_length':len(sequence),'provenance':sources}
    except (ValueError,KeyError,TypeError) as e:
        return {'status':'input_error','reason':str(e)}
