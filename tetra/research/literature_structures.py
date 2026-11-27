"""Link a screened paper to reference structures and measure transferable contacts."""
import hashlib
import json
import re

import requests

SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
DATA = 'https://data.rcsb.org/rest/v1/core'
FILES = 'https://files.rcsb.org/download'


def linked_entries(doi):
    """Find experimental entries whose primary citation has the exact DOI."""
    if not doi:
        return []
    query = {
        'query': {'type': 'terminal', 'service': 'text', 'parameters': {
            'attribute': 'rcsb_primary_citation.pdbx_database_id_DOI',
            'operator': 'exact_match', 'value': doi}},
        'return_type': 'entry',
        'request_options': {'paginate': {'start': 0, 'rows': 3}},
    }
    response = requests.get(SEARCH, params={'json': json.dumps(query)}, timeout=(5, 15))
    if response.status_code == 204:
        return []
    response.raise_for_status()
    return [row['identifier'] for row in response.json().get('result_set', [])
            if re.fullmatch(r'[0-9][A-Za-z0-9]{3}', row['identifier'])][:3]


def _metadata(path):
    response = requests.get(f'{DATA}/{path}', timeout=(5, 15))
    response.raise_for_status()
    return response.json()


def reference_components(pdb_id, target_connectivity):
    """Screen all nonpolymer components before fetching any coordinates."""
    entry = _metadata(f'entry/{pdb_id}')
    entities = entry['rcsb_entry_container_identifiers'].get('non_polymer_entity_ids', [])
    components = set()
    ignored = {'HOH', 'DOD', 'ZN', 'MG', 'CA', 'NA', 'CL', 'K', 'MN', 'FE', 'CU', 'CO', 'NI'}
    for entity in entities:
        row = _metadata(f'nonpolymer_entity/{pdb_id}/{entity}')
        components.add(row['pdbx_entity_nonpoly']['comp_id'])
    eligible = []
    for component in sorted(components - ignored):
        data = _metadata(f'chemcomp/{component}')
        key = data.get('rcsb_chem_comp_descriptor', {}).get('InChIKey', '')
        if not re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]', key):
            raise ValueError(f'Component {component} has unverified chemical identity.')
        if key.split('-', 1)[0] == target_connectivity:
            raise ValueError('Entry contains the target ligand connectivity; coordinates excluded.')
        if data.get('rcsb_chem_comp_info', {}).get('atom_count_heavy', 0) >= 8:
            eligible.append(component)
    return eligible


def _residue_mapping(reference, target):
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(match_score=2, mismatch_score=-1,
                              open_gap_score=-5, extend_gap_score=-0.2)
    alignments = aligner.align(reference, target)
    try:
        if len(alignments) > 64:
            return {}, 0.0, 0.0
    except OverflowError:
        return {}, 0.0, 0.0
    consensus = None
    identity = coverage = 1.0
    for alignment in alignments:
        left, right = alignment.indices
        pairs = [(int(i), int(j)) for i, j in zip(left, right) if i >= 0 and j >= 0]
        if not pairs:
            return {}, 0.0, 0.0
        exact = {(i, j + 1) for i, j in pairs if reference[i] == target[j]}
        consensus = exact if consensus is None else consensus & exact
        identity = min(identity, len(exact) / len(pairs))
        coverage = min(coverage, len(pairs) / len(target))
    return dict(consensus or ()), identity, coverage


def measure_reference(cif_text, components, target_sequence, cutoff=4.0):
    """Measure contacts in deposited chains and map identical aligned residues."""
    import gemmi
    import numpy as np

    structure = gemmi.make_structure_from_block(gemmi.cif.read_string(cif_text).sole_block())
    model = structure[0]
    measurements = []
    ligands_by_chain = {}
    for chain in model:
        for residue in chain:
            if residue.name in components and residue.entity_type == gemmi.EntityType.NonPolymer:
                ligands_by_chain.setdefault(chain.name, []).append(residue)
    for chain in model:
        residues = [residue for residue in chain if residue.entity_type == gemmi.EntityType.Polymer]
        if not residues:
            continue
        sequence = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                           for r in residues)
        mapping, identity, coverage = _residue_mapping(sequence, target_sequence)
        if identity < 0.4 or coverage < 0.7:
            continue
        for ligand in ligands_by_chain.get(chain.name, []):
            ligand_atoms = [atom for atom in ligand if not atom.element.is_hydrogen]
            if not ligand_atoms:
                continue
            points = np.array([list(atom.pos) for atom in ligand_atoms])
            contacts = []
            for i, residue in enumerate(residues):
                atoms = [atom for atom in residue if not atom.element.is_hydrogen]
                if not atoms:
                    continue
                distances = np.linalg.norm(np.array([list(atom.pos) for atom in atoms])[:, None] - points, axis=2)
                a, b = np.unravel_index(distances.argmin(), distances.shape)
                if distances[a, b] <= cutoff:
                    contacts.append({'reference_residue': str(residue.seqid), 'residue_name': residue.name,
                                     'target_position': mapping.get(i), 'distance_A': round(float(distances[a, b]), 3),
                                     'reference_atoms': [atoms[a].name, ligand_atoms[b].name]})
            if contacts:
                measurements.append({'chain': chain.name, 'reference_ligand': ligand.name,
                                     'ligand_residue': str(ligand.seqid), 'sequence_identity': round(identity, 3),
                                     'target_coverage': round(coverage, 3), 'contacts': contacts})
    return measurements[:4]


def search_structure_citations(record, *, fetch=None, screen=None, metadata=None):
    """Find comparable complexes by sequence, then return their primary citations."""
    from rdkit import Chem

    fetch = fetch or requests.get
    screen = screen or reference_components
    metadata = metadata or _metadata
    ligand = Chem.MolFromSmiles(record['smiles'])
    key = Chem.MolToInchiKey(ligand) if ligand is not None else ''
    if not key:
        return {'results': [], 'status': 'unavailable', 'reason': 'Unverified input ligand identity.'}
    query = {
        'query': {'type': 'terminal', 'service': 'sequence', 'parameters': {
            'sequence_type': 'protein', 'value': record['sequence'],
            'identity_cutoff': 0.4, 'evalue_cutoff': 0.01}},
        'return_type': 'polymer_entity',
        'request_options': {'scoring_strategy': 'sequence', 'results_content_type': ['experimental'],
                            'paginate': {'start': 0, 'rows': 12}},
    }
    response = fetch(SEARCH, params={'json': json.dumps(query)}, timeout=(5, 15))
    response.raise_for_status()
    rows = response.json().get('result_set', []) if response.status_code != 204 else []
    results = {}
    excluded = []
    seen = set()
    for row in rows:
        match = re.fullmatch(r'([0-9][A-Za-z0-9]{3})_[0-9]+', row.get('identifier', ''))
        if not match or match[1] in seen:
            continue
        pdb_id = match[1]
        seen.add(pdb_id)
        try:
            if not screen(pdb_id, key.split('-', 1)[0]):
                raise ValueError('No suitable reference ligand.')
            citation = metadata(f'entry/{pdb_id}').get('rcsb_primary_citation') or {}
            pmid = str(citation.get('pdbx_database_id_PubMed', ''))
            if not pmid.isdigit():
                raise ValueError('No primary citation PMID.')
        except (requests.RequestException, ValueError, KeyError) as error:
            excluded.append({'pdb_id': pdb_id, 'reason': str(error)})
            continue
        hit = results.setdefault(pmid, {'source_id': f'MED:{pmid}', 'title': citation.get('title'),
                                       'doi': citation.get('pdbx_database_id_DOI'), 'pdb_ids': []})
        hit['pdb_ids'].append(pdb_id)
    return {'route': 'sequence', 'results': list(results.values())[:5], 'excluded': excluded}


def pocket_drafts(references):
    """Summarize measured sites separately without merging different reference poses."""
    drafts = []
    for entry in references.get('entries', []):
        for measurement in entry.get('measurements', []):
            positions = sorted({row['target_position'] for row in measurement['contacts']
                                if row.get('target_position') is not None})
            if positions:
                drafts.append({
                    'pdb_id': entry['pdb_id'], 'coordinate_sha256': entry['coordinate_sha256'],
                    'chain': measurement['chain'], 'reference_ligand': measurement['reference_ligand'],
                    'target_positions': positions,
                    'scope': 'Mapped reference proximity. Choose applicable residues and justify the distance bound.',
                })
    return drafts


def structural_references(source, record):
    from rdkit import Chem

    doi = source.get('doi')
    ligand = Chem.MolFromSmiles(record['smiles'])
    target_key = Chem.MolToInchiKey(ligand) if ligand is not None else ''
    if not target_key:
        return {'status': 'unavailable', 'reason': 'Cannot verify target ligand identity.'}
    try:
        pdb_ids = source.get('pdb_ids') or linked_entries(doi)
    except (requests.RequestException, ValueError, KeyError) as error:
        return {'status': 'unavailable', 'reason': f'RCSB citation search failed: {error}'}
    entries = []
    for pdb_id in pdb_ids:
        row = {'pdb_id': pdb_id, 'url': f'https://www.rcsb.org/structure/{pdb_id}'}
        try:
            components = reference_components(pdb_id, target_key.split('-', 1)[0])
            if not components:
                raise ValueError('No suitable nonpolymer ligand components.')
            response = requests.get(f'{FILES}/{pdb_id}.cif', timeout=(5, 20))
            response.raise_for_status()
            measurements = measure_reference(response.text, components, record['sequence'])
            row.update(coordinate_sha256=hashlib.sha256(response.content).hexdigest(),
                       measurements=measurements)
        except (requests.RequestException, ValueError, KeyError, RuntimeError, IndexError) as error:
            row['reason'] = str(error)
        entries.append(row)
    usable = any(contact['target_position'] is not None for row in entries
                 for measurement in row.get('measurements', []) for contact in measurement['contacts'])
    return {'status': 'retrieved' if usable else 'unavailable', 'doi': doi,
            'cutoff_A': 4.0, 'entries': entries,
            'scope': 'Reference ligand proximity in deposited coordinates. Target positions are exact '
                     'residue matches shared by all considered optimal sequence alignments. Ligand atom labels refer to the reference '
                     'compound only and do not establish target ligand contacts.'}
