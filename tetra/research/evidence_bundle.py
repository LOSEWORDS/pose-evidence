"""Read existing measurement summaries together; raw evidence remains immutable."""
from tetra.research.evidence_files import read_evidence
from tetra.research.evidence_semantics import evidence_view


def read_bundle(state, folder, evidence_ids):
    if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 64 or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError('Choose 1..64 distinct evidence IDs.')
    rows = {}
    for start in range(0, len(evidence_ids), 4):
        batch = read_evidence(state, folder, evidence_ids[start:start+4], max_bytes=None)
        for key, record in batch['evidence'].items():
            record = evidence_view(record)
            observation = record['observation']
            result = dict(observation.get('result', {}))
            omitted = []
            for field in ('residues', 'per_residue_CA_displacements'):
                if field in result:
                    result.pop(field)
                    omitted.append(field)
            if 'provenance' in result:
                result['provenance'] = {k:v for k,v in result['provenance'].items() if k != 'command'}
            summary_field = 'workflow_summary' if record['epistemic_type'] == 'workflow_record' else 'measurement_summary'
            rows[key] = {'action': observation['action'], summary_field: result,
                         'epistemic_type': record['epistemic_type'],
                         'model_interpretations': record['model_interpretations'],
                         'support_boundary': record['support_boundary'],
                         'measurement_facts': record['measurement_facts'],
                         'measurement_source_sha256': record['measurement_source_sha256'],
                         'selectable_facts': record['selectable_facts'],
                         'raw_evidence': state['evidence_index'][key], 'omitted_detail_fields': omitted}
    return {'evidence_summaries': rows, 'scope': 'Summaries only; read_evidence supplies full residue-level detail. Evidence IDs refer to the immutable originals.'}
