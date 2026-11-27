"""Experimental CNN–empirical agreement selector; never reads reference metrics."""
import hashlib
import math
from pathlib import Path
from tetra.evaluation.assessment import CHECKS, verify_candidate

VERSION = 'cnn-empirical-agreement-v1'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def select(rows, incumbent_id):
    """Compare S0 and S1 on the same admissible pool without modifying Agent state.

    revision_eligible must be resolved by the caller from the original revision
    contracts; True also denotes a declared non-contract docking diagnostic.
    Exact receptor bytes are deliberately stricter than coordinate equivalence.
    """
    eligible, excluded = {}, []
    seen = set()
    for row in rows:
        candidate = row['candidate']
        key = candidate['candidate_id']
        if key in seen:
            raise ValueError('Duplicate candidate ID')
        seen.add(key)
        verify_candidate(candidate)
        assessment = row['assessment']
        passed = (assessment.get('candidate_sha256') == candidate['sha256']
                  and assessment.get('status') == 'assessed'
                  and all(assessment.get('checks', {}).get(k) == 'pass' for k in ('identity', *CHECKS))
                  and row.get('revision_eligible') is True)
        if passed:
            eligible[key] = row
        else:
            excluded.append(key)
    if incumbent_id not in eligible:
        raise ValueError('Incumbent is not eligible; no implicit replacement or invalid delivery')

    result = {'policy_version':VERSION,'incumbent_id':incumbent_id,
              'eligible_ids':sorted(eligible),'excluded_ids':sorted(excluded)}
    scores = {k:r.get('score', {}) for k,r in eligible.items()}
    profiles = {s.get('profile') for s in scores.values()}
    valid_cnn = all(s.get('candidate_sha256') == eligible[k]['candidate']['sha256']
                    and finite(s.get('cnn_score')) and 0 <= s['cnn_score'] <= 1
                    and isinstance(s.get('profile'),str) and bool(s['profile']) for k,s in scores.items())
    if not valid_cnn or len(profiles) != 1:
        fallback = {'selected_id':incumbent_id,'reason':'INCOMPLETE_OR_INCOMPARABLE_SCORING'}
        return {**result,'s0':dict(fallback),'s1':dict(fallback),
                'alternatives':[k for k in sorted(eligible) if k != incumbent_id],
                'scope':'Fallback to supplied valid original incumbent; no complete ranking available'}

    ranked = sorted(eligible,key=lambda k:(-scores[k]['cnn_score'],k))
    top = ranked[0]
    result['s0'] = {'selected_id':top,'reason':'CNN_TOP1','ranked_ids':ranked}
    chosen = incumbent_id
    if not all(finite(s.get('affinity')) for s in scores.values()):
        reason = 'INCOMPLETE_EMPIRICAL_SCORING'
    elif top == incumbent_id:
        reason = 'INCUMBENT_IS_CNN_TOP1'
    else:
        challenger, incumbent = eligible[top]['candidate'],eligible[incumbent_id]['candidate']
        receptor_same = Path(challenger['receptor']).read_bytes() == Path(incumbent['receptor']).read_bytes()
        identity_same = all(challenger[k] == incumbent[k] for k in ('sequence','smiles'))
        if not identity_same:
            reason = 'DIFFERENT_MOLECULAR_INPUT'
        elif not receptor_same:
            reason = 'DIFFERENT_RECEPTOR_BYTES'
        elif scores[top]['cnn_score'] <= scores[incumbent_id]['cnn_score']:
            reason = 'NO_STRICT_CNN_IMPROVEMENT'
        elif scores[top]['affinity'] >= scores[incumbent_id]['affinity']:
            reason = 'EMPIRICAL_DISAGREEMENT'
        else:
            chosen, reason = top, 'CNN_AND_EMPIRICAL_IMPROVE'
    result.update(s1={'selected_id':chosen,'challenger_id':top,'reason':reason},
                  alternatives=[k for k in ranked if k != chosen],
                  receptor_hashes={k:hashlib.sha256(Path(r['candidate']['receptor']).read_bytes()).hexdigest()
                                   for k,r in eligible.items()},
                  limitations=['Uncalibrated experimental risk-control rule, not demonstrated accuracy improvement.',
                               'Only CNN Top-1 is challenged; no runner-up substitution.',
                               'Different receptor bytes cannot promote; unchanged coordinates with different serialization also fall back.',
                               'No statistical near-tie threshold; strict comparison of recorded values only.'])
    return result
