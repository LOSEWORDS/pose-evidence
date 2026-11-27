"""Conservative physical-validity selection; not an estimator of pose accuracy."""

def select_candidates(candidates,assessments):
    by_id={c['candidate_id']:c for c in candidates}
    for c in candidates:
        a=assessments.get(c['candidate_id'])
        if not a or a.get('candidate_sha256')!=c['sha256']:raise ValueError('Missing or stale assessment.')
    selected=[];decisions=[]
    for root in candidates:
        if root.get('parent_id'):continue
        best=root;visited={root['candidate_id']}
        while True:
            children=[c for c in candidates if c.get('parent_id')==best['candidate_id'] and c['candidate_id'] not in visited]
            replacement=None
            for child in children:
                visited.add(child['candidate_id'])
                old=assessments[best['candidate_id']];new=assessments[child['candidate_id']]
                keys=set(old['checks']);valid=(old.get('status')==new.get('status')=='assessed' and keys and keys==set(new['checks']))
                valid=valid and all(new['checks'][k] in ('pass','fail') and old['checks'][k] in ('pass','fail') for k in keys)
                improved=valid and new['checks'].get('identity')=='pass' and all(new['checks'][k]=='pass' for k in keys if old['checks'][k]=='pass') and any(old['checks'][k]=='fail' and new['checks'][k]=='pass' for k in keys)
                decisions.append({'parent':best['candidate_id'],'child':child['candidate_id'],'accepted':bool(improved)})
                if improved:replacement=child;break
            if replacement is None:break
            best=replacement
        selected.append(best['candidate_id'])
    if not selected:raise ValueError('No original candidate family.')
    return {'selected_ids':selected,'comparisons':decisions,'rule':'Keep original family ordering; replace parent only with complete, nonregressing physical-check improvement.',
            'limitations':['Selection is based on physical validity, not experimental pose accuracy. Unavailable checks are not passes.']}
