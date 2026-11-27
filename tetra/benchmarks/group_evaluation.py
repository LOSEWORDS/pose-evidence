"""Offline reference evaluation after confidence-only candidate selection."""
import argparse
import json
from pathlib import Path
from tetra.benchmarks.benchmark import score_model
from tetra.runtime.inputs import digest


def summarize_accuracy(prediction_report,evaluated):
    index={}
    for row in evaluated:
        key=(row['job_id'],row['sample'])
        if key in index:raise ValueError('Duplicate candidate evaluation.')
        index[key]=row
    selected={}
    for chain,entry in prediction_report['per_ligand'].items():
        if not entry['ranking']:
            selected[chain]={'scores':None,'error':'No candidate available'};continue
        choice=entry['ranking'][0]
        row=index.get((choice['job_id'],choice['sample']),{})
        selected[chain]={'job_id':choice['job_id'],'sample':choice['sample'],'seed':choice['seed'],
                         'confidence_rank_score':choice['score'],'scores':row.get('scores'),'error':row.get('error')}
    scored=sum(bool(row.get('scores')) and all(s['success'] is not None for s in row['scores']) for row in evaluated)
    return {'group_id':prediction_report['group_id'],'prediction_complete':prediction_report['complete'],
            'expected_candidates':prediction_report['expected_candidates'],'fully_scored_candidates':scored,
            'missing_evaluations':prediction_report['expected_candidates']-scored,'selected':selected,'evaluations':evaluated,
            'limitations':['Selected candidates are fixed by prediction confidence before reference evaluation.',
                           'For multiple ligands, each selected whole-complex candidate lists all reference-ligand scores; these are not automatically assigned to the selecting input chain.',
                           'Unassigned reference ligands are unavailable; failures remain in the expected denominator.']}


def evaluate_group(report_path,ost,receptor,ligands,output):
    report_path=Path(report_path);report_bytes=report_path.read_bytes();report=json.loads(report_bytes)
    if not report.get('complete'):raise ValueError('Freeze a complete prediction report before reference evaluation.')
    candidates=report['candidates'];keys=[(c['job_id'],c['sample']) for c in candidates]
    if len(keys)!=len(set(keys)) or len(keys)!=report['expected_candidates']:raise ValueError('Candidate manifest is incomplete or duplicated.')
    index={key:c for key,c in zip(keys,candidates)}
    for chain,entry in report['per_ligand'].items():
        ranks=entry['ranking']
        if len(ranks)!=len(keys) or {(c['job_id'],c['sample']) for c in ranks}!=set(keys):raise ValueError('Ranking does not cover all candidates.')
        for ranked in ranks:
            candidate=index[(ranked['job_id'],ranked['sample'])]
            if any(ranked[k]!=candidate[k] for k in ('cif','sha256','seed')):raise ValueError('Ranking candidate identity mismatch.')
            if ranked['score']!=candidate['ligand_ranks'][chain]:raise ValueError('Ranking confidence changed.')
        expected=sorted(candidates,key=lambda c:(-c['ligand_ranks'][chain],c['seed'],c['sample']))
        if [(c['job_id'],c['sample']) for c in ranks]!=[(c['job_id'],c['sample']) for c in expected]:
            raise ValueError('Ranking order does not match confidence policy.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'prediction-report.json').write_bytes(report_bytes)
    evaluated=[]
    for number,candidate in enumerate(candidates):
        row={'job_id':candidate['job_id'],'seed':candidate['seed'],'sample':candidate['sample']}
        try:
            if digest(Path(candidate['cif']).read_bytes())!=candidate['sha256']:raise ValueError('Prediction changed after ranking.')
            row['scores']=score_model(ost,candidate['cif'],receptor,ligands,output/f'candidate-{number:03d}.json')
        except Exception as exc:row['scores']=None;row['error']=str(exc)
        evaluated.append(row)
        (output/'progress.json').write_text(json.dumps(evaluated,indent=2))
    result=summarize_accuracy(report,evaluated);result['prediction_report_sha256']=digest(report_bytes)
    (output/'summary.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--ost',required=True)
    p.add_argument('--receptor',required=True);p.add_argument('--ligands',required=True,nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();result=evaluate_group(a.report,a.ost,a.receptor,a.ligands,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!='evaluations'},indent=2))
if __name__=='__main__':main()
