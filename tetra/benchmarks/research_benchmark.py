"""Isolated full-agent cases followed by strictly offline paired evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest
from tetra.benchmarks.overnight_agent import export_candidates, ordered_candidates, terminate_tree

PROJECT = Path(__file__).resolve().parents[2]
CASE_SECONDS = 2700
MAX_STEPS = 64


def read(path: Path):
    return json.loads(path.read_text())


def baseline_candidate(provenance: dict) -> dict:
    from tetra.evaluation.assessment import verify_candidate
    candidate = dict(provenance['initial_candidates'][0], sha256=provenance['export_sha256'][0])
    verify_candidate(candidate)
    return candidate


def prepare_exports(baseline: Path, row: dict, folder: Path, expected_hash: str) -> tuple:
    from tetra.evaluation.assessment import fingerprint
    path = baseline / 'cases' / (row['system_id'] + '.prediction-report')
    if digest(path.read_bytes()) != expected_hash:
        raise ValueError('Frozen baseline report changed.')
    prediction = read(path)
    exports = export_candidates(baseline, row, prediction, folder / 'exports')
    if len(exports) != 5:
        raise ValueError('The frozen initial pool must contain exactly five candidates.')
    provenance = {
        'baseline_top1_sha256': ordered_candidates(prediction)[0]['sha256'],
        'baseline_report_sha256': expected_hash,
        'initial_candidates': exports,
        'export_sha256': [fingerprint(item['receptor'], item['ligand']) for item in exports],
    }
    atomic_json(folder / 'provenance.json', provenance)
    return exports, provenance


def collect_costs(folder: Path) -> dict:
    executions = [read(path) for path in (folder / 'state/jobs').glob('*/execution.json')]
    attempts = [read(path) for path in (folder / 'state/research').glob('*/refinements/*/attempt.json')]
    return {
        'boltz_execution_count': len(executions),
        'boltz_execution_seconds': sum(item.get('elapsed_seconds', 0) for item in executions),
        'openmm_attempts': len(attempts),
        'openmm_reported_seconds': sum(item.get('metrics', {}).get('elapsed_seconds', 0) for item in attempts),
        'openmm_attempts_without_timing': sum('elapsed_seconds' not in item.get('metrics', {}) for item in attempts),
    }


def cleanup_jobs(folder: Path) -> None:
    """Reap only owned jobs, including workers detached from an exited case process."""
    import psutil
    from tetra.runtime.store import Store
    root = folder / 'state'
    if not (root / 'state.sqlite').exists():
        return
    store = Store(root)
    for directory in (root / 'jobs').iterdir():
        job = store.job(directory.name)
        if job['status'] not in ('running', 'queued') or not job.get('worker_pid'):
            continue
        try:
            process = psutil.Process(job['worker_pid'])
            if abs(process.create_time() - job['worker_started']) >= 0.01:
                continue
            owned = process.children(recursive=True) + [process]
            for child in owned:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(owned, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
        except psutil.NoSuchProcess:
            pass
        store.update_job(directory.name, 'interrupted', error='Case ended; owned prediction worker cleaned up.')


def sample_memory(stream) -> None:
    try:
        sample = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        if sample.returncode:
            raise RuntimeError('nvidia-smi failed')
        for line in sample.stdout.splitlines():
            stream.write(f'{time.time()},{line}\n')
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        stream.write(f'# {time.time()},unavailable,{type(error).__name__}\n')
    stream.flush()


def build_request(baseline: Path, row: dict, exports: list) -> dict:
    molecules = row['molecules']
    if len(molecules['sequences']) != 1 or len(molecules['smiles']) != 1:
        raise ValueError('Only a single protein–ligand pair is supported.')
    chain, sequence = next(iter(molecules['sequences'].items()))
    msa = baseline / 'extracted/msa_files' / row['system_id'].lower() / (chain + '.csv')
    if not msa.is_file():
        raise ValueError('Missing cached MSA; no network MSA fallback in this campaign.')
    smiles = molecules['smiles'][0]
    return {
        'goal': 'Evaluate the supplied five initial protein–ligand complex predictions. '
                'Assess candidates, integrate tool feedback into session-local memory, and identify actionable issues. '
                'Use evidence-based revision when justified: local geometry repair, pocket-conditioned contact testing, or independent receptor-geometry-guided site exploration. '
                'Resolve the provided input before any resampling; reuse its cached MSA. '
                'Measure and compare revised structures and evaluate the frozen acceptance contract. '
                'Do not force revisions or invent biological evidence. Preserve molecular identity and original candidates. '
                'When available, use GNINA score_candidates on the final pool, then select with the policy and finish, or report why no candidate is acceptable. '
                'No experimental target structures or scores are available to this decision process.',
        'input': {'sequence': sequence, 'smiles': smiles, 'msa': str(msa)},
        'candidates': [dict(item, sequence=sequence, smiles=smiles) for item in exports],
    }


def summarize_rows(rows: list) -> dict:
    paired = [row for row in rows if 'agent' in row and 'baseline' in row]
    return {
        'denominator': len(rows),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'agent_pose_successes': sum(row.get('agent', {}).get('success') is True for row in rows),
        'baseline_pose_successes': sum(row.get('baseline', {}).get('success') is True for row in rows),
        'paired_scored': len(paired),
        'paired_agent_successes': sum(row['agent']['success'] for row in paired),
        'paired_baseline_successes': sum(row['baseline']['success'] for row in paired),
        'success_gained': sum(row['agent']['success'] and not row['baseline']['success'] for row in paired),
        'success_lost': sum(not row['agent']['success'] and row['baseline']['success'] for row in paired),
    }


def snapshot(root: Path, rows: list, phase: str, started: float) -> dict:
    outcomes = []
    for row in rows:
        path = root / 'cases' / row['system_id'] / 'status.json'
        outcomes.append(read(path) if path.exists() else {'system_id': row['system_id'], 'status': 'pending'})
    result = dict(summarize_rows(outcomes), phase=phase, elapsed_seconds=time.time() - started,
                  updated_epoch=time.time(), cases=outcomes)
    atomic_json(root / 'status.json', result)
    return result


def outcome_status(agent_status: str, selected_ids: list) -> str:
    if agent_status not in ('finished', 'stopped'):
        return agent_status
    return 'completed' if selected_ids else 'abstained'


def worker(baseline: Path, root: Path, sid: str) -> None:
    from tetra.orchestration.scientific_agent import ScientificAgent, AgentConfig
    from tetra.runtime.store import Store
    folder = root / 'cases' / sid
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['TETRA_LLM_EVENT_LOG'] = str(folder / 'api-events.jsonl')
    os.environ['TETRA_COST_CASE_ID'] = sid
    start = time.time()
    result = {'system_id': sid, 'status': 'running', 'started_epoch': start,
              'gpu': int(os.environ['TETRA_GPU_INDEX'])}
    atomic_json(folder / 'status.json', result)
    try:
        row = next(item for item in read(baseline / 'prediction_inputs.json') if item['system_id'] == sid)
        expected = read(root / 'config.json')['baseline_report_hashes'][sid]
        exports, _ = prepare_exports(baseline, row, folder, expected)
        request = build_request(baseline, row, exports)
        atomic_json(folder / 'request.json', request)
        store = Store(folder / 'state')
        agent = ScientificAgent(store, config=AgentConfig(max_steps=MAX_STEPS, refinement=True))
        state = agent.run(request)
        while state['status'] in ('running', 'waiting_prediction'):
            atomic_json(folder / 'progress.json', {
                'session_id': state['session_id'], 'status': state['status'],
                'llm_calls': state['llm_calls'], 'prediction_count': state['prediction_count'],
                'refinement_count': state['refinement_count'], 'elapsed_seconds': time.time() - start,
            })
            time.sleep(5)
            state = agent.resume(state['session_id'])
        report_path = agent.folder(state) / 'report.json'
        report = read(report_path)
        selected_ids = (report.get('selection') or {}).get('selected_ids', [])
        # Commit outcome before any experimental target is opened by the offline stage.
        selected = report['candidates'][selected_ids[0]] if selected_ids else None
        atomic_json(folder / 'selected.json', {
            'candidate_id': selected_ids[0] if selected_ids else None,
            'sha256': selected['sha256'] if selected else None,
            'report_sha256': digest(report_path.read_bytes()),
        })
        result.update(status=outcome_status(state['status'], selected_ids), report=str(report_path),
                      agent_status=state['status'], task_complete=report['task_complete'],
                      execution_completed=report['execution_completed'],
                      computational_acceptance=report['computational_acceptance'],
                      llm_calls=state['llm_calls'], prediction_count=state['prediction_count'],
                      refinement_count=state['refinement_count'], usage=report['usage'])
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    cleanup_jobs(folder)
    result['costs'] = collect_costs(folder)
    result.update(finished_epoch=time.time(), elapsed_seconds=time.time() - start)
    atomic_json(folder / 'status.json', result)


def score_candidate(baseline: Path, folder: Path, sid: str, label: str, candidate: dict) -> dict:
    from tetra.evaluation.assessment import verify_candidate
    from tetra.benchmarks.benchmark import score_model, is_success
    from tetra.benchmarks.overnight_agent import offline_metric
    verify_candidate(candidate)
    metadata = next(row for row in read(baseline / 'evaluation_metadata.json') if row['system_id'] == sid)
    chain = next(item['chain'] for item in metadata['ligands'] if item['proper'])
    references = [path for path in (baseline / 'extracted').rglob('receptor.cif')
                  if path.parent.name.lower() == sid.lower()]
    if len(references) != 1:
        raise ValueError('Expected unique offline reference.')
    out = folder / (label + '-score.json')
    score_model(PROJECT / 'scripts/ost_scoring_compat.py', candidate['receptor'], references[0],
                sorted((references[0].parent / 'ligand_files').glob('*.sdf')), out,
                timeout=180, model_ligands=[candidate['ligand']])
    raw = read(out)
    rmsd = [item for item in raw.get('rmsd', {}).get('assigned_scores', [])
            if Path(item['reference_ligand']).stem == chain]
    lddt = [item for item in raw.get('lddt_pli', {}).get('assigned_scores', [])
            if Path(item['reference_ligand']).stem == chain]
    if len(rmsd) != 1 or len(lddt) != 1:
        raise ValueError('Scoring assignment unavailable.')
    if rmsd[0]['model_ligand'] != str(Path(candidate['ligand']).resolve()):
        raise ValueError('Wrong model ligand assignment.')
    metrics = offline_metric(label, chain, rmsd[0], lddt[0])
    success = is_success(metrics['rmsd'], metrics['lddt_pli'])
    if success is None:
        raise ValueError('Scoring metrics unavailable.')
    verify_candidate(candidate)
    return {'success': success, 'metrics': metrics, 'sha256': candidate['sha256']}


def offline_case(baseline: Path, root: Path, row: dict) -> None:
    from tetra.evaluation.assessment import fingerprint
    sid = row['system_id']
    folder = root / 'cases' / sid
    result = read(folder / 'status.json')
    start = time.time()
    # Score baseline independently even for an abstaining or failed agent.
    try:
        if not (folder / 'provenance.json').exists():
            expected = read(root / 'config.json')['baseline_report_hashes'][sid]
            _, provenance = prepare_exports(baseline, row, folder / 'offline-baseline', expected)
        else:
            provenance = read(folder / 'provenance.json')
        prediction_path = baseline / 'cases' / (sid + '.prediction-report')
        if digest(prediction_path.read_bytes()) != provenance['baseline_report_sha256']:
            raise ValueError('Frozen baseline report changed.')
        original = baseline_candidate(provenance)
        result['baseline'] = score_candidate(baseline, folder, sid, 'baseline', original)
    except Exception as error:
        result['baseline_score_error'] = str(error)
    if result.get('report'):
        try:
            report_path = Path(result['report'])
            commit = read(folder / 'selected.json')
            if digest(report_path.read_bytes()) != commit['report_sha256']:
                raise ValueError('Committed agent report changed.')
            if commit['candidate_id']:
                report = read(report_path)
                candidate = report['candidates'][commit['candidate_id']]
                if candidate['sha256'] != commit['sha256']:
                    raise ValueError('Committed selection changed.')
                result['agent'] = score_candidate(baseline, folder, sid, 'agent', candidate)
        except Exception as error:
            result['agent_score_error'] = str(error)
    result['costs'] = collect_costs(folder)
    result['offline_seconds'] = time.time() - start
    atomic_json(folder / 'status.json', result)


def recovery_rows(source: Path, rows: list) -> list:
    selected = []
    for row in rows:
        path = source / 'cases' / row['system_id'] / 'status.json'
        if not path.exists():
            continue
        status = read(path)
        if status.get('report') and read(Path(status['report']))['status'] == 'service_error':
            selected.append(row)
    return selected


def merge_recovery(source: Path, root: Path, all_rows: list) -> None:
    results = []
    recovered_ids = set(read(root / 'config.json')['case_ids'])
    for row in all_rows:
        sid = row['system_id']
        chosen = root if sid in recovered_ids else source
        result = read(chosen / 'cases' / sid / 'status.json')
        result['result_source'] = str(chosen)
        if result.get('report') and result['status'] in ('completed', 'abstained'):
            report = read(Path(result['report']))
            selected = (report.get('selection') or {}).get('selected_ids', [])
            result['status'] = outcome_status(report['status'], selected)
        results.append(result)
    summary = dict(summarize_rows(results), cases=results, phase='finished',
                   original_run=str(source), recovery_run=str(root),
                   note='Service-interrupted cases rerun from original five candidates; originals preserved. Recovery adds compute.')
    atomic_json(root / 'combined-summary.json', summary)


def validate_recovery_config(config: dict) -> None:
    from tetra.evaluation.external_science import validate_tool_snapshot
    validate_tool_snapshot(config.get('scientific_tools'))
    expected = {'provider': 'deepseek', 'model': 'deepseek-flash',
                'reasoning_effort': 'high', 'max_tokens': '8192',
                'board_protocol': 'decision-v2', 'evidence_access_mode': 'files-v1', 'board_context_mode': 'checkpoint-v1',
                'policy': 'revision-policy-v1', 'selection_policy': 'selection-policy-v3'}
    differences = [key for key, value in expected.items() if str(config.get(key)) != value]
    if differences:
        raise ValueError('Recovery cannot change frozen experiment settings (' + ', '.join(differences)
                         + '); start a new benchmark variant.')


def run(baseline: Path, root: Path, recovery_from: Path | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'config.json').exists():
            raise ValueError('Run directory already used; preserve prior attempts.')
        rows = read(baseline / 'prediction_inputs.json')
        if len(rows) != 30 or len({row['system_id'] for row in rows}) != 30:
            raise ValueError('Expected the frozen 30-case cohort.')
        all_rows = rows
        if recovery_from:
            rows = recovery_rows(recovery_from, rows)
            if not rows:
                raise ValueError('No service-error cases to rerun.')
            original_config = read(recovery_from / 'config.json')
            validate_recovery_config(original_config)
            for row in rows:
                path = baseline / 'cases' / (row['system_id'] + '.prediction-report')
                if digest(path.read_bytes()) != original_config['baseline_report_hashes'][row['system_id']]:
                    raise ValueError('Original baseline report changed before recovery.')
        from tetra.evaluation.external_science import tool_snapshot
        scientific_tools = tool_snapshot()
        concurrency = 2 if recovery_from else 4
        started = time.time()
        source_hashes = {str(path.relative_to(PROJECT)): digest(path.read_bytes())
                         for path in sorted((PROJECT / 'tetra').rglob('*.py'))}
        env = dict(os.environ, TETRA_LLM_PROVIDER='deepseek', TETRA_LLM_MODEL='deepseek-flash',
                   TETRA_SCIENCE_TOOLS_SNAPSHOT=json.dumps(scientific_tools),
                   TETRA_REASONING_EFFORT='high', TETRA_LLM_MAX_TOKENS='8192',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', TETRA_COST_LEDGER=str(root / 'cost-ledger.json'),
                   TETRA_CASE_BUDGET_USD='2', TETRA_BATCH_BUDGET_USD='10')
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key, None)
        atomic_json(root / 'config.json', {
            'baseline_report_hashes': {row['system_id']: digest((baseline / 'cases' / (row['system_id'] + '.prediction-report')).read_bytes()) for row in rows},
            'board_protocol': 'decision-v2', 'evidence_access_mode': 'files-v1', 'board_context_mode': 'checkpoint-v1',
            'scientific_tools': scientific_tools, 'baseline': str(baseline), 'case_ids': [row['system_id'] for row in rows],
            'source_hashes': source_hashes, 'max_steps': MAX_STEPS, 'case_seconds': CASE_SECONDS,
            'case_concurrency': concurrency, 'recovery_from': str(recovery_from) if recovery_from else None, 'gpus': [0, 1], 'one_prediction_per_gpu': True,
            'provider': env['TETRA_LLM_PROVIDER'], 'model': env['TETRA_LLM_MODEL'], 'reasoning_effort': env['TETRA_REASONING_EFFORT'],
            'max_tokens': env['TETRA_LLM_MAX_TOKENS'], 'started_epoch': started,
            'cost_control': {'case_limit_usd': 2, 'batch_limit_usd': 10, 'max_request_bytes': 350000, 'context_projection': 'lossless-v1'},
            'policy': 'revision-policy-v1', 'selection_policy': 'selection-policy-v3', 'cross_case_memory': False,
            'selection': 'complete comparable GNINA pool ranking; otherwise original Boltz confidence; no fallback on abstention',
            'scope': 'exploratory previously seen 30 cases; additional-compute agent comparison',
        })
        atomic_json(root / 'controller-process.json', {'pid': os.getpid(), 'started_epoch': started})
        active = {}
        pending = list(enumerate(rows))
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (root / 'gpu-memory.csv').open('w') as memory_log:
                memory_log.write('epoch,index,memory_used_MiB,utilization_percent\n')
                while active or pending:
                    for sid, (process, log, launched) in list(active.items()):
                        timed_out = time.time() - launched > CASE_SECONDS
                        if timed_out:
                            terminate_tree(process)
                        if process.poll() is not None:
                            log.close()
                            cleanup_jobs(root / 'cases' / sid)
                            path = root / 'cases' / sid / 'status.json'
                            result = read(path) if path.exists() else {'system_id': sid}
                            if timed_out or result.get('status') not in ('completed', 'abstained', 'failed', 'service_error', 'budget_exhausted', 'prediction_failed', 'artifact_error', 'revision_unverified', 'cost_budget_exhausted'):
                                result.update(status='failed', error='Case timeout' if timed_out else 'Worker exited without terminal result',
                                              elapsed_seconds=time.time() - launched)
                                atomic_json(path, result)
                            del active[sid]
                    while pending and len(active) < concurrency:
                        index, row = pending.pop(0)
                        sid = row['system_id']
                        folder = root / 'cases' / sid
                        folder.mkdir(parents=True, exist_ok=True)
                        log = (folder / 'worker.log').open('ab')
                        process = subprocess.Popen([sys.executable, '-m', 'tetra.research_benchmark', 'worker',
                                                    '--baseline', str(baseline), '--root', str(root), '--system-id', sid],
                                                   cwd=PROJECT, env=dict(env, TETRA_GPU_INDEX=str(index % 2)),
                                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                        active[sid] = (process, log, time.time())
                    snapshot(root, rows, 'running_agent', started)
                    sample_memory(memory_log)
                    if active:
                        time.sleep(5)
            atomic_json(root / 'agent-phase-finished.json', {'epoch': time.time(), 'seconds': time.time() - started})
            snapshot(root, rows, 'offline_scoring', started)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda row: offline_case(baseline, root, row), rows))
            result = snapshot(root, rows, 'finished', started)
            atomic_json(root / 'summary.json', result)
            if recovery_from:
                merge_recovery(recovery_from, root, all_rows)
            print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2), flush=True)
        finally:
            for sid, (process, log, _) in active.items():
                terminate_tree(process)
                cleanup_jobs(root / 'cases' / sid)
                log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'worker', 'status'])
    parser.add_argument('--baseline', type=Path, help='Baseline experiment directory.')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--system-id')
    parser.add_argument('--recovery-from', type=Path)
    args = parser.parse_args()
    if args.action in ('run', 'worker') and args.baseline is None:
        parser.error('--baseline is required for run and worker')
    baseline = args.baseline.resolve() if args.baseline else None
    root = args.root.resolve()
    if args.action == 'run':
        run(baseline, root, args.recovery_from.resolve() if args.recovery_from else None)
    elif args.action == 'worker':
        worker(baseline, root, args.system_id)
    else:
        result = read(root / 'status.json')
        print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
        for row in result['cases']:
            if row['status'] == 'running':
                sessions = list((root / 'cases' / row['system_id'] / 'state/research').glob('*/session.json'))
                state = read(sessions[0]) if len(sessions) == 1 else {}
                print(row['system_id'], state.get('status', 'preparing'), 'calls=', state.get('llm_calls', 0),
                      'predictions=', state.get('prediction_count', 0), 'repairs=', state.get('refinement_count', 0))


if __name__ == '__main__':
    main()
