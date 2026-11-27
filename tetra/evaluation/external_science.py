"""Reference-free external scientific tools with immutable inputs and bounded execution."""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from tetra.evaluation.assessment import identity_check, verify_candidate
from tetra.runtime.download import atomic_json
from tetra.runtime.inputs import digest, stable

PROJECT = Path(__file__).resolve().parents[2]
BINARIES = {'gnina': PROJECT / 'tools/vendor/gnina-1.3.3',
            'fpocket': PROJECT / 'tools/fpocket-4.2.3/bin/fpocket'}


@lru_cache(maxsize=128)
def _binary_hash(path, size, modified):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def tool_identity(name):
    binary = BINARIES[name]
    stat = binary.stat()
    result = {'tool': name, 'binary_sha256': _binary_hash(str(binary), stat.st_size, stat.st_mtime_ns)}
    if name == 'gnina':
        libraries = {}
        for library in sorted((PROJECT / 'tools/gnina-runtime/lib').glob('*.so*')):
            info = library.stat()
            libraries[library.name] = _binary_hash(str(library.resolve()), info.st_size, info.st_mtime_ns)
        result['runtime_sha256'] = digest(stable(libraries))
    return result


def available(name):
    """Advertise only binaries that passed a recorded real local validation."""
    try:
        record = json.loads((PROJECT / 'environment/scientific-tools-ready.json').read_text())[name]
        return record['validated'] is True and all(record.get(k) == v for k, v in tool_identity(name).items())
    except (OSError, KeyError, ValueError):
        return False


def _run(name, args, folder, timeout):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    identity = tool_identity(name)
    command = [str(BINARIES[name]), *map(str, args)]
    start = time.monotonic()
    environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
    if name == 'gnina':
        environment['LD_LIBRARY_PATH'] = str(PROJECT / 'tools/gnina-runtime/lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
    try:
        with (folder / 'tool.log').open('w') as stream:
            result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout, env=environment)
    except (subprocess.TimeoutExpired, OSError) as error:
        atomic_json(folder / 'execution.json', {**identity, 'command': command,
                    'elapsed_seconds': time.monotonic() - start, 'error_type': type(error).__name__,
                    'status': 'failed', 'log': str(folder / 'tool.log')})
        raise
    provenance = {**identity, 'command': command, 'elapsed_seconds': time.monotonic() - start,
                  'returncode': result.returncode, 'log': str(folder / 'tool.log')}
    atomic_json(folder / 'execution.json', provenance)
    if result.returncode:
        raise ValueError(f'{name} failed with exit {result.returncode}; see {folder / "tool.log"}')
    return (folder / 'tool.log').read_text(), provenance


def parse_gnina(text):
    fields = {}
    for label, field in [('CNNscore', 'cnn_score'), ('CNNaffinity', 'cnn_affinity'), ('Affinity', 'affinity')]:
        matches = re.findall(r'^\s*' + label + r':\s*(\S+)', text, re.MULTILINE)
        if not matches and field != 'cnn_score':
            continue
        if len(matches) != 1:
            raise ValueError('Expected one finite GNINA ' + label)
        value = float(matches[0])
        if not math.isfinite(value) or (field == 'cnn_score' and not 0 <= value <= 1):
            raise ValueError('Invalid GNINA ' + label)
        fields[field] = value
    return fields


def score_candidate(candidate, folder, timeout=180):
    verify_candidate(candidate)
    if not identity_check(candidate):
        raise ValueError('GNINA requires verified molecular identity.')
    args = ['--receptor', candidate['receptor'], '--ligand', candidate['ligand'],
            '--score_only', '--cnn_scoring', 'rescore', '--cpu', '2', '--no_gpu', '--seed', '0']
    text, provenance = _run('gnina', args, folder, timeout)
    verify_candidate(candidate)
    scores = parse_gnina(text)
    profile = digest(stable({'binary_sha256': provenance['binary_sha256'],
                             'runtime_sha256': provenance.get('runtime_sha256'),
                             'mode': 'score_only', 'cnn_scoring': 'rescore', 'seed': 0, 'device': 'cpu'}))
    result = {'candidate_id': candidate['candidate_id'], 'candidate_sha256': candidate['sha256'],
              **scores, 'profile': profile, 'provenance': provenance,
              'limitations': ['CNN score ranks predicted poses; it is not measured accuracy or binding evidence.',
                             'No minimization, docking or coordinate edits were requested.']}
    atomic_json(Path(folder) / 'score.json', result)
    return result


def parse_fpocket_info(text):
    pockets = {}
    current = None
    for line in text.splitlines():
        match = re.match(r'\s*Pocket\s+(\d+)\s*:', line)
        if match:
            current = int(match[1])
            if current in pockets:
                raise ValueError('Duplicate pocket identifier.')
            pockets[current] = {}
        elif current is not None and ':' in line:
            name, value = line.split(':', 1)
            try:
                number = float(value.strip())
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError('Nonfinite pocket descriptor.')
            pockets[current][name.strip()] = number
    return pockets


def detect_pockets(candidate, folder):
    """fpocket sees only receptor atoms, never ligand coordinates or target references."""
    import gemmi
    from tetra.evaluation.structural_metrics import sequence_residues
    verify_candidate(candidate)
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    structure = gemmi.read_structure(candidate['receptor'])
    mapping = sequence_residues(structure, candidate['sequence'])
    # Explicit input-position numbering avoids insertion-code ambiguity in external output.
    structure[0][0].name = 'A'
    for index, residue in enumerate(structure[0][0], 1):
        residue.seqid = gemmi.SeqId(index, ' ')
    receptor = folder / 'receptor.pdb'
    structure.write_pdb(str(receptor))
    _, provenance = _run('fpocket', ['-f', receptor], folder, 60)
    info = folder / 'receptor_out/receptor_info.txt'
    if not info.exists():
        raise ValueError('fpocket did not produce a pocket descriptor artifact.')
    descriptors = parse_fpocket_info(info.read_text())
    pockets = []
    for number, values in descriptors.items():
        path = folder / f'receptor_out/pockets/pocket{number}_atm.pdb'
        pocket = gemmi.read_structure(str(path))
        positions = set()
        for model in pocket:
            for chain in model:
                for residue in chain:
                    if residue.name not in {row['residue_name'] for row in mapping}:
                        continue
                    index = residue.seqid.num
                    if chain.name != 'A' or not 1 <= index <= len(mapping) or residue.name != mapping[index-1]['residue_name']:
                        raise ValueError('fpocket residue identity/numbering changed.')
                    positions.add(index)
        positions = sorted(positions)
        if not positions or any(not 1 <= position <= len(mapping) for position in positions):
            raise ValueError('Pocket residues cannot be mapped to the exact input sequence.')
        pockets.append({'pocket_id': number, 'input_positions': positions, 'descriptors': values,
                        'artifact_path': str(path), 'artifact_sha256': digest(path.read_bytes())})
    verify_candidate(candidate)
    result = {'candidate_id': candidate['candidate_id'], 'candidate_sha256': candidate['sha256'],
              'input_kind': 'receptor_only', 'pockets': pockets, 'residue_mapping': mapping,
              'provenance': provenance, 'receptor_input_sha256': digest(receptor.read_bytes()),
              'limitations': ['Geometric cavities are hypotheses, not independently validated binding sites.',
                             'Input receptor is predicted and can retain ligand-conditioned conformation.']}
    atomic_json(folder / 'pockets.json', result)
    return result


def rank_scored_candidates(ids, candidates, scores):
    """No partial-pool ranking or comparisons across different scoring profiles."""
    profiles = set()
    for key in ids:
        score = scores.get(key, {})
        value = score.get('cnn_score')
        if (score.get('candidate_sha256') != candidates[key]['sha256'] or not score.get('profile')
                or type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1):
            return None
        profiles.add(score['profile'])
    if len(profiles) != 1:
        return None
    return sorted(ids, key=lambda key: scores[key]['cnn_score'], reverse=True)


def tool_snapshot():
    snapshot = {}
    for name in BINARIES:
        ready = available(name)
        snapshot[name] = dict(tool_identity(name), available=True) if ready else {'available': False}
    return snapshot


def validate_tool_snapshot(expected):
    if expected != tool_snapshot():
        raise ValueError('Scientific tool availability or binary changed; start a separate experiment.')
