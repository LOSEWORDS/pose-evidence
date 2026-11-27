"""Download the pinned official GNINA asset through the GitHub asset API.

The configured proxy is used for release storage; signed URLs are not persisted in the ledger.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
SIZE = 2056131000
SHA256 = '3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172'
CHUNK = 1024 * 1024

def session():
    client = requests.Session()
    client.trust_env = False
    return client


def main():
    client = session()
    release = client.get('https://api.github.com/repos/gnina/gnina/releases/tags/v1.3.3', timeout=30)
    release.raise_for_status()
    asset = next(a for a in release.json()['assets'] if a['name'] == 'gnina.cuda12.8.static')
    assert asset['size'] == SIZE and asset['digest'] == 'sha256:' + SHA256
    response = client.get(asset['url'] + '?download=1', headers={'Accept': 'application/octet-stream'},
                          allow_redirects=False, timeout=30)
    if not response.is_redirect:
        raise RuntimeError('Official asset API did not redirect to release storage.')
    url = response.headers['Location']
    destination = ROOT / 'tools/vendor/gnina-1.3.3'
    part = destination.with_suffix('.requests.part')
    ledger = destination.with_suffix('.requests.progress.json')
    done = set(json.loads(ledger.read_text()).get('done', [])) if ledger.exists() else set()
    if done and not part.exists():
        raise ValueError('Missing partial download data.')
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    lock = threading.Lock()
    def fetch(start):
        end = min(SIZE, start + CHUNK) - 1
        for attempt in range(4):
            try:
                download_client = requests.Session()
                with download_client.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 30)) as result:
                    if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Invalid asset range response ' + str(result.status_code))
                    offset = start
                    for data in result.iter_content(65536):
                        if offset + len(data) > end + 1:
                            raise ValueError('Oversized range response')
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], offset + written)
                        offset += len(data)
                    if offset != end + 1:
                        raise ValueError('Incomplete asset range')
                with lock:
                    done.add(start)
                    os.fsync(fd)
                    temp = ledger.with_suffix('.tmp')
                    temp.write_text(json.dumps({'sha256': SHA256, 'size': SIZE, 'chunk_size': CHUNK,
                                                'done': sorted(done), 'completed_bytes': sum(min(CHUNK, SIZE-i) for i in done)}))
                    temp.replace(ledger)
                    if len(done) % 20 == 0:
                        print('GNINA completed MiB', len(done), '/', (SIZE + CHUNK - 1)//CHUNK, flush=True)
                return
            except Exception as error:
                if attempt == 3:
                    raise
                print('retry',start,type(error).__name__,flush=True)
                time.sleep(2 ** attempt)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, [i for i in range(0, SIZE, CHUNK) if i not in done]))
    finally:
        os.close(fd)
    hasher = hashlib.sha256()
    with part.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            hasher.update(data)
    if hasher.hexdigest() != SHA256:
        raise ValueError('Official binary checksum mismatch')
    part.chmod(0o755)
    part.replace(destination)
    print('GNINA verified', SHA256, flush=True)

if __name__ == '__main__':
    main()
