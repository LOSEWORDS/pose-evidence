"""Durable ranged downloads with remote checksum validation."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def atomic_json(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(data,indent=2));os.replace(temporary,path)

def checksum_file(path,checksum):
    algorithm,expected=checksum.split(':',1);hasher=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(block)
    if hasher.hexdigest()!=expected:raise ValueError(f'Checksum mismatch for {path}')


def download_file(url,path,size,checksum,chunk_size=64*1024*1024,workers=4):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if path.exists():
            if path.stat().st_size!=size:raise ValueError('Existing archive has wrong size.')
            checksum_file(path,checksum);return
        part=path.with_name(path.name+'.part');ledger=path.with_name(path.name+'.progress.json')
        identity={'url':url,'size':size,'checksum':checksum,'chunk_size':chunk_size}
        progress=json.loads(ledger.read_text()) if ledger.exists() else dict(identity,done=[],completed_bytes=0)
        if any(progress[k]!=v for k,v in identity.items()):raise ValueError('Download metadata changed; use a new destination.')
        if progress['done'] and not part.exists():raise ValueError('Partial data missing; cannot trust completion ledger.')
        fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);lock=threading.Lock()
        try:
            os.ftruncate(fd,size)
            def fetch(lo):
                hi=min(size-1,lo+chunk_size-1)
                for attempt in range(5):
                    try:
                        request=urllib.request.Request(url+('&' if '?' in url else '?')+f'part={lo}',headers={'Range':f'bytes={lo}-{hi}'})
                        with urllib.request.urlopen(request,timeout=120) as response:
                            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':raise ValueError('Server did not honor exact byte range.')
                            offset=lo
                            while offset<=hi:
                                block=response.read(min(4*1024*1024,hi-offset+1))
                                if not block:raise IOError('Premature download EOF.')
                                written=0
                                while written<len(block):written+=os.pwrite(fd,block[written:],offset+written)
                                offset+=len(block)
                        with lock:
                            os.fsync(fd);progress['done'].append(lo);progress['completed_bytes']+=hi-lo+1
                            progress['updated_epoch']=time.time();atomic_json(ledger,progress)
                            print(f"DOWNLOAD {path.name} {progress['completed_bytes']}/{size} ({100*progress['completed_bytes']/size:.1f}%)",flush=True)
                        return
                    except Exception:
                        if attempt==4:raise
                        time.sleep(2**attempt)
            done=set(progress['done'])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fetch,[lo for lo in range(0,size,chunk_size) if lo not in done]))
        finally:os.close(fd)
        checksum_file(part,checksum);os.replace(part,path);progress['verified']=True;atomic_json(ledger,progress)
        print('VERIFIED',path.name,checksum,flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--record',required=True);parser.add_argument('--dest',required=True);parser.add_argument('--files',nargs='+',required=True)
    args=parser.parse_args();record=json.loads(Path(args.record).read_text());files={f['key']:f for f in record['files']}
    for name in args.files:
        item=files[name];download_file(item['links']['self'],Path(args.dest)/name,item['size'],item['checksum'])


if __name__ == "__main__":
    main()
