"""SQLite is authoritative; directories hold immutable inputs and raw artifacts."""
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from tetra.runtime.inputs import digest,stable,validate,validate_config

SCHEMA='''
CREATE TABLE IF NOT EXISTS inputs(id TEXT PRIMARY KEY, record TEXT NOT NULL, confirmed INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, input_id TEXT NOT NULL, status TEXT NOT NULL, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, scope TEXT, created TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, record TEXT NOT NULL);
'''

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        for name in ['inputs','jobs','sources']:(self.root/name).mkdir(exist_ok=True)
        with self.db() as db:db.executescript(SCHEMA)
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.root/'state.sqlite',timeout=30)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db;db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()
    def event(self,scope,payload):
        with self.db() as db:db.execute('INSERT INTO events(scope,created,payload) VALUES(?,?,?)',(scope,now(),stable(payload)))
    def register_input(self,sequence,smiles,msa=None,source=None,confirmed=False):
        if not source:raise ValueError('Input provenance is required.')
        data,msa_bytes=validate(sequence,smiles,msa)
        data.update(source=source,msa_mode='file' if msa_bytes else 'server')
        msa_name='msa.csv' if msa_bytes and Path(msa).suffix.lower()=='.csv' else 'msa.a3m'
        if msa_name=='msa.csv':data['msa_format']='csv'
        key=digest(stable(data))[:24]
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT record,confirmed FROM inputs WHERE id=?',(key,)).fetchone()
            if row:
                if confirmed:db.execute('UPDATE inputs SET confirmed=1 WHERE id=?',(key,))
                result=json.loads(row[0]);result['confirmed']=bool(row[1] or confirmed);return result
            folder=self.root/'inputs'/key;folder.mkdir(exist_ok=True)
            if msa_bytes:(folder/msa_name).write_bytes(msa_bytes)
            data.update(input_id=key,created_utc=now(),msa=str(folder/msa_name) if msa_bytes else None)
            (folder/'input.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
            db.execute('INSERT INTO inputs VALUES(?,?,?)',(key,stable(data),int(confirmed)))
        return dict(data,confirmed=bool(confirmed))
    def register_complex(self,proteins,ligands,source=None,confirmed=False,benchmark=False):
        if not source or not isinstance(proteins,list) or not isinstance(ligands,list) or not proteins or not ligands:
            raise ValueError('Explicit proteins, ligands and provenance are required.')
        if len(proteins)+len(ligands)>26:raise ValueError('At most 26 component chains supported.')
        ids=[];ps=[];ls=[];files={}
        for i,protein in enumerate(proteins):
            if set(protein)-{'id','sequence','msa'} or not protein.get('msa'):
                raise ValueError('Complex proteins require id, sequence and a local MSA; no extra fields.')
            validated,raw=validate(protein['sequence'],'CCO',protein['msa'],max_residues=4000 if benchmark else 1500)
            chain=chr(65+i);suffix='.csv' if Path(protein['msa']).suffix.lower()=='.csv' else '.a3m'
            filename=chain+suffix;files[filename]=raw;ids.append(protein['id'])
            ps.append({'source_id':protein['id'],'chain_id':chain,'sequence':validated['sequence'],
                       'msa_file':filename,'msa_sha256':digest(raw)})
        for i,ligand in enumerate(ligands):
            if set(ligand)-{'id','smiles'}:raise ValueError('Ligands support only id and smiles.')
            validated,_=validate('A',ligand['smiles'],allow_unspecified_stereo=benchmark);ids.append(ligand['id'])
            ls.append({'source_id':ligand['id'],'chain_id':chr(65+len(ps)+i),'smiles':ligand['smiles'],
                       'ligand_heavy_atoms':validated['ligand_heavy_atoms']})
        if any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Component source IDs must be unique nonempty strings.')
        if sum(len(p['sequence']) for p in ps)>(4000 if benchmark else 1500) or sum(l['ligand_heavy_atoms'] for l in ls)>(1000 if benchmark else 450):
            raise ValueError('Complex exceeds configured total resource cap; do not truncate.')
        data={'schema_version':2,'proteins':ps,'ligands':ls,'source':source,'msa_mode':'file'}
        if benchmark:data['benchmark_specification']=True
        key=digest(stable(data))[:24]
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT record,confirmed FROM inputs WHERE id=?',(key,)).fetchone()
            if row:
                if confirmed:db.execute('UPDATE inputs SET confirmed=1 WHERE id=?',(key,))
                return dict(json.loads(row[0]),confirmed=bool(row[1] or confirmed))
            folder=self.root/'inputs'/key;folder.mkdir(exist_ok=True)
            for name,raw in files.items():(folder/name).write_bytes(raw)
            for protein in ps:protein['msa']=str(folder/protein['msa_file'])
            data.update(input_id=key,created_utc=now())
            (folder/'input.json').write_text(json.dumps(data,indent=2))
            db.execute('INSERT INTO inputs VALUES(?,?,?)',(key,stable(data),int(confirmed)))
        return dict(data,confirmed=bool(confirmed))
    def input(self,key):
        with self.db() as db:row=db.execute('SELECT record,confirmed FROM inputs WHERE id=?',(key,)).fetchone()
        if not row:raise ValueError('Unknown input ID.')
        data=json.loads(row[0])
        snapshot=json.loads((self.root/'inputs'/key/'input.json').read_text())
        if snapshot!=data:raise ValueError('Input snapshot was modified.')
        for item in data.get('proteins',[data]):
            if item.get('msa') and digest(Path(item['msa']).read_bytes())!=item['msa_sha256']:raise ValueError('Frozen MSA was modified.')
        return dict(data,confirmed=bool(row[1]))
    def confirm_input(self,key):
        self.input(key)
        with self.db() as db:db.execute('UPDATE inputs SET confirmed=1 WHERE id=?',(key,))
        self.event(key,{'action':'user_confirmed_identity'})
        return self.input(key)
    def submit(self,input_id,config=None):
        data=self.input(input_id)
        if not data['confirmed']:raise ValueError('User confirmation is required before prediction.')
        config=validate_config(config)
        gpu_text = os.environ.get("TETRA_GPU_INDEX", "1")
        if gpu_text not in ("0", "1"):
            raise ValueError("TETRA_GPU_INDEX must be 0 or 1.")
        gpu = int(gpu_text)
        fingerprint=digest(stable({'input_id':input_id,'config':config,'backend':'boltz-2.2.1','gpu':gpu}))
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT record,status FROM jobs WHERE fingerprint=?',(fingerprint,)).fetchone()
            if row:return dict(json.loads(row[0]),status=row[1],reused=True)
            key=uuid.uuid4().hex
            data={'job_id':key,'input_id':input_id,'config':config,'gpu':gpu,'created_utc':now(),'created_epoch':time.time(),'worker_pid':None,'worker_started':None}
            (self.root/'jobs'/key).mkdir()
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',(key,fingerprint,input_id,'queued',stable(data)))
        self.event(key,{'action':'queued','input_id':input_id,'config':config})
        return dict(data,status='queued',reused=False)
    def job(self,key):
        with self.db() as db:row=db.execute('SELECT record,status FROM jobs WHERE id=?',(key,)).fetchone()
        if not row:raise ValueError('Unknown job ID.')
        return dict(json.loads(row[0]),status=row[1])
    def update_job(self,key,status,**fields):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT record FROM jobs WHERE id=?',(key,)).fetchone()
            if not row:raise ValueError('Unknown job ID.')
            data=json.loads(row[0]);data.update(fields)
            db.execute('UPDATE jobs SET status=?,record=? WHERE id=?',(status,stable(data),key))
        return self.job(key)
    def save_entity(self,record):
        key=digest(stable(record))[:24]
        with self.db() as db:db.execute('INSERT OR IGNORE INTO entities VALUES(?,?)',(key,stable(record)))
        return dict(record,entity_id=key)
    def entity(self,key):
        with self.db() as db:row=db.execute('SELECT record FROM entities WHERE id=?',(key,)).fetchone()
        if not row:raise ValueError('Unknown entity ID.')
        return dict(json.loads(row[0]),entity_id=key)
    def save_session(self,data):
        with self.db() as db:db.execute('INSERT OR REPLACE INTO sessions VALUES(?,?)',(data['session_id'],stable(data)))
    def session(self,key):
        with self.db() as db:row=db.execute('SELECT record FROM sessions WHERE id=?',(key,)).fetchone()
        if not row:raise ValueError('Unknown session ID.')
        return json.loads(row[0])
