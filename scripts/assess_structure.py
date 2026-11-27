"""Executed in the independent PoseBusters environment. Never reads a reference pose."""
import json,sys,math,importlib.metadata
from pathlib import Path
import numpy as np
from rdkit import Chem
from posebusters import PoseBusters

request=json.loads(Path(sys.argv[1]).read_text())
# No mol_true: disable reference alignment/RMSD and identity-to-reference calculations.
buster=PoseBusters(config='dock')
df=buster.bust(mol_pred=request['ligand'],mol_cond=request['receptor'],full_report=True)
row=df.iloc[0].to_dict()
checks={k:('pass' if bool(v) else 'fail') for k,v in row.items() if isinstance(v,(bool,np.bool_))}
checks={k.lower().replace(' ','_'):v for k,v in checks.items()}
raw={}
for k,v in row.items():
 if isinstance(v,(bool,np.bool_)):raw[k]=bool(v)
 elif isinstance(v,(int,float,np.number)):raw[k]=float(v) if math.isfinite(float(v)) else None
 else:raw[k]=str(v)
Path(sys.argv[2]).write_text(json.dumps({'checks':checks,'raw_metrics':raw,'versions':{k:importlib.metadata.version(k) for k in ('posebusters','rdkit')},'reference_used':False},indent=2,allow_nan=False))
