#!/usr/bin/env python3
"""OST 2.8 CLI adapter: serialize inconsistent residue pairs; scores untouched."""
import os
import re
import sys
from pathlib import Path

def patch_source(source,expected_count=1):
    pattern=r'\[_QualifiedResidueNotation\(r\) for r in\s+aux_data\["inconsistent_residues"\]\]'
    patched,count=re.subn(pattern,'[[_QualifiedResidueNotation(residue) for residue in pair] for pair in aux_data["inconsistent_residues"]]',source)
    if count!=expected_count:raise ValueError('Unsupported OST source; unexpected number of known serialization expressions')
    return patched

if __name__=='__main__':
    if sys.argv[1:2]!=['compare-ligand-structures']:raise ValueError('Only ligand scoring supported')
    entry_point = os.environ.get('TETRA_OST_COMPARE_SOURCE')
    if not entry_point:
        raise ValueError('Set TETRA_OST_COMPARE_SOURCE to the OpenStructure 2.8 compare-ligand-structures script.')
    source = Path(entry_point).expanduser()
    code=patch_source(source.read_text(),expected_count=2);sys.argv=[str(source),*sys.argv[2:]]
    exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
