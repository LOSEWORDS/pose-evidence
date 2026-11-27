"""Reference-free, receptor-frame fragment diagnostics and controlled edits."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, Lipinski
from tetra.evaluation.structural_metrics import _load, align_points

CONFIG = {'offsets_deg': [-60, 60, -120, 120], 'core_max_A': 0.3,
          'minimum_retained_angle_deg': 20.0, 'stable_core_max_A': 0.5,
          'minimum_dispersion': 0.01, 'timeout_seconds': 600,
          'max_iterations': 250, 'restraint_k_kj_mol_nm2': 1000.0,
          'protein_forcefield': 'amber14-all.xml', 'ligand_forcefield': 'openff-2.2.1',
          'platform': 'CPU', 'threads': 2}


def angle_difference(a, b):
    return float((a - b + 180) % 360 - 180)


def circular_variance(angles):
    if not angles or not np.isfinite(angles).all():
        raise ValueError('Finite angles required.')
    return float(1 - abs(np.mean(np.exp(1j * np.radians(angles)))))


def torsions(mol):
    """Conservative acyclic single bonds; indices refer to this heavy-atom graph."""
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('One connected heavy-atom graph required.')
    eligible = {frozenset(pair) for pair in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)}
    rows = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((j, k)) not in eligible or bond.IsInRing() or bond.GetIsConjugated() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        seen, todo = {j}, [j]
        while todo:
            for atom in mol.GetAtomWithIdx(todo.pop()).GetNeighbors():
                i = atom.GetIdx()
                if i not in seen and i != k:
                    seen.add(i)
                    todo.append(i)
        other = set(range(mol.GetNumAtoms())) - seen
        if len(seen) < len(other):
            j, k, seen, other = k, j, other, seen
        if len(seen) < 3 or len(other) < 2:
            continue
        left = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k)
        right = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j)
        rows.append({'torsion_id': f'{j}-{k}', 'bond': [j, k],
                     'atoms': [left[0], j, k, right[0]], 'core': sorted(seen),
                     'fragment': sorted(other)})
    return rows


def edit_torsion(mol, row, offset):
    if row not in torsions(mol) or offset not in CONFIG['offsets_deg']:
        raise ValueError('Unsupported bond, orientation, or offset.')
    edited = Chem.Mol(mol)
    before = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *row['atoms'])
    rdMolTransforms.SetDihedralDeg(edited.GetConformer(), *row['atoms'], before + offset)
    measured = rdMolTransforms.GetDihedralDeg(edited.GetConformer(), *row['atoms'])
    if abs(angle_difference(measured, before + offset)) > 1e-6:
        raise ValueError('Requested torsion edit did not occur.')
    if np.max(np.linalg.norm(edited.GetConformer().GetPositions()[row['core']] - mol.GetConformer().GetPositions()[row['core']], axis=1)) > 1e-6:
        raise ValueError('Edit moved the protected core.')
    return edited


def manipulation_check(parent, edited, relaxed, row):
    before = rdMolTransforms.GetDihedralDeg(parent.GetConformer(), *row['atoms'])
    edit = rdMolTransforms.GetDihedralDeg(edited.GetConformer(), *row['atoms'])
    after = rdMolTransforms.GetDihedralDeg(relaxed.GetConformer(), *row['atoms'])
    displacement = np.linalg.norm(relaxed.GetConformer().GetPositions() - parent.GetConformer().GetPositions(), axis=1)
    core = float(max(displacement[row['core']]))
    edit_success = abs(angle_difference(edit, before)) >= CONFIG['minimum_retained_angle_deg']
    retained = abs(angle_difference(after, before)) >= CONFIG['minimum_retained_angle_deg']
    return {'torsion_before_deg': before, 'torsion_after_edit_deg': edit,
            'torsion_after_relaxation_deg': after, 'edit_success': edit_success,
            'retained_new_state': retained, 'core_max_displacement_A': core,
            'fragment_rms_displacement_A': float(np.sqrt(np.mean(displacement[row['fragment']] ** 2))),
            'status': 'PASS' if edit_success and retained and core <= CONFIG['core_max_A'] else 'FAILED',
            'accuracy_improvement': 'NOT_EVALUATED',
            'interpretation_limit': 'A retained new state is not evidence of real flexibility or accuracy.'}


def variability(candidates, parent_id):
    """Whole-chain CA frame, complete chiral graph maps; no free ligand fit."""
    if not 2 <= len(candidates) <= 16 or len({c['sha256'] for c in candidates}) != len(candidates):
        raise ValueError('Two to sixteen distinct candidate structures required.')
    parent = next(c for c in candidates if c['candidate_id'] == parent_id)
    structure, _, reference, xyz = _load(parent)
    def ca(st):
        rows = [[a for a in r if a.name == 'CA'] for r in st[0][0]]
        if any(len(r) != 1 for r in rows):
            raise ValueError('Incomplete CA mapping.')
        return np.array([[r[0].pos.x, r[0].pos.y, r[0].pos.z] for r in rows])
    ref_ca = ca(structure)
    mapped, maps, pocket = [], [], []
    protein_atoms = [(i, a) for i, r in enumerate(structure[0][0]) for a in r if not a.element.is_hydrogen]
    pocket_indices = sorted({i for i, a in protein_atoms if np.linalg.norm(xyz - [a.pos.x, a.pos.y, a.pos.z], axis=1).min() <= 6})
    for c in candidates:
        if c['sequence'] != parent['sequence'] or Chem.MolToSmiles(Chem.MolFromSmiles(c['smiles'])) != Chem.MolToSmiles(Chem.MolFromSmiles(parent['smiles'])):
            raise ValueError('Different molecular identities.')
        st, _, mol, coords = _load(c)
        aligned, rotation, translation = align_points(ca(st), ref_ca)
        coords = coords @ rotation + translation
        matches = mol.GetSubstructMatches(reference, uniquify=False, useChirality=True, maxMatches=4097)
        if not matches or len(matches) > 4096:
            raise ValueError('Unresolved or excessive graph symmetry.')
        match = min(matches, key=lambda m: float(np.sum((coords[list(m)] - xyz) ** 2)))
        clone = Chem.Mol(reference)
        for i, point in enumerate(coords[list(match)]):
            clone.GetConformer().SetAtomPosition(i, point)
        mapped.append(clone)
        maps.append({'candidate_id': c['candidate_id'], 'reference_to_candidate': list(match), 'mapping_count': len(matches)})
        pocket.append({'candidate_id': c['candidate_id'], 'pocket_CA_rms_A': float(np.sqrt(np.mean(np.sum((aligned[pocket_indices] - ref_ca[pocket_indices]) ** 2, axis=1)))) if pocket_indices else None})
    rows = []
    for row in torsions(reference):
        angles = [rdMolTransforms.GetDihedralDeg(m.GetConformer(), *row['atoms']) for m in mapped]
        displacements = [np.linalg.norm(m.GetConformer().GetPositions() - xyz, axis=1) for m in mapped]
        core = max(float(max(d[row['core']])) for d in displacements)
        dispersion = circular_variance(angles)
        rows.append({**row, 'angles_deg': angles, 'circular_variance': dispersion,
                     'core_max_displacement_A': core,
                     'fragment_max_rms_A': max(float(np.sqrt(np.mean(d[row['fragment']] ** 2))) for d in displacements),
                     'eligible': core <= CONFIG['stable_core_max_A'] and dispersion >= CONFIG['minimum_dispersion']})
    return {'parent_id': parent_id, 'candidate_hashes': {c['candidate_id']: c['sha256'] for c in candidates},
            'alignment': 'whole_chain_CA; no ligand fitting; graph symmetry selected globally in receptor frame',
            'atom_indexing': 'zero-based parent heavy-atom SDF order', 'torsions': rows,
            'mappings': maps, 'pocket_backbone': pocket,
            'threshold_origin': 'predeclared engineering limits, not calibrated error probabilities',
            'limitations': ['Prediction dispersion does not establish error or physical flexibility.',
                           'Equivalent-atom mappings can affect torsions; mappings are explicit and shared across fragments.',
                           'Core is a protected graph component, not a proven biological anchor.',
                           'No receptor side-chain chi or chemical interaction classification in this version.']}
