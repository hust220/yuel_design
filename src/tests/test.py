import pytest
from rdkit import Chem
import sys
from pathlib import Path
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
import os
from rdkit.Chem import rdDetermineBonds

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent  # Goes up two levels from test.py
print(project_root)
sys.path.append(str(project_root))
# sys.path.append('../../')

import src
from src.molecule_builder import build_molecule, build_molecules
from src.datasets1 import parse_molecule, get_pocket, MOADDataset, get_dataloader, collate
from src.metrics import is_valid, qed, sas
from src import const

# write a function to convert smiles to one_hot and positions based on the function parse_molecule in datasets1
def rebuild_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    # build the 3d coordinates of the molecule
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    # Chem.UFFOptimizeMolecule(mol)
    pos, one_hot, _ = parse_molecule(mol)
    pos = torch.tensor(pos)
    one_hot = torch.tensor(one_hot)
    one_hot = one_hot[:,const.GEOM_NUMBER_OF_RESIDUE_TYPES:]
    atom_types = one_hot.argmax(dim=1)
    molecule = build_molecule(pos, atom_types)
    return molecule

def test_build_molecules():
    print(os.path.join(project_root, 'datasets'))
    ds = MOADDataset(
        data_path=os.path.join(project_root, 'datasets'),
        prefix='MOAD_val',
        device='cpu'
    )
    dataloader = get_dataloader(
        ds, batch_size=1, collate_fn=collate
    )
    for data in dataloader:
        molecules = build_molecules(
            data['one_hot'][:, :, const.GEOM_NUMBER_OF_RESIDUE_TYPES:],
            data['positions'],
            data['ligand_mask'],
        )
        for i, (molecule,name) in enumerate(zip(molecules, data['name'])):
            # if not is_valid(molecule):
                # draw molecule
                # molecule = Chem.RemoveHs(molecule)
            img = Draw.MolToImage(molecule)
            img.save(f"tests/molecule_1_{name}.png")
            assert is_valid(molecule), "The molecule should be valid."

            # rdDetermineBonds.DetermineBonds(molecule, allowChargedFragments=True, charge=-1, embedChiral=False, useVdw=False)
            Chem.SanitizeMol(molecule)
            img = Draw.MolToImage(molecule)
            img.save(f"tests/molecule_2_{name}.png")
            assert is_valid(molecule), "The molecule should be valid."

            q = qed(molecule)
            s = sas(molecule)
            print(q, s)
            assert q, "QED"
            assert s, "SAS"


    assert is_valid(rebuild_molecule("C1CCCCC1C2CCCCC2")), "The molecule should be valid."
    assert is_valid(rebuild_molecule("C1=CC=CC=C1")), "The molecule should be valid."


