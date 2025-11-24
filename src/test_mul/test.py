import pytest
import sys
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from src.e2efinal.app import run_e2e_mode, save_structure_pdb, save_trajectory
from src.e2efinal.dataset import (
    get_ligand_atoms_and_coords,
    parse_pocket,
    create_ligand_coords_features,
)

def get_random_moad_sample():
    """Get a random sample from moad_pockets table"""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT mp.id, mp.pdb, ml.mol, ml.name
                FROM moad_pockets mp
                JOIN moad_ligands ml ON mp.ligand_name = ml.name
                WHERE mp.split = 'train'
                ORDER BY RANDOM()
                LIMIT 1
                """
            )
            row = c.fetchone()
            pocket_id, pocket_pdb, ligand_mol, ligand_name = row
            return pocket_id, pocket_pdb, ligand_mol, ligand_name


def read_molecule_from_molblock(molblock: str):
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(molblock)
    return Chem.RemoveHs(mol) if mol is not None else None


def get_ligand_size_from_mol(mol):
    """Get ligand size from molecule.
    
    Returns:
        int: Total number of ligand atoms (including C atoms, excluding H)
    """
    ligand_reduced_atoms, ligand_reduced_coords, ligand_full_atoms, ligand_full_coords = get_ligand_atoms_and_coords(mol)
    return len(ligand_full_coords)


def test_e2e_mode(device):
    """Test e2e mode with interaction index - predicts both ligand coordinates and atom types"""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing e2e mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break
    
    # Get ligand size from molecule
    ligand_size = get_ligand_size_from_mol(mol)
    print(f"Ligand size: {ligand_size}")
    
    final_coords, final_atoms, chain, pocket_info = run_e2e_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )

    import os
    output_dir = f"test_outputs/e2efinal_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)

    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)

    ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)

    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    print(f"Saving predicted coordinates and atom types to {predicted_pdb_path}")
    save_structure_pdb(final_coords, final_atoms, pocket_info, predicted_pdb_path)

    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    print(f"Saving trajectory to {trajectory_path}")
    save_trajectory(chain, final_atoms, pocket_info, trajectory_path)

    print(f"✓ E2E mode test passed! Generated coordinates with shape: {final_coords.shape}")
    print(f"  Generated atom types with shape: {final_atoms.shape}")
    print(f"  Ligand atom types: {final_atoms[final_atoms > 0].cpu().numpy()}")


def test_e2e_mode_with_dataset(device):
    """Test using E2EDataset to load data and E2EModel to generate coordinates and atom types"""
    from src.e2efinal.dataset import E2EDataset
    from src.lightning1 import LightningWrapper
    from src.utils import pick_latest
    
    dataset = E2EDataset(split='train')
    data = dataset[0]
    # Add batch dimension
    data = {k: v.unsqueeze(0).to(device) for k, v in data.items()}
    
    e2e_checkpoint = pick_latest(['checkpoints/*e2efinal*/*.ckpt'])
    print(f"Loading model from: {e2e_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(e2e_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        final_result, chain = model.sample_chain(data=data)
    
    # Extract coords and atoms from result dict
    final_coords = final_result['coords']  # [B, N, 3]
    final_atoms = final_result['atoms']  # [B, N]
    
    # Remove batch dimension
    if final_coords.dim() == 3:
        final_coords = final_coords.squeeze(0)  # [N, 3]
        final_atoms = final_atoms.squeeze(0)  # [N]
    
    # Mask out receptor atoms
    receptor_mask = data['receptor_mask'].squeeze(0)  # [N]
    final_atoms = final_atoms * (1 - receptor_mask.long())
    
    assert final_coords.shape[0] == data['x'].squeeze(0).shape[0]
    assert final_coords.shape[1] == 3
    assert final_atoms.shape[0] == data['x'].squeeze(0).shape[0]
    print(f"✓ E2E mode with dataset test passed!")
    print(f"  Generated coordinates with shape: {final_coords.shape}")
    print(f"  Generated atom types with shape: {final_atoms.shape}")
    ligand_mask = 1 - receptor_mask
    if ligand_mask.sum() > 0:
        ligand_atoms = final_atoms[ligand_mask.bool()]
        print(f"  Ligand atom types (non-zero): {ligand_atoms[ligand_atoms > 0].cpu().numpy()}")


@pytest.fixture
def moad_sample():
    return get_random_moad_sample()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
