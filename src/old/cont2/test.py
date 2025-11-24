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
from src.cont2.app import run_cont_mode, save_structure_pdb, save_trajectory
from src.cont2.dataset import (
    get_ligand_atoms_and_coords,
    parse_pocket,
    create_interaction_index,
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


def create_int_index_from_mol(mol, pocket_info):
    """Create interaction index from molecule and pocket info.
    
    Returns:
        tuple: (int_index, ligand_reduced_atoms, ligand_size)
            - int_index: List of tuples, each tuple is (receptor_full_idx, ligand_full_idx)
            - ligand_reduced_atoms: List of ligand reduced atom types (e.g., ['_O', '_N'])
            - ligand_size: Total number of ligand atoms (including C atoms)
    """
    ligand_reduced_atoms, ligand_reduced_coords, ligand_full_atoms, ligand_full_coords = get_ligand_atoms_and_coords(mol)
    
    print(f"Ligand reduced atoms: {ligand_reduced_atoms}")
    print(f"Ligand full atoms: {ligand_full_atoms}")
    print(f"Ligand size: {len(ligand_full_coords)}")

    # Extract reduced coords from pocket
    receptor_reduced_coords = pocket_info['reduced_coords']
    receptor_atoms = pocket_info['atoms']
    
    # Create interaction index
    int_index = create_interaction_index(
        receptor_reduced_coords=receptor_reduced_coords,
        ligand_reduced_coords=ligand_reduced_coords,
        receptor_atoms=receptor_atoms,
        interaction_threshold=5.0
    )
    
    ligand_size = len(ligand_full_coords)
    
    return int_index, ligand_reduced_atoms, ligand_size


def test_cont_mode(device):
    """Test cont mode with interaction index - predicts all non-CA atoms"""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing cont mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break

    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info = parse_pocket(structure)
    
    # Create interaction index from molecule
    int_index, ligand_reduced_atoms, ligand_size = create_int_index_from_mol(mol, pocket_info)
    
    if len(int_index) == 0:
        print("Warning: int_index is empty, skipping this sample")
        return
    
    final_prediction, chain, pocket_info = run_cont_mode(
        pocket_structure=pocket_pdb,
        int_index=int_index,
        ligand_fixed_atoms=ligand_reduced_atoms,
        ligand_size=ligand_size,
        device=device,
    )

    import os
    output_dir = f"test_outputs/cont2_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)

    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)

    ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)

    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    print(f"Saving predicted coordinates to {predicted_pdb_path}")
    save_structure_pdb(final_prediction, pocket_info, predicted_pdb_path)

    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    print(f"Saving trajectory to {trajectory_path}")
    save_trajectory(chain, pocket_info, trajectory_path)

    print(f"✓ Cont mode test passed! Generated coordinates with shape: {final_prediction.shape}")




def test_cont_mode_with_dataset(device):
    """Test using ContDataset to load data and ContModel to generate coordinates"""
    from src.cont2.dataset import ContDataset
    from src.cont2.model import ContModel
    from src.lightning1 import LightningWrapper
    from src.utils import pick_latest
    
    dataset = ContDataset(split='train')
    data = dataset[0]
    data = {k: v.to(device) for k, v in data.items()}
    
    cont_checkpoint = pick_latest(['checkpoints/*cont2*/*.ckpt'])
    model = LightningWrapper.load_from_checkpoint(cont_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        final_prediction, chain = model.sample_chain(data=data)
    
    assert final_prediction.shape[0] == data['x'].shape[0]
    assert final_prediction.shape[1] == 3
    print(f"✓ Cont mode with dataset test passed! Generated coordinates with shape: {final_prediction.shape}")


@pytest.fixture
def moad_sample():
    return get_random_moad_sample()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
