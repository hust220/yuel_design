import pytest
import sys
import torch
import numpy as np
import os
from pathlib import Path

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from src.sidechain.app import run_sidechain_mode, save_sidechain_pdb, save_sidechain_trajectory
from src.sidechain.dataset import parse_pocket, create_dist_matrix
from src.pdb_utils import Structure
from io import StringIO

def get_random_moad_sample():
    """Get a random sample from moad_pockets table"""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT mp.id, mp.pdb
                FROM moad_pockets mp
                WHERE mp.split = 'train'
                ORDER BY RANDOM()
                LIMIT 1
                """
            )
            row = c.fetchone()
            pocket_id, pocket_pdb = row
            return pocket_id, pocket_pdb

def test_sidechain_mode(device):
    """Test sidechain mode with protein structure"""
    pocket_id, pocket_pdb = get_random_moad_sample()
    print(f"Testing sidechain mode with MOAD sample: {pocket_id}")
    
    # Parse pocket to get receptor_reduced_coords
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        raise ValueError("Failed to parse pocket structure")
    
    # Create distance matrix for reduced atoms
    receptor_reduced_coords = pocket_info['reduced_coords']
    dist_matrix = create_dist_matrix(receptor_reduced_coords, discretization_config='b12')
    
    final_prediction, chain, pocket_info = run_sidechain_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=dist_matrix,
        device=device,
    )

    output_dir = f"test_outputs/sidechain_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)

    original_pdb_path = os.path.join(output_dir, "original.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)

    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    print(f"Saving predicted coordinates to {predicted_pdb_path}")
    save_sidechain_pdb(final_prediction, pocket_info, predicted_pdb_path)

    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    print(f"Saving trajectory to {trajectory_path}")
    save_sidechain_trajectory(chain, pocket_info, trajectory_path)

    print(f"✓ Sidechain mode test passed! Generated coordinates with shape: {final_prediction.shape}")


def test_sidechain_mode_with_dataset(device):
    """Test using SidechainDataset to load data and SidechainModel to generate coordinates"""
    from src.sidechain.dataset import SidechainDataset
    from src.sidechain.model import SidechainModel
    from src.lightning1 import LightningWrapper
    from src.utils import pick_latest
    
    dataset = SidechainDataset(split='train')
    graph = dataset[0]
    graph = graph.to(device)
    
    sidechain_checkpoint = pick_latest(['checkpoints/*sidechain*/*.ckpt'])
    model = LightningWrapper.load_from_checkpoint(sidechain_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        final_prediction, chain = model.sample_chain(graph=graph)
    
    assert final_prediction.shape[0] == graph.ndata['x'].shape[0]
    assert final_prediction.shape[1] == 3
    print(f"✓ Sidechain mode with dataset test passed! Generated coordinates with shape: {final_prediction.shape}")


@pytest.fixture
def moad_sample():
    return get_random_moad_sample()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
