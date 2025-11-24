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
from src.disc2.app import (
    run_disc_mode, 
    save_predictions, 
    save_ligand_sdf_from_predictions,
    save_dist_matrix_gif,
)
from src.disc2.dataset import (
    get_ligand_atoms_and_coords,
    parse_pocket,
    create_dist_matrix,
    LIGAND_ATOM_TYPES,
    LIGAND_BOND_TYPES,
)


def save_dist_matrix_comparison(
    original_dist,
    predicted_dist,
    output_path,
    title="Distance Matrix Comparison"
):
    """Save comparison of original and predicted distance matrices.
    
    Args:
        original_dist: [N, N] numpy array or tensor of original distance class indices
        predicted_dist: [N, N] numpy array or tensor of predicted distance class indices
        output_path: Path to save PNG file
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import os
    
    # Ensure parent directory exists
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(original_dist, torch.Tensor):
        original_dist = original_dist.cpu().numpy()
    if isinstance(predicted_dist, torch.Tensor):
        predicted_dist = predicted_dist.cpu().numpy()
    
    # Calculate difference
    diff = np.abs(original_dist - predicted_dist)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im1 = ax1.imshow(original_dist, cmap='viridis', aspect='equal')
    ax1.set_title('Original Distance Matrix')
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Atom Index')
    plt.colorbar(im1, ax=ax1)
    
    # Predicted
    im2 = ax2.imshow(predicted_dist, cmap='viridis', aspect='equal')
    ax2.set_title('Predicted Distance Matrix')
    ax2.set_xlabel('Atom Index')
    ax2.set_ylabel('Atom Index')
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    im3 = ax3.imshow(diff, cmap='Reds', aspect='equal')
    ax3.set_title('Absolute Difference')
    ax3.set_xlabel('Atom Index')
    ax3.set_ylabel('Atom Index')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distance matrix comparison to {output_path}")


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


def create_disc_predictions_from_mol(mol, pocket_info):
    """Create disc project predictions from molecule and pocket info.
    
    This simulates ground truth data for comparison:
    - dist_matrix: (n_ca_sc + ligand_size, n_ca_sc + ligand_size) - distance class indices
    - ligand_atoms: list of atom class indices (integers)
    """
    ligand_atoms_idx, ligand_coords = get_ligand_atoms_and_coords(mol, pocket_info)
    ligand_size = len(ligand_atoms_idx)
    
    # Use all pocket coordinates (CA + non-C atoms + ring centers)
    pocket_coords = np.array(pocket_info['coords'])
    
    dist_matrix = create_dist_matrix(pocket_coords, np.array(ligand_coords), discretization_config='b12')
    
    return dist_matrix, ligand_atoms_idx


def test_disc_mode(device):
    """Test disc mode with ground truth data"""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing disc mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break

    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info = parse_pocket(structure)
    
    # Get ligand info
    ligand_atoms_idx, ligand_coords = get_ligand_atoms_and_coords(mol, pocket_info)
    ligand_size = len(ligand_atoms_idx)
    
    # Run disc model to get predictions
    results, chain, pocket_info = run_disc_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )
    
    dist_matrix_pred = results['dist_matrix']
    ligand_atoms_pred = results['ligand_atoms']
    
    # Get ground truth for comparison
    # Use all pocket coordinates (CA + non-C atoms + ring centers)
    pocket_coords = np.array(pocket_info['coords'])
    dist_matrix_gt = create_dist_matrix(pocket_coords, np.array(ligand_coords), discretization_config='b12')
    
    # Print statistics
    print(f"✓ Disc mode test passed!")
    print(f"  Distance matrix shape: {dist_matrix_pred.shape}")
    print(f"  Ligand atoms shape: {ligand_atoms_pred.shape}")
    print(f"  Chain length: {len(chain)}")
    print(f"  Pocket atoms: {len(pocket_info['coords'])}")
    
    # Save predictions to files
    output_dir = f"test_outputs/disc_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original receptor PDB and ligand SDF
    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)
    
    original_ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(original_ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)
    
    # Save basic predictions
    save_predictions(dist_matrix_pred, ligand_atoms_pred, output_dir)
    
    # Save distance matrix GIF
    gif_path = os.path.join(output_dir, "dist_matrix_diffusion.gif")
    save_dist_matrix_gif(chain, gif_path)
    
    # Save distance matrix comparison
    comparison_path = os.path.join(output_dir, "dist_matrix_comparison.png")
    save_dist_matrix_comparison(dist_matrix_gt, dist_matrix_pred, comparison_path)


def test_disc_mode_with_dataset(device):
    """Test using DiscDataset to load data and DiscModel to generate predictions"""
    from src.disc2.dataset import DiscDataset
    from src.lightning1 import LightningWrapper
    from src.utils import pick_latest
    
    dataset = DiscDataset(split='train')
    sample = dataset[0]
    
    disc_checkpoint = pick_latest(['checkpoints/*disc*/*.ckpt'])
    print(f"Loading model from: {disc_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(disc_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    # Add batch dimension
    data = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    
    with torch.no_grad():
        final_predictions, chain = model.sample_chain(data=data)
    
    assert 'dist' in final_predictions
    assert 'atoms' in final_predictions
    # Note: model doesn't predict bonds
    print(f"✓ Disc mode with dataset test passed! Generated predictions")
    print(f"  Dist shape: {final_predictions['dist'][0].shape}")
    print(f"  Atoms shape: {final_predictions['atoms'][0].shape}")


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')