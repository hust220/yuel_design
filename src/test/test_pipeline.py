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
from src.disc2.app import run_disc_mode, save_predictions, save_dist_matrix_gif, save_dist_matrix_png
from src.sidechain.app import run_sidechain_mode, save_sidechain_pdb, save_sidechain_trajectory
from src.disc2.dataset import LIGAND_ATOM_TYPES, parse_pocket as parse_pocket_disc2
from src.sidechain.dataset import parse_pocket as parse_pocket_sidechain
from rdkit import Chem


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
    """Read molecule from molblock string."""
    mol = Chem.MolFromMolBlock(molblock)
    return Chem.RemoveHs(mol) if mol is not None else None


def convert_atom_indices_to_names(ligand_atoms_indices):
    """Convert ligand atom class indices to atom names (strings).
    
    Args:
        ligand_atoms_indices: tensor or numpy array of atom class indices
        
    Returns:
        list of atom names (strings like 'C', 'N', 'O')
    """
    if isinstance(ligand_atoms_indices, torch.Tensor):
        ligand_atoms_indices = ligand_atoms_indices.cpu().numpy()
    
    ligand_atoms = []
    for idx in ligand_atoms_indices:
        idx = int(idx)
        if 0 <= idx < len(LIGAND_ATOM_TYPES):
            ligand_atoms.append(LIGAND_ATOM_TYPES[idx])
        else:
            ligand_atoms.append('C')  # default to carbon
    
    return ligand_atoms


def test_pipeline(device):
    """Test complete pipeline: disc2 -> sidechain
    
    Randomly selects a pocket, uses disc2 to generate distance matrix,
    then feeds to sidechain to generate sidechain coordinates.
    """
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing pipeline with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break
    
    # Parse pocket to get protein size for disc2
    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info_disc2 = parse_pocket_disc2(structure)
    protein_size = len(pocket_info_disc2['coords'])
    
    # For disc2, we need ligand_size (set to 0 since we're only predicting protein sidechains)
    ligand_size = 0
    
    print(f"Protein size: {protein_size}")
    
    # Step 1: Run disc2 model to generate distance matrix
    print("\n=== Step 1: Running DISC2 model ===")
    disc_results, disc_chain, pocket_info_disc2 = run_disc_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )
    
    dist_matrix_full = disc_results['dist_matrix']
    
    print(f"DISC2 predictions:")
    print(f"  Full distance matrix shape: {dist_matrix_full.shape}")
    
    # Extract protein-only distance matrix (reduced receptor atoms)
    # disc2 returns [protein_size + ligand_size, protein_size + ligand_size]
    # We need the protein part: [protein_size, protein_size]
    dist_matrix_np = dist_matrix_full.numpy() if isinstance(dist_matrix_full, torch.Tensor) else dist_matrix_full
    receptor_dist_matrix = dist_matrix_np[:protein_size, :protein_size]
    
    print(f"  Extracted receptor distance matrix shape: {receptor_dist_matrix.shape}")
    
    # Step 2: Run sidechain model to generate sidechain coordinates
    print("\n=== Step 2: Running SIDECHAIN model ===")
    final_prediction, sidechain_chain, pocket_info = run_sidechain_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=receptor_dist_matrix,
        device=device,
    )
    
    print(f"SIDECHAIN predictions:")
    print(f"  Final coordinates shape: {final_prediction.shape}")
    print(f"  Chain length: {len(sidechain_chain)}")
    
    # Step 3: Save results
    print("\n=== Step 3: Saving results ===")
    output_dir = f"test_outputs/pipeline_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original receptor and ligand
    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)
    print(f"Saved original receptor to {original_pdb_path}")
    
    original_ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(original_ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)
    print(f"Saved original ligand to {original_ligand_sdf_path}")
    
    # Save disc2 predictions (distance matrix only, no ligand atoms)
    dist_matrix_file = os.path.join(output_dir, "disc2_dist_matrix.txt")
    np.savetxt(dist_matrix_file, dist_matrix_np, fmt='%d')
    print(f"Saved distance matrix to {dist_matrix_file}")
    
    # Save distance matrix as PNG
    dist_matrix_png = os.path.join(output_dir, "disc2_dist_matrix.png")
    save_dist_matrix_png(dist_matrix_np, dist_matrix_png, title="DISC2 Distance Matrix (Full)")
    
    # Save distance matrix diffusion chain as GIF
    dist_matrix_gif = os.path.join(output_dir, "disc2_dist_matrix_diffusion.gif")
    save_dist_matrix_gif(disc_chain, dist_matrix_gif, title="DISC2 Distance Matrix Diffusion", fps=10, dpi=100)
    
    # Save final predicted coordinates
    predicted_pdb_path = os.path.join(output_dir, "predicted_sidechain.pdb")
    print(f"Saving predicted sidechain coordinates to {predicted_pdb_path}")
    save_sidechain_pdb(final_prediction, pocket_info, predicted_pdb_path)
    
    # Save diffusion trajectory
    trajectory_path = os.path.join(output_dir, "sidechain_trajectory.pdb")
    print(f"Saving sidechain trajectory to {trajectory_path}")
    save_sidechain_trajectory(sidechain_chain, pocket_info, trajectory_path)
    
    print(f"\n✓ Pipeline test passed!")
    print(f"  Pocket ID: {pocket_id}")
    print(f"  Ligand name: {ligand_name}")
    print(f"  Protein size: {protein_size}")
    print(f"  Output directory: {output_dir}")


def test_pipeline_with_ground_truth_comparison(device):
    """Test pipeline and compare with ground truth sidechain coordinates."""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing pipeline with comparison: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break
    
    # Parse pocket to get protein size for disc2
    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info_disc2 = parse_pocket_disc2(structure)
    protein_size = len(pocket_info_disc2['coords'])
    
    # For disc2, we need ligand_size (set to 0 since we're only predicting protein sidechains)
    ligand_size = 0
    
    print(f"Protein size: {protein_size}")
    
    # Step 1: Run disc2 model
    print("\n=== Step 1: Running DISC2 model ===")
    disc_results, disc_chain, pocket_info_disc2 = run_disc_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )
    
    dist_matrix_full = disc_results['dist_matrix']
    
    # Extract protein-only distance matrix (reduced receptor atoms)
    dist_matrix_np = dist_matrix_full.numpy() if isinstance(dist_matrix_full, torch.Tensor) else dist_matrix_full
    receptor_dist_matrix = dist_matrix_np[:protein_size, :protein_size]
    
    # Step 2: Run sidechain model
    print("\n=== Step 2: Running SIDECHAIN model ===")
    final_prediction, sidechain_chain, pocket_info = run_sidechain_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=receptor_dist_matrix,
        device=device,
    )
    
    # Extract predicted sidechain coordinates (non-CA atoms)
    final_prediction_np = final_prediction.detach().cpu().numpy()
    
    # Get ground truth sidechain coordinates from original structure
    structure_gt = Structure()
    structure_gt.read(StringIO(pocket_pdb))
    pocket_info_gt = parse_pocket_sidechain(structure_gt)
    full_coords_gt = pocket_info_gt['full_coords']
    
    # Calculate RMSD for sidechain atoms (non-CA atoms)
    # Note: This is a simplified comparison - in practice, you'd need to match atoms properly
    if final_prediction_np.shape[0] == len(full_coords_gt):
        # Align coordinates (center of mass)
        pred_centered = final_prediction_np - final_prediction_np.mean(axis=0)
        gt_centered = full_coords_gt - full_coords_gt.mean(axis=0)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_centered - gt_centered) ** 2, axis=1)))
        print(f"\nSidechain RMSD (centered): {rmsd:.3f} Å")
    else:
        print(f"\nWarning: Coordinate size mismatch (predicted: {final_prediction_np.shape[0]}, ground truth: {len(full_coords_gt)})")
    
    # Save results
    output_dir = f"test_outputs/pipeline_comparison_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)
    
    original_ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(original_ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)
    
    # Save disc2 distance matrix
    dist_matrix_file = os.path.join(output_dir, "disc2_dist_matrix.txt")
    np.savetxt(dist_matrix_file, dist_matrix_np, fmt='%d')
    
    # Save distance matrix as PNG
    dist_matrix_png = os.path.join(output_dir, "disc2_dist_matrix.png")
    save_dist_matrix_png(dist_matrix_np, dist_matrix_png, title="DISC2 Distance Matrix (Full)")
    
    # Save distance matrix diffusion chain as GIF
    dist_matrix_gif = os.path.join(output_dir, "disc2_dist_matrix_diffusion.gif")
    save_dist_matrix_gif(disc_chain, dist_matrix_gif, title="DISC2 Distance Matrix Diffusion", fps=10, dpi=100)
    
    predicted_pdb_path = os.path.join(output_dir, "predicted_sidechain.pdb")
    save_sidechain_pdb(final_prediction, pocket_info, predicted_pdb_path)
    
    trajectory_path = os.path.join(output_dir, "sidechain_trajectory.pdb")
    save_sidechain_trajectory(sidechain_chain, pocket_info, trajectory_path)
    
    print(f"✓ Pipeline with comparison test passed!")
    print(f"  Output directory: {output_dir}")


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

