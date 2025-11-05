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
from src.disc.app import run_disc_mode, save_predictions
from src.cont.app import run_cont_mode, save_coords_pdb, save_coords_trajectory
from src.disc.dataset import LIGAND_ATOM_TYPES
from src.cont.dataset import get_ligand_atoms_and_coords, parse_pocket
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
    """Test complete pipeline: disc -> cont
    
    Randomly selects a pocket, uses disc to generate distance matrix and ligand atoms,
    then feeds to cont to generate final PDB file.
    """
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing pipeline with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break
    
    # Get ligand size from molecule
    ligand_atoms_cont, ligand_coords, _ = get_ligand_atoms_and_coords(mol)
    ligand_size = len(ligand_atoms_cont)
    
    print(f"Ligand size: {ligand_size}")
    
    # Step 1: Run disc model to generate distance matrix and ligand atoms/bonds
    print("\n=== Step 1: Running DISC model ===")
    disc_results, disc_chain, pocket_info = run_disc_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )
    
    dist_matrix = disc_results['dist_matrix']
    ligand_atoms_indices = disc_results['ligand_atoms']
    ligand_bonds = disc_results['ligand_bonds']
    
    print(f"DISC predictions:")
    print(f"  Distance matrix shape: {dist_matrix.shape}")
    print(f"  Ligand atoms shape: {ligand_atoms_indices.shape}")
    print(f"  Ligand bonds shape: {ligand_bonds.shape}")
    
    # Convert atom indices to names
    ligand_atoms = convert_atom_indices_to_names(ligand_atoms_indices)
    print(f"  Ligand atoms: {ligand_atoms[:10]}..." if len(ligand_atoms) > 10 else f"  Ligand atoms: {ligand_atoms}")
    
    # Step 2: Run cont model to generate coordinates
    print("\n=== Step 2: Running CONT model ===")
    final_prediction, cont_chain, pocket_info = run_cont_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=dist_matrix.numpy() if isinstance(dist_matrix, torch.Tensor) else dist_matrix,
        ligand_atoms=ligand_atoms,
        ligand_bonds=ligand_bonds.numpy() if isinstance(ligand_bonds, torch.Tensor) else ligand_bonds,
        device=device,
    )
    
    print(f"CONT predictions:")
    print(f"  Final coordinates shape: {final_prediction.shape}")
    print(f"  Chain length: {len(cont_chain)}")
    
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
    
    # Save disc predictions
    save_predictions(
        dist_matrix,
        ligand_atoms_indices,
        ligand_bonds,
        output_dir,
        prefix="disc"
    )
    
    # Save final predicted coordinates
    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    print(f"Saving predicted coordinates to {predicted_pdb_path}")
    save_coords_pdb(final_prediction, pocket_info, predicted_pdb_path)
    
    # Save diffusion trajectory
    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    print(f"Saving trajectory to {trajectory_path}")
    save_coords_trajectory(cont_chain, pocket_info, trajectory_path)
    
    print(f"\n✓ Pipeline test passed!")
    print(f"  Pocket ID: {pocket_id}")
    print(f"  Ligand name: {ligand_name}")
    print(f"  Ligand size: {ligand_size}")
    print(f"  Output directory: {output_dir}")


def test_pipeline_with_ground_truth_comparison(device):
    """Test pipeline and compare with ground truth ligand coordinates."""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing pipeline with comparison: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break
    
    # Get ground truth ligand coordinates
    ligand_atoms_cont, ligand_coords_gt, ligand_bonds_gt = get_ligand_atoms_and_coords(mol)
    ligand_size = len(ligand_atoms_cont)
    ligand_coords_gt = np.array(ligand_coords_gt)
    
    print(f"Ligand size: {ligand_size}")
    
    # Step 1: Run disc model
    print("\n=== Step 1: Running DISC model ===")
    disc_results, disc_chain, pocket_info = run_disc_mode(
        pocket_structure=pocket_pdb,
        ligand_size=ligand_size,
        device=device,
    )
    
    dist_matrix = disc_results['dist_matrix']
    ligand_atoms_indices = disc_results['ligand_atoms']
    ligand_bonds = disc_results['ligand_bonds']
    
    # Convert atom indices to names
    ligand_atoms = convert_atom_indices_to_names(ligand_atoms_indices)
    
    # Step 2: Run cont model
    print("\n=== Step 2: Running CONT model ===")
    final_prediction, cont_chain, pocket_info = run_cont_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=dist_matrix.numpy() if isinstance(dist_matrix, torch.Tensor) else dist_matrix,
        ligand_atoms=ligand_atoms,
        ligand_bonds=ligand_bonds.numpy() if isinstance(ligand_bonds, torch.Tensor) else ligand_bonds,
        device=device,
    )
    
    # Extract predicted ligand coordinates
    protein_size = len(pocket_info['coords'])
    final_prediction_np = final_prediction.detach().cpu().numpy()
    ligand_coords_pred = final_prediction_np[protein_size:]
    
    # Calculate RMSD
    if ligand_coords_pred.shape[0] == ligand_coords_gt.shape[0]:
        # Align coordinates (center of mass)
        ligand_coords_pred_centered = ligand_coords_pred - ligand_coords_pred.mean(axis=0)
        ligand_coords_gt_centered = ligand_coords_gt - ligand_coords_gt.mean(axis=0)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((ligand_coords_pred_centered - ligand_coords_gt_centered) ** 2, axis=1)))
        print(f"\nLigand RMSD (centered): {rmsd:.3f} Å")
    else:
        print(f"\nWarning: Ligand size mismatch (predicted: {ligand_coords_pred.shape[0]}, ground truth: {ligand_coords_gt.shape[0]})")
    
    # Save results
    output_dir = f"test_outputs/pipeline_comparison_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)
    
    original_ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(original_ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)
    
    save_predictions(dist_matrix, ligand_atoms_indices, ligand_bonds, output_dir, prefix="disc")
    
    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    save_coords_pdb(final_prediction, pocket_info, predicted_pdb_path)
    
    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    save_coords_trajectory(cont_chain, pocket_info, trajectory_path)
    
    print(f"✓ Pipeline with comparison test passed!")
    print(f"  Output directory: {output_dir}")


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

