import os
import glob
import torch
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.disc.dataset import (
    parse_pocket,
    init_dist_features,
    get_ligand_atoms_and_coords,
    LIGAND_ATOM_TYPES,
    LIGAND_BOND_TYPES,
)


@torch.no_grad()
def run_disc_mode(
    pocket_structure,
    ligand_size: int,
    disc_checkpoint: str = None,
    device: torch.device = None,
    seed: int = None,
):
    """High-level prediction of discrete features using DiscModel.
    
    Args:
        pocket_structure: PDB string of protein pocket
        ligand_size: Number of ligand atoms to predict
        disc_checkpoint: Path to disc model checkpoint (None for auto-detection)
        device: PyTorch device (None for auto-detection)
        seed: Random seed (None for random)
    
    Returns:
        dict: Dictionary with predicted values
            - 'dist_matrix': [N, N] tensor of predicted distance class indices
            - 'ligand_atoms': [ligand_size] tensor of predicted ligand atom class indices  
            - 'ligand_bonds': [ligand_size, ligand_size] tensor of predicted ligand bond class indices
            - 'pocket_info': dict with pocket metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
    
    from src.pdb_utils import Structure
    from io import StringIO
    
    # Parse pocket structure
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info = parse_pocket(structure)
    
    # Build features
    features = init_dist_features(pocket_info, ligand_size)
    
    # Convert to tensors and batch dimension
    data = {
        'seq': torch.tensor(features['seq'], dtype=torch.float32).unsqueeze(0),
        'z': torch.tensor(features['z'], dtype=torch.float32).unsqueeze(0),
        'bb_dist': torch.tensor(features['bb_dist'], dtype=torch.int64).unsqueeze(0),
        'seq_mask': torch.tensor(features['seq_mask'], dtype=torch.float32).unsqueeze(0),
        'pair_mask': torch.tensor(features['pair_mask'], dtype=torch.float32).unsqueeze(0),
        'seq_ligand_mask': torch.tensor(features['seq_ligand_mask'], dtype=torch.float32).unsqueeze(0),
        'pair_ligand_mask': torch.tensor(features['pair_ligand_mask'], dtype=torch.float32).unsqueeze(0),
    }
    
    # Move to device
    for key in data:
        data[key] = data[key].to(device)
    
    if disc_checkpoint is None:
        disc_checkpoint = pick_latest(['checkpoints/*disc*/*.ckpt'])
    
    print(f"Loading model from: {disc_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(disc_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_predictions, chain = model.sample_chain(data=data)
    
    # Extract predictions
    # final_predictions is a dict with 'dist', 'atoms', 'bonds'
    protein_size = len(pocket_info['coords'])
    
    # Get ligand-only parts (remove first dimension which is batch)
    results = {
        'dist_matrix': final_predictions['dist'][0].view(protein_size + ligand_size, protein_size + ligand_size).cpu(),
        'ligand_atoms': final_predictions['atoms'][0, protein_size:].cpu(),
        'ligand_bonds': final_predictions['bonds'][0].view(protein_size + ligand_size, protein_size + ligand_size)[protein_size:, protein_size:].cpu(),
        'pocket_info': pocket_info,
    }
    
    return results


def save_predictions(
    dist_matrix,
    ligand_atoms,
    ligand_bonds,
    output_dir,
    prefix="disc",
):
    """Save disc predictions to files.
    
    Args:
        dist_matrix: [N, N] numpy array or tensor of distance class indices
        ligand_atoms: [ligand_size] numpy array or tensor of ligand atom class indices  
        ligand_bonds: [ligand_size, ligand_size] numpy array or tensor of ligand bond class indices
        output_dir: Directory to save files
        prefix: Prefix for output filenames
    """
    _ensure_parent(output_dir)
    
    # Convert to numpy if needed
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    if isinstance(ligand_atoms, torch.Tensor):
        ligand_atoms = ligand_atoms.cpu().numpy()
    if isinstance(ligand_bonds, torch.Tensor):
        ligand_bonds = ligand_bonds.cpu().numpy()
    
    # Ensure integer type
    dist_matrix = dist_matrix.astype(np.int64)
    ligand_atoms = ligand_atoms.astype(np.int64)
    ligand_bonds = ligand_bonds.astype(np.int64)
    
    # Save distance matrix
    dist_file = os.path.join(output_dir, f"{prefix}_dist_matrix.txt")
    np.savetxt(dist_file, dist_matrix, fmt='%d')
    print(f"Saved distance matrix to {dist_file}")
    
    # Save ligand atoms (as class indices, which can be converted to names)
    atoms_file = os.path.join(output_dir, f"{prefix}_ligand_atoms_classes.txt")
    np.savetxt(atoms_file, ligand_atoms, fmt='%d')
    print(f"Saved ligand atoms to {atoms_file}")
    
    # Save ligand bonds
    bonds_file = os.path.join(output_dir, f"{prefix}_ligand_bonds.txt")
    np.savetxt(bonds_file, ligand_bonds, fmt='%d')
    print(f"Saved ligand bonds to {bonds_file}")
    
    # Also save ligand atoms as names (for convenience)
    atoms_names_file = os.path.join(output_dir, f"{prefix}_ligand_atoms_names.txt")
    with open(atoms_names_file, 'w') as f:
        for atom_idx in ligand_atoms:
            if 0 <= atom_idx < len(LIGAND_ATOM_TYPES):
                f.write(f"{LIGAND_ATOM_TYPES[atom_idx]}\n")
            else:
                f.write("C\n")
    print(f"Saved ligand atom names to {atoms_names_file}")


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
