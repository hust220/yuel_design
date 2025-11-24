import os
import torch
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.disc2.dataset import (
    parse_pocket,
    init_dist_features,
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
        tuple: (results, chain, pocket_info)
            - results: dict with predicted values
                - 'dist_matrix': [N, N] tensor of predicted distance class indices
                - 'ligand_atoms': [ligand_size] tensor of predicted ligand atom class indices  
            - chain: list of dicts containing all diffusion steps
            - pocket_info: dict with pocket metadata
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
    }
    
    # Move to device
    for key in data:
        data[key] = data[key].to(device)
    
    if disc_checkpoint is None:
        disc_checkpoint = pick_latest(['checkpoints/*disc2*/*.ckpt'])
    
    print(f"Loading model from: {disc_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(disc_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_predictions, chain = model.sample_chain(data=data)
    
    # Extract predictions
    # final_predictions is a dict with 'dist', 'atoms' (model doesn't predict bonds)
    protein_size = len(pocket_info['coords'])
    
    # Process chain to match results format
    processed_chain = []
    for step_dict in chain:
        processed_step = {
            'dist_matrix': step_dict['dist'][0].view(protein_size + ligand_size, protein_size + ligand_size).cpu(),
            'ligand_atoms': step_dict['atoms'][0, protein_size:].cpu(),
        }
        processed_chain.append(processed_step)
    
    # Get ligand-only parts for final results (remove first dimension which is batch)
    results = {
        'dist_matrix': final_predictions['dist'][0].view(protein_size + ligand_size, protein_size + ligand_size).cpu(),
        'ligand_atoms': final_predictions['atoms'][0, protein_size:].cpu(),
    }
    
    return results, processed_chain, pocket_info


def save_predictions(
    dist_matrix,
    ligand_atoms,
    output_dir=None,
    prefix="disc",
):
    """Save disc predictions to files.
    
    Args:
        dist_matrix: [N, N] numpy array or tensor of distance class indices
        ligand_atoms: [ligand_size] numpy array or tensor of ligand atom class indices  
        output_dir: Directory to save files
        prefix: Prefix for output filenames
    """
    _ensure_parent(output_dir)
    
    # Convert to numpy if needed
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    if isinstance(ligand_atoms, torch.Tensor):
        ligand_atoms = ligand_atoms.cpu().numpy()
    
    # Ensure integer type
    dist_matrix = dist_matrix.astype(np.int64)
    ligand_atoms = ligand_atoms.astype(np.int64)
    
    # Save distance matrix
    dist_file = os.path.join(output_dir, f"{prefix}_dist_matrix.txt")
    np.savetxt(dist_file, dist_matrix, fmt='%d')
    print(f"Saved distance matrix to {dist_file}")
    
    # Save ligand atoms (as class indices, which can be converted to names)
    atoms_file = os.path.join(output_dir, f"{prefix}_ligand_atoms_classes.txt")
    np.savetxt(atoms_file, ligand_atoms, fmt='%d')
    print(f"Saved ligand atoms to {atoms_file}")
    
    # Also save ligand atoms as names (for convenience)
    atoms_names_file = os.path.join(output_dir, f"{prefix}_ligand_atoms_names.txt")
    with open(atoms_names_file, 'w') as f:
        for atom_idx in ligand_atoms:
            if 0 <= atom_idx < len(LIGAND_ATOM_TYPES):
                f.write(f"{LIGAND_ATOM_TYPES[atom_idx]}\n")
            else:
                f.write("C\n")
    print(f"Saved ligand atom names to {atoms_names_file}")


def save_ligand_sdf_from_predictions(
    ligand_atoms,
    ligand_bonds,
    output_path,
    ligand_coords=None,
):
    """Save ligand as SDF file from predicted atoms and bonds.
    
    Args:
        ligand_atoms: [ligand_size] numpy array or tensor of ligand atom class indices
        ligand_bonds: [ligand_size, ligand_size] numpy array or tensor of ligand bond class indices
        output_path: Path to save SDF file
        ligand_coords: [ligand_size, 3] numpy array or tensor of ligand coordinates (optional)
            If None, will use dummy coordinates
    """
    from rdkit import Chem, Geometry
    
    _ensure_parent(output_path)
    
    # Convert to numpy if needed
    if isinstance(ligand_atoms, torch.Tensor):
        ligand_atoms = ligand_atoms.cpu().numpy()
    if isinstance(ligand_bonds, torch.Tensor):
        ligand_bonds = ligand_bonds.cpu().numpy()
    if ligand_coords is not None and isinstance(ligand_coords, torch.Tensor):
        ligand_coords = ligand_coords.cpu().numpy()
    
    ligand_size = len(ligand_atoms)
    
    # Create RDKit molecule
    mol = Chem.RWMol()
    
    # Add atoms
    for atom_idx in ligand_atoms:
        if 0 <= atom_idx < len(LIGAND_ATOM_TYPES):
            atom_symbol = LIGAND_ATOM_TYPES[atom_idx]
        else:
            atom_symbol = 'C'
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)
    
    # Add bonds
    for i in range(ligand_size):
        for j in range(i + 1, ligand_size):
            bond_idx = ligand_bonds[i, j]
            if bond_idx > 0 and bond_idx < len(LIGAND_BOND_TYPES):
                bond_type_str = LIGAND_BOND_TYPES[bond_idx]
                # Convert bond type string to RDKit bond type
                # All bond types (SINGLE, DOUBLE, TRIPLE, AROMATIC) are merged into BONDED
                # Use SINGLE as default when BONDED is predicted
                if bond_type_str == 'BONDED':
                    bond_type = Chem.BondType.SINGLE
                else:
                    raise ValueError(f"Invalid bond type: {bond_type_str}")
                mol.AddBond(i, j, bond_type)
    
    # Add coordinates if provided, otherwise generate 2D coordinates with RDKit
    # Generate 2D coordinates with RDKit
    from rdkit.Chem import AllChem
    AllChem.Compute2DCoords(mol)
    
    # Save as SDF
    Chem.MolToMolFile(mol, output_path)
    print(f"Saved ligand SDF to {output_path}")


def save_dist_matrix_png(dist_matrix, output_path, title="Distance Matrix", dpi=100):
    """Save distance matrix as PNG image.
    
    Args:
        dist_matrix: [N, N] numpy array or tensor of distance class indices
        output_path: Path to save PNG file
        title: Title for the plot
        dpi: Dots per inch
    """
    import matplotlib.pyplot as plt
    
    _ensure_parent(output_path)
    
    # Convert to numpy if needed
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot distance matrix
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('Atom Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance Class')
    
    # Save as PNG
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved distance matrix PNG to {output_path}")


def save_dist_matrix_gif(chain, output_path, title="Distance Matrix Diffusion", fps=10, dpi=100):
    """Save distance matrix diffusion chain as GIF animation.
    
    Args:
        chain: list of dicts, each containing 'dist_matrix'
        output_path: Path to save GIF file
        title: Title for animation
        fps: Frames per second
        dpi: Dots per inch
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    _ensure_parent(output_path)
    
    # Extract distance matrices from chain
    dist_matrices = []
    for step_dict in chain:
        dist_matrix = step_dict['dist_matrix']
        if isinstance(dist_matrix, torch.Tensor):
            dist_matrix = dist_matrix.cpu().numpy()
        dist_matrices.append(dist_matrix)
    
    if not dist_matrices:
        print("Warning: Empty chain, skipping GIF generation")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize with first frame
    im = ax.imshow(dist_matrices[0], cmap='viridis', aspect='equal')
    ax.set_title(f"{title} - Frame 0/{len(dist_matrices)-1}")
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('Atom Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance Class')
    
    def animate(frame):
        im.set_array(dist_matrices[frame])
        ax.set_title(f"{title} - Frame {frame}/{len(dist_matrices)-1}")
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(dist_matrices), 
                                   interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    
    print(f"Saved distance matrix GIF to {output_path}")


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
