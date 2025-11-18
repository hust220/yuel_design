import argparse
import os
import torch
import numpy as np
from pathlib import Path

# Local imports
from src.console import section, info, success, warn, error
from src.distance_discretization import get_bin_edges, classes_to_distances
from src.disc2.app import (
    run_disc_mode,
    save_dist_matrix_png,
    save_dist_matrix_gif,
)
from src.disc2.dataset import LIGAND_ATOM_TYPES
from src.cont2.app import (
    run_cont_mode,
    save_structure_pdb,
    save_trajectory,
)


def parse_args():
    p = argparse.ArgumentParser(description='Yuel Design: Protein-Ligand Design Pipeline (Disc + Cont)')

    # Pipeline stages
    p.add_argument('--pipeline', type=str, default='disc:cont', 
                   help='Pipeline stages: disc, cont, or disc:cont (default: disc:cont)')

    # Model checkpoints
    p.add_argument('--disc_checkpoint', type=str, default=None, 
                   help='Checkpoint path for disc model (distance/atom prediction)')
    p.add_argument('--cont_checkpoint', type=str, default=None, 
                   help='Checkpoint path for cont model (coordinate prediction)')

    # Input/Output
    p.add_argument('--input_pdb', type=str, required=True, 
                   help='Input protein PDB file')
    p.add_argument('--output_dir', type=str, default='output', 
                   help='Output directory for results')
    
    # (No save control args needed - all outputs are saved by default)

    # Generation parameters
    p.add_argument('--ligand_size', type=int, required=True, 
                   help='Total number of ligand atoms to generate (for cont stage, including all atoms)')
    p.add_argument('--disc_ligand_size', type=int, default=5,
                   help='Number of reduced ligand atoms for disc stage (non-C + ring centers, default: 5)')
    p.add_argument('--interaction_cutoff', type=float, default=5.0,
                   help='Distance cutoff (angstrom) for receptor-ligand interactions (default: 5.0)')
    p.add_argument('--seed', type=int, default=None, 
                   help='Random seed (None for random)')
    p.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'], 
                   help='Device to use (default: auto)')

    return p.parse_args()


def parse_pipeline(pipeline_str: str) -> list:
    """Parse pipeline string like 'disc:cont' or 'disc' into list of stages"""
    stage_mapping = {
        'disc': 'disc', 
        'disc2': 'disc',
        'discrete': 'disc',
        'cont': 'cont',
        'cont2': 'cont',
        'continuous': 'cont',
    }
    
    if ':' in pipeline_str:
        stages = pipeline_str.split(':')
    else:
        stages = [pipeline_str]
    
    parsed_stages = []
    for stage in stages:
        parsed_stage = stage_mapping.get(stage.strip().lower())
        if parsed_stage is None:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {list(stage_mapping.keys())}")
        parsed_stages.append(parsed_stage)
    
    return parsed_stages


def pick_device(user_choice: str) -> torch.device:
    """Select device based on user choice and availability"""
    if user_choice == 'cpu':
        return torch.device('cpu')
    if user_choice == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    if path:
        os.makedirs(path, exist_ok=True)


def build_interaction_index_from_reduced(
    dist_matrix, 
    receptor_reduced_coords,
    ligand_reduced_coords, 
    receptor_atoms,
    cutoff_angstrom=5.0, 
    config_name='b12'
):
    """Build interaction index from predicted distance matrix (with reduced indices).
    
    This function:
    1. Finds receptor-ligand pairs with distance <= cutoff in the reduced distance matrix
    2. Converts reduced indices to full indices using receptor_atoms mapping
    
    Args:
        dist_matrix: [N_r + N_l, N_r + N_l] tensor of distance class indices
                     where N_r = len(receptor_reduced_coords), N_l = len(ligand_reduced_coords)
        receptor_reduced_coords: Array of reduced receptor coordinates (CA + non-C + ring centers)
        ligand_reduced_coords: Array of reduced ligand coordinates (non-C + ring centers)
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
        cutoff_angstrom: Distance cutoff in angstroms for interactions
        config_name: Distance discretization config name
    
    Returns:
        List of tuples: [(receptor_full_idx, ligand_full_idx), ...]
                        where indices are in full atoms space
    """
    # Get bin edges to determine cutoff bin
    bin_edges = get_bin_edges(config_name)
    
    # Find the bin index corresponding to cutoff_angstrom
    cutoff_bin = None
    for i, edge in enumerate(bin_edges):
        if cutoff_angstrom <= edge:
            cutoff_bin = i
            break
    
    if cutoff_bin is None:
        cutoff_bin = len(bin_edges) - 1
    
    info(f"Using distance cutoff: {cutoff_angstrom} Å (bin <= {cutoff_bin})")
    
    # Build mapping from reduced receptor index to full receptor index
    reduced_to_full_receptor = {}
    reduced_idx = 0
    full_idx_offset = 0
    
    for reduced_atoms_list, full_atoms_list in receptor_atoms:
        for atom_name in reduced_atoms_list:
            if atom_name in full_atoms_list:
                full_atom_idx = full_atoms_list.index(atom_name)
                reduced_to_full_receptor[reduced_idx] = full_idx_offset + full_atom_idx
            reduced_idx += 1
        full_idx_offset += len(full_atoms_list)
    
    n_receptor_full = full_idx_offset
    n_receptor_reduced = len(receptor_reduced_coords)
    n_ligand_reduced = len(ligand_reduced_coords)
    
    # Extract interactions from distance matrix
    # dist_matrix indices: [0, n_receptor_reduced) = receptor, [n_receptor_reduced, ...) = ligand
    int_index = []
    
    for i_reduced in range(n_receptor_reduced):
        for j_reduced in range(n_ligand_reduced):
            ligand_idx_in_matrix = n_receptor_reduced + j_reduced
            dist_class = dist_matrix[i_reduced, ligand_idx_in_matrix].item()
            
            # If distance class <= cutoff_bin, it's an interaction
            if dist_class <= cutoff_bin:
                # Convert to full indices
                receptor_full_idx = reduced_to_full_receptor.get(i_reduced)
                if receptor_full_idx is None:
                    continue  # Skip if mapping not found
                
                # Ligand reduced atoms are at the front of ligand full atoms
                # So ligand_reduced_idx j maps to ligand_full_idx j
                ligand_full_idx = n_receptor_full + j_reduced
                
                int_index.append((receptor_full_idx, ligand_full_idx))
    
    info(f"Found {len(int_index)} receptor-ligand interactions (distance <= {cutoff_angstrom} Å)")
    info(f"Receptor: {n_receptor_reduced} reduced atoms -> {n_receptor_full} full atoms")
    info(f"Ligand: {n_ligand_reduced} reduced atoms")
    
    return int_index


def convert_ligand_atoms_to_names(ligand_atom_indices):
    """Convert ligand atom class indices to atom names.
    
    Args:
        ligand_atom_indices: [ligand_size] tensor or numpy array of atom class indices
    
    Returns:
        List of atom names (with _ prefix, e.g., ['_O', '_N', '_C'])
    """
    if isinstance(ligand_atom_indices, torch.Tensor):
        ligand_atom_indices = ligand_atom_indices.cpu().numpy()
    
    ligand_atom_names = []
    for atom_idx in ligand_atom_indices:
        if 0 <= atom_idx < len(LIGAND_ATOM_TYPES):
            atom_name = LIGAND_ATOM_TYPES[atom_idx]
        else:
            atom_name = '_C'  # Default to carbon
        ligand_atom_names.append(atom_name)
    
    return ligand_atom_names


def run_disc_stage(args, device):
    """Run disc stage: predict distance matrix and atom types"""
    section("DISC STAGE: Predicting Distance Matrix and Atom Types")
    
    # Disc stage ligand_size for reduced atoms (non-C + ring centers)
    disc_ligand_size = args.disc_ligand_size
    info(f"Disc stage using ligand_size: {disc_ligand_size} (reduced atoms)")
    
    # Read pocket structure
    info(f"Reading pocket structure from: {args.input_pdb}")
    with open(args.input_pdb, 'r') as f:
        pocket_structure = f.read()
    
    # Run disc model
    results, chain, pocket_info = run_disc_mode(
        pocket_structure=pocket_structure,
        ligand_size=disc_ligand_size,
        disc_checkpoint=args.disc_checkpoint,
        device=device,
        seed=args.seed,
    )
    
    # Extract predictions
    dist_matrix = results['dist_matrix']
    ligand_atoms = results['ligand_atoms']
    
    success(f"Predicted distance matrix: {dist_matrix.shape}")
    success(f"Predicted ligand atoms: {ligand_atoms.shape}")
    
    # Save disc outputs to output_dir
    ensure_dir(args.output_dir)
    
    # Save ligand atom names (no need to save classes)
    ligand_atom_names_list = convert_ligand_atoms_to_names(ligand_atoms)
    atoms_names_file = os.path.join(args.output_dir, 'ligand_atoms.txt')
    with open(atoms_names_file, 'w') as f:
        for atom_name in ligand_atom_names_list:
            f.write(f"{atom_name}\n")
    success(f"Saved ligand atoms to: {atoms_names_file}")
    
    # Save distance matrix as PNG
    png_path = os.path.join(args.output_dir, 'dist_matrix.png')
    save_dist_matrix_png(dist_matrix, png_path, title='Predicted Distance Matrix')
    success(f"Saved distance matrix PNG to: {png_path}")
    
    # Save distance matrix diffusion as GIF
    if len(chain) > 0:
        gif_path = os.path.join(args.output_dir, 'dist_matrix_diffusion.gif')
        save_dist_matrix_gif(chain, gif_path, title='Distance Matrix Diffusion')
        success(f"Saved diffusion GIF to: {gif_path}")
    
    return dist_matrix, ligand_atoms, pocket_info, chain


def run_cont_stage(args, device, dist_matrix, ligand_atoms, pocket_info_disc):
    """Run cont stage: predict continuous coordinates"""
    section("CONT STAGE: Predicting Continuous Coordinates")
    
    # Parse pocket using cont2's parse_pocket to get receptor_atoms structure
    from src.pdb_utils import Structure
    from io import StringIO
    from src.cont2.dataset import parse_pocket as parse_pocket_cont
    
    with open(args.input_pdb, 'r') as f:
        pocket_structure = f.read()
    
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info_cont = parse_pocket_cont(structure)
    
    # Convert predicted ligand atom indices to names (these are REDUCED atoms from disc)
    ligand_reduced_names = convert_ligand_atoms_to_names(ligand_atoms)
    n_ligand_reduced = len(ligand_reduced_names)
    
    info(f"Disc predicted {n_ligand_reduced} reduced ligand atoms: {ligand_reduced_names}")
    
    # Use user-specified ligand_size as full ligand size (including C atoms)
    ligand_full_size = args.ligand_size
    info(f"Cont stage using ligand_size: {ligand_full_size} (full atoms, including C)")
    
    # Prepare ligand_fixed_atoms (all atoms for cont, padded with C)
    n_c_atoms = ligand_full_size - n_ligand_reduced
    if n_c_atoms < 0:
        error(f"Error: User-specified ligand_size ({ligand_full_size}) < reduced atoms from disc ({n_ligand_reduced})")
        error(f"The ligand_size must be >= {n_ligand_reduced} to accommodate the predicted non-C atoms")
        raise ValueError(f"ligand_size must be >= {n_ligand_reduced}")
    
    ligand_fixed_atoms = ligand_reduced_names + ['_C'] * n_c_atoms
    info(f"Ligand composition: {n_ligand_reduced} reduced (from disc) + {n_c_atoms} C = {ligand_full_size} total")
    
    # Create dummy ligand_reduced_coords (we don't actually need the coords, just the count)
    ligand_reduced_coords = np.zeros((n_ligand_reduced, 3))
    
    # Convert distance matrix to interaction index
    receptor_reduced_coords = np.array(pocket_info_disc['coords'])
    receptor_atoms = pocket_info_cont['atoms']  # This has the (reduced, full) structure
    
    int_index = build_interaction_index_from_reduced(
        dist_matrix=dist_matrix,
        receptor_reduced_coords=receptor_reduced_coords,
        ligand_reduced_coords=ligand_reduced_coords,
        receptor_atoms=receptor_atoms,
        cutoff_angstrom=args.interaction_cutoff,
        config_name='b12'
    )
    
    if len(int_index) == 0:
        warn("Warning: No interactions found! The cont model may not work well.")
        warn("Consider increasing --interaction_cutoff or using a different distance prediction.")
    
    # Run cont model with FULL ligand size
    final_coords, chain, pocket_info_cont_final = run_cont_mode(
        pocket_structure=pocket_structure,
        int_index=int_index,
        ligand_fixed_atoms=ligand_fixed_atoms,
        ligand_size=ligand_full_size,
        cont_checkpoint=args.cont_checkpoint,
        device=device,
        seed=args.seed,
    )
    
    success(f"Predicted coordinates: {final_coords.shape}")
    
    # Save cont outputs to output_dir (same as disc)
    # Save predicted structure
    coords_pdb_path = os.path.join(args.output_dir, 'predicted_structure.pdb')
    save_structure_pdb(final_coords, pocket_info_cont_final, coords_pdb_path)
    success(f"Saved predicted structure to: {coords_pdb_path}")
    
    # Save trajectory
    if len(chain) > 0:
        traj_path = os.path.join(args.output_dir, 'trajectory.pdb')
        save_trajectory(chain, pocket_info_cont_final, traj_path)
        success(f"Saved trajectory to: {traj_path}")
    
    return final_coords, chain


def main():
    args = parse_args()
    
    # Parse pipeline stages
    stages = parse_pipeline(args.pipeline)
    info(f"Running pipeline: {' -> '.join(stages)}")
    
    # Select device
    device = pick_device(args.device)
    info(f"Using device: {device}")
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        info(f"Set random seed: {args.seed}")
    
    # Run pipeline stages
    dist_matrix = None
    ligand_atoms = None
    pocket_info = None
    
    for stage in stages:
        if stage == 'disc':
            dist_matrix, ligand_atoms, pocket_info, disc_chain = run_disc_stage(args, device)
        
        elif stage == 'cont':
            if dist_matrix is None or ligand_atoms is None:
                error("Error: 'cont' stage requires 'disc' stage to run first!")
                error("Please use --pipeline disc:cont or run disc stage separately.")
                return
            
            final_coords, cont_chain = run_cont_stage(args, device, dist_matrix, ligand_atoms, pocket_info)
    
    success("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
