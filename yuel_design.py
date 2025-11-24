import argparse
import os
import torch
import numpy as np
from rdkit import Chem, Geometry

from src.console import section, info, success, warn, error
from src.e2efinal.app import (
    run_e2e_mode,
    save_structure_pdb,
    save_trajectory as save_trajectory_func,
)
from src.e2efinal.dataset import LIGAND_ATOM_TYPES


def parse_args():
    p = argparse.ArgumentParser(description='Yuel Design: End-to-End Protein-Ligand Design')

    # Model checkpoint
    p.add_argument('--checkpoint', type=str, default=None, 
                   help='Checkpoint path (None for auto-detection)')

    # Input/Output
    p.add_argument('-i', '--input', type=str, required=True, 
                   help='Input protein PDB file')
    p.add_argument('-o', '--output', type=str, default=None,
                   help='Output PDB file path (required if -n not specified)')
    p.add_argument('--output-prefix', type=str, default=None,
                   help='Output file prefix for multiple molecules (used with -n)')

    # Generation parameters
    p.add_argument('--ligand_size', type=int, default=None, 
                   help='Number of ligand atoms to generate (None for auto-estimation)')
    p.add_argument('-n', '--num_molecules', type=int, default=1,
                   help='Number of molecules to generate (default: 1)')
    p.add_argument('--seed', type=int, default=None, 
                   help='Random seed (None for random)')
    p.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'], 
                   help='Device to use (default: auto)')
    p.add_argument('--save_trajectory', type=str, default=None,
                   help='Trajectory file path (optional, supports {index} placeholder)')
    p.add_argument('--log', type=str, default=None,
                   help='Log file path to save validation statistics (optional, supports {index} placeholder)')
    p.add_argument('--max_attempts', type=int, default=20,
                   help='Max attempts per molecule (default: 20)')

    return p.parse_args()


def pick_device(user_choice: str) -> torch.device:
    if user_choice == 'cpu':
        return torch.device('cpu')
    if user_choice == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str):
    if path and path != '.':
        os.makedirs(path, exist_ok=True)


def build_molecule_from_coords(coords, atom_types):
    coords = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    atom_types = atom_types.detach().cpu().numpy() if isinstance(atom_types, torch.Tensor) else atom_types
    
    mol = Chem.RWMol()
    
    for atom_idx in atom_types:
        if 0 <= atom_idx < len(LIGAND_ATOM_TYPES):
            atom_name = LIGAND_ATOM_TYPES[atom_idx]
            if atom_name.startswith('_'):
                atom_symbol = atom_name[1:]
            elif atom_name == 'X':
                atom_symbol = 'C'
            else:
                atom_symbol = atom_name
        else:
            atom_symbol = 'C'
        mol.AddAtom(Chem.Atom(atom_symbol))
    
    n_atoms = len(atom_types)
    if n_atoms == 0:
        return mol.GetMol()
    
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    
    for i in range(n_atoms):
        for j in range(i):
            atom1_idx = atom_types[i]
            atom2_idx = atom_types[j]
            
            if atom1_idx == 0 or atom2_idx == 0:
                continue
            
            atom1_symbol = LIGAND_ATOM_TYPES[atom1_idx] if atom1_idx < len(LIGAND_ATOM_TYPES) else 'C'
            atom2_symbol = LIGAND_ATOM_TYPES[atom2_idx] if atom2_idx < len(LIGAND_ATOM_TYPES) else 'C'
            
            if atom1_symbol.startswith('_'):
                atom1_symbol = atom1_symbol[1:]
            if atom2_symbol.startswith('_'):
                atom2_symbol = atom2_symbol[1:]
            if atom1_symbol == 'X':
                atom1_symbol = 'C'
            if atom2_symbol == 'X':
                atom2_symbol = 'C'
            
            distance = dists[i, j]
            max_bond_length = get_max_bond_length(atom1_symbol, atom2_symbol)
            
            if distance < max_bond_length:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    
    conf = Chem.Conformer(n_atoms)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Geometry.Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    
    return mol.GetMol()


def get_max_bond_length(atom1, atom2):
    bond_lengths = {
        ('C', 'C'): 1.8, ('C', 'N'): 1.7, ('C', 'O'): 1.6, ('C', 'S'): 2.0,
        ('N', 'N'): 1.6, ('N', 'O'): 1.5, ('O', 'O'): 1.5, ('S', 'S'): 2.2,
        ('C', 'F'): 1.5, ('C', 'Cl'): 2.0, ('C', 'Br'): 2.1, ('C', 'I'): 2.3,
        ('N', 'F'): 1.4, ('O', 'F'): 1.4, ('S', 'F'): 1.8,
    }
    pair = tuple(sorted([atom1, atom2]))
    return bond_lengths.get(pair, 2.5)


def check_ring_sizes(mol):
    ring_info = mol.GetRingInfo()
    ring_sizes = []
    for ring in ring_info.AtomRings():
        ring_sizes.append(len(ring))
    
    invalid_rings = [size for size in ring_sizes if size == 3 or size == 4 or size > 6]
    return len(invalid_rings) == 0, ring_sizes


def check_connectivity(mol):
    fragments = Chem.GetMolFrags(mol, asMols=True)
    return len(fragments) == 1


def check_kekulization(mol):
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return True
    except:
        return False


def validate_ligand(coords, atom_types, n_receptor):
    ligand_coords = coords[n_receptor:]
    ligand_atoms = atom_types[n_receptor:]
    
    non_zero_mask = ligand_atoms != 0
    if non_zero_mask.sum() == 0:
        return False, "ring_size", "No ligand atoms found"
    
    ligand_coords = ligand_coords[non_zero_mask]
    ligand_atoms = ligand_atoms[non_zero_mask]
    
    try:
        mol = build_molecule_from_coords(ligand_coords, ligand_atoms)
        
        is_connected = check_connectivity(mol)
        if not is_connected:
            return False, "intact", "Ligand is fragmented"
        
        valid_rings, ring_sizes = check_ring_sizes(mol)
        if not valid_rings:
            return False, "ring_size", f"Invalid ring sizes found: {ring_sizes}"
        
        can_kekulize = check_kekulization(mol)
        if not can_kekulize:
            return False, "kekulization", "Cannot be kekulized by RDKit"
        
        return True, "valid", "Valid"
    except Exception as e:
        return False, "ring_size", f"Error building molecule: {str(e)}"


def write_validation_log(log_path, n_attempts, is_valid, failure_counts):
    log_dir = os.path.dirname(log_path) or '.'
    ensure_dir(log_dir)
    with open(log_path, 'w') as f:
        f.write("Validation Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total attempts: {n_attempts}\n")
        f.write(f"Successful: {1 if is_valid else 0}\n")
        f.write(f"Failed due to ring size: {failure_counts['ring_size']}\n")
        f.write(f"Failed due to not intact: {failure_counts['intact']}\n")
        f.write(f"Failed due to kekulization: {failure_counts['kekulization']}\n")


def run_design(
    pocket_structure,
    output_pdb_path,
    ligand_size=None,
    checkpoint=None,
    device=None,
    seed=None,
    save_trajectory=None,
    log_path=None,
    max_attempts=10,
    verbose=True
):
    if verbose:
        section("E2E MODE: End-to-End Ligand Generation")
    
    if isinstance(pocket_structure, (str, bytes)):
        if isinstance(pocket_structure, bytes):
            pocket_structure = pocket_structure.decode('utf-8')
        elif os.path.isfile(pocket_structure):
            with open(pocket_structure, 'r') as f:
                pocket_structure = f.read()
    
    current_seed = seed
    is_valid = False
    failure_reason = None
    
    failure_counts = {
        'ring_size': 0,
        'intact': 0,
        'kekulization': 0
    }
    
    final_coords = None
    final_atoms = None
    chain = []
    pocket_info = None
    
    for attempt in range(max_attempts):
        if attempt > 0:
            current_seed = np.random.randint(0, 2**31)
            if verbose:
                info(f"Attempt {attempt + 1}/{max_attempts}: Redesigning ligand...")
        
        final_coords, final_atoms, chain, pocket_info = run_e2e_mode(
            pocket_structure=pocket_structure,
            ligand_size=ligand_size,
            e2e_checkpoint=checkpoint,
            device=device,
            seed=current_seed,
        )
        
        n_receptor = len(pocket_info.get('full_coords', []))
        if n_receptor == 0:
            receptor_mask = (final_atoms == 0).cpu().numpy() if isinstance(final_atoms, torch.Tensor) else (final_atoms == 0)
            n_receptor = receptor_mask.sum()
        
        is_valid, failure_reason, message = validate_ligand(final_coords, final_atoms, n_receptor)
        
        if is_valid:
            if verbose:
                success(f"Predicted coordinates: {final_coords.shape}")
                success(f"Predicted atom types: {final_atoms.shape}")
                if attempt > 0:
                    success(f"Valid ligand generated after {attempt + 1} attempts")
            break
        else:
            if failure_reason in failure_counts:
                failure_counts[failure_reason] += 1
            if verbose:
                warn(f"Validation failed: {message}")
            if attempt == max_attempts - 1:
                if verbose:
                    error(f"Failed to generate valid ligand after {max_attempts} attempts")
                    error("Saving the last generated structure anyway")
                break
    
    n_attempts = attempt + 1
    
    if log_path:
        write_validation_log(log_path, n_attempts, is_valid, failure_counts)
        if verbose:
            info(f"Saved validation statistics to: {log_path}")
    
    if output_pdb_path:
        output_dir = os.path.dirname(output_pdb_path) or '.'
        ensure_dir(output_dir)
        save_structure_pdb(final_coords, final_atoms, pocket_info, output_pdb_path)
        if verbose:
            success(f"Saved predicted structure to: {output_pdb_path}")
    
    if save_trajectory and len(chain) > 0:
        traj_dir = os.path.dirname(save_trajectory) or '.'
        ensure_dir(traj_dir)
        save_trajectory_func(chain, final_atoms, pocket_info, save_trajectory)
        if verbose:
            success(f"Saved trajectory to: {save_trajectory}")
    
    return {
        'is_valid': is_valid,
        'failure_reason': failure_reason,
        'n_attempts': n_attempts,
        'n_failures_ring_size': failure_counts['ring_size'],
        'n_failures_intact': failure_counts['intact'],
        'n_failures_kekulization': failure_counts['kekulization'],
    }


def run(args, device):
    if isinstance(args.input, (bytes, str)) and os.path.isfile(args.input):
        info(f"Reading pocket structure from: {args.input}")
    
    # Check arguments
    if args.num_molecules > 1 and args.output_prefix is None:
        error("--output-prefix is required when generating multiple molecules (-n > 1)")
        return
    
    if args.num_molecules == 1 and args.output is None:
        error("--output is required when generating a single molecule (-n=1)")
        return
    
    # Read pocket structure once
    pocket_structure = args.input
    if os.path.isfile(pocket_structure):
        with open(pocket_structure, 'r') as f:
            pocket_structure = f.read()
    
    # Generate multiple molecules
    if args.num_molecules > 1:
        section(f"Generating {args.num_molecules} molecules")
        
        base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**31)
        valid_count = 0
        failed_count = 0
        
        for i in range(1, args.num_molecules + 1):
            # Generate output file paths
            if args.output_prefix:
                output_pdb = f"{args.output_prefix}_{i}.pdb"
                save_traj = None
                if args.save_trajectory:
                    save_traj = args.save_trajectory.replace('{index}', str(i))
                log_path = None
                if args.log:
                    log_path = args.log.replace('{index}', str(i))
            else:
                output_pdb = args.output
                save_traj = args.save_trajectory
                log_path = args.log
            
            # Use different seed for each molecule
            current_seed = base_seed + i if base_seed is not None else None
            
            info(f"Generating molecule {i}/{args.num_molecules}...")
            
            # Keep retrying until valid
            attempt_count = 0
            while True:
                attempt_count += 1
                if attempt_count > 1:
                    current_seed = np.random.randint(0, 2**31)
                    info(f"  Retry attempt {attempt_count} with seed={current_seed}")
                
                result = run_design(
                    pocket_structure=pocket_structure,
                    output_pdb_path=output_pdb,
                    ligand_size=args.ligand_size,
                    checkpoint=args.checkpoint,
                    device=device,
                    seed=current_seed,
                    save_trajectory=save_traj,
                    log_path=log_path,
                    max_attempts=args.max_attempts,
                    verbose=False  # Less verbose for batch generation
                )
                
                if result['is_valid']:
                    success(f"  ✓ Molecule {i} generated successfully in {result['n_attempts']} attempts")
                    valid_count += 1
                    break
                else:
                    if attempt_count >= 10:  # Limit external retries
                        warn(f"  ✗ Molecule {i} failed after {attempt_count} external retries: {result['failure_reason']}")
                        failed_count += 1
                        break
                    warn(f"  ✗ Molecule {i} failed: {result['failure_reason']}, retrying with new seed...")
        
        section("Summary")
        success(f"Generated {valid_count}/{args.num_molecules} valid molecules")
        if failed_count > 0:
            warn(f"Failed: {failed_count}/{args.num_molecules} molecules")
    else:
        # Single molecule generation
        result = run_design(
            pocket_structure=pocket_structure,
            output_pdb_path=args.output,
            ligand_size=args.ligand_size,
            checkpoint=args.checkpoint,
            device=device,
            seed=args.seed,
            save_trajectory=args.save_trajectory,
            log_path=args.log,
            max_attempts=args.max_attempts,
            verbose=True
        )
        
        if result['is_valid']:
            success("Molecule generated successfully!")
        else:
            error(f"Molecule generation failed: {result['failure_reason']}")
    
    return result if args.num_molecules == 1 else None


def main():
    args = parse_args()
    
    device = pick_device(args.device)
    info(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        info(f"Set random seed: {args.seed}")
    
    run(args, device)
    success("Completed successfully!")


if __name__ == "__main__":
    main()
