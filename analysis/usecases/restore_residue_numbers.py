#!/usr/bin/env python
"""
Restore original residue numbers and chain IDs in docking PDB files.
MedusaDock renumbers residues starting from 1, but we need to restore
the original numbering from reference protein PDB files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdb_utils import Structure


def build_ca_index(ref_structure: Structure):
    """Build an index of CA atoms from reference structure.
    
    Returns:
        dict mapping (res_name, ca_coord_tuple) -> (res_id, chain_id)
        where ca_coord_tuple is rounded to 2 decimal places for matching
    """
    index = {}
    for model in ref_structure.models:
        for chain in model.chains:
            for residue in chain.residues:
                ca = residue.get_atom("CA")
                if ca is not None:
                    coord = ca.get_coord()
                    # Round to 2 decimal places for matching
                    coord_key = tuple(np.round(coord, 2))
                    index[coord_key] = (residue.res_id, chain.chain_id, residue.res_name)
    return index


def find_matching_residue(ca_coord: np.ndarray, ref_index: dict, threshold: float = 0.5):
    """Find matching residue in reference structure by CA coordinate.
    
    Args:
        ca_coord: CA atom coordinate from docking structure
        ref_index: Reference CA index from build_ca_index
        threshold: Maximum distance for matching (in Angstroms)
    
    Returns:
        (res_id, chain_id, res_name) if match found, None otherwise
    """
    coord_key = tuple(np.round(ca_coord, 2))
    
    # Try exact match first
    if coord_key in ref_index:
        return ref_index[coord_key]
    
    # Try nearby coordinates (within threshold)
    best_match = None
    best_dist = threshold
    
    for ref_key, ref_info in ref_index.items():
        ref_coord = np.array(ref_key)
        dist = np.linalg.norm(ca_coord - ref_coord)
        if dist < best_dist:
            best_dist = dist
            best_match = ref_info
    
    return best_match


def restore_residue_numbers(dock_pdb_path: Path, ref_structure: Structure, output_path: Path):
    """Restore residue numbers and chain IDs in docking PDB file.
    
    Args:
        dock_pdb_path: Path to docking PDB file
        ref_structure: Reference structure with correct numbering
        output_path: Path to save restored PDB file
    """
    # Build reference CA index
    ref_index = build_ca_index(ref_structure)
    
    # Load docking structure
    dock_structure = Structure()
    try:
        dock_structure.read(str(dock_pdb_path), skip_hetatm=False, skip_water=False)
    except Exception as e:
        print(f"    Warning: Failed to load docking PDB {dock_pdb_path}: {e}")
        return False
    
    # Create new structure with restored numbering
    from src.pdb_utils import Model, Chain, Residue, Atom
    
    new_structure = Structure()
    
    for model in dock_structure.models:
        new_model = Model(model.model_id)
        
        # Group residues by their new chain ID
        chain_residues = {}  # chain_id -> list of (residue, new_res_id, new_chain_id)
        
        for chain in model.chains:
            for residue in chain.residues:
                # Skip ligand residues
                if residue.res_name.strip() == "LIG":
                    # Keep ligand as-is, use original chain
                    if chain.chain_id not in chain_residues:
                        chain_residues[chain.chain_id] = []
                    chain_residues[chain.chain_id].append((residue, residue.res_id, chain.chain_id))
                    continue
                
                # Find CA atom
                ca = residue.get_atom("CA")
                if ca is None:
                    # No CA atom, try to match by residue name and position
                    # For now, keep original numbering
                    if chain.chain_id not in chain_residues:
                        chain_residues[chain.chain_id] = []
                    chain_residues[chain.chain_id].append((residue, residue.res_id, chain.chain_id))
                    continue
                
                # Find matching residue in reference
                match = find_matching_residue(ca.get_coord(), ref_index)
                if match is not None:
                    new_res_id, new_chain_id, ref_res_name = match
                    # Update residue name if it matches
                    if residue.res_name.strip() == ref_res_name.strip():
                        if new_chain_id not in chain_residues:
                            chain_residues[new_chain_id] = []
                        chain_residues[new_chain_id].append((residue, new_res_id, new_chain_id))
                    else:
                        # Residue name mismatch, keep original
                        if chain.chain_id not in chain_residues:
                            chain_residues[chain.chain_id] = []
                        chain_residues[chain.chain_id].append((residue, residue.res_id, chain.chain_id))
                else:
                    # No match found, keep original numbering
                    if chain.chain_id not in chain_residues:
                        chain_residues[chain.chain_id] = []
                    chain_residues[chain.chain_id].append((residue, residue.res_id, chain.chain_id))
        
        # Create new chains with restored numbering
        for new_chain_id, residue_list in chain_residues.items():
            new_chain = Chain(new_chain_id)
            
            for residue, new_res_id, actual_chain_id in residue_list:
                new_residue = Residue(residue.res_name, new_res_id, new_chain_id, residue.insertion)
                
                for atom in residue.atoms:
                    new_atom = Atom(
                        record=atom.record,
                        atom_id=atom.atom_id,
                        atom_name=atom.atom_name,
                        alt_loc=atom.alt_loc,
                        res_name=atom.res_name,
                        chain_id=new_chain_id,
                        res_id=new_res_id,
                        insertion=atom.insertion,
                        x=atom.x,
                        y=atom.y,
                        z=atom.z,
                        occupancy=atom.occupancy,
                        temp_factor=atom.temp_factor,
                        element=atom.element,
                        charge=atom.charge,
                    )
                    new_residue.add_atom(new_atom)
                
                new_chain.add_residue(new_residue)
            
            new_model.add_chain(new_chain)
        
        new_structure.models.append(new_model)
    
    # Write restored structure
    try:
        with open(output_path, 'w') as f:
            for model in new_structure.models:
                f.write(model.to_pdb())
        return True
    except Exception as e:
        print(f"    Warning: Failed to write restored PDB {output_path}: {e}")
        return False


def process_folder(folder_path: Path, protein_id: str, ref_pdb_path: Path, root: Path):
    """Process all docking PDB files in a folder.
    
    Args:
        folder_path: Path to folder containing docking PDBs
        protein_id: Protein ID (e.g., "6cyb")
        ref_pdb_path: Path to reference protein PDB
        root: Root directory for output
    """
    if not folder_path.exists():
        print(f"  Warning: Folder does not exist: {folder_path}")
        return
    
    # Load reference structure
    ref_structure = Structure()
    try:
        ref_structure.read(str(ref_pdb_path), skip_hetatm=False, skip_water=False)
    except Exception as e:
        print(f"  Error: Failed to load reference PDB {ref_pdb_path}: {e}")
        return
    
    # Find all docking PDB files
    dock_pattern = f"{protein_id}_*_dock.pdb"
    dock_files = sorted(folder_path.glob(dock_pattern))
    
    if not dock_files:
        print(f"  No docking PDB files found matching {dock_pattern}")
        return
    
    print(f"  Processing {len(dock_files)} docking PDB files in {folder_path.name}...")
    
    from tqdm import tqdm
    success_count = 0
    
    for dock_file in tqdm(dock_files, desc=f"  {folder_path.name}", unit="file"):
        # Create output filename: original_name_renumbered.pdb
        output_name = dock_file.stem + "_renumbered.pdb"
        output_path = folder_path / output_name
        
        if restore_residue_numbers(dock_file, ref_structure, output_path):
            success_count += 1
    
    print(f"  Successfully restored {success_count}/{len(dock_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Restore original residue numbers and chain IDs in docking PDB files."
    )
    parser.add_argument(
        "--protein-id",
        choices=["3jqa", "6cyb"],
        nargs="+",
        default=["3jqa", "6cyb"],
        help="Protein ID(s) to process (default: both 3jqa and 6cyb)",
    )
    parser.add_argument(
        "--ref-pdb",
        help="Path to reference protein PDB (default: {protein_id}_protein.pdb)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["diffsbdd", "pmdm"],
        help="Folders to process (default: diffsbdd pmdm)",
    )
    args = parser.parse_args()
    
    root = Path(__file__).parent
    
    # Ensure protein_id is a list
    if isinstance(args.protein_id, str):
        protein_ids = [args.protein_id]
    else:
        protein_ids = args.protein_id
    
    # Process each protein
    for protein_id in protein_ids:
        print(f"\n{'='*60}")
        print(f"Processing {protein_id.upper()}")
        print(f"{'='*60}")
        
        # Determine reference PDB path
        if args.ref_pdb:
            ref_pdb_path = Path(args.ref_pdb)
        else:
            ref_pdb_path = root / f"{protein_id}_protein.pdb"
        
        if not ref_pdb_path.exists():
            print(f"  Error: Reference PDB not found: {ref_pdb_path}")
            continue
        
        print(f"  Using reference PDB: {ref_pdb_path}")
        
        # Process each folder
        for folder_name in args.folders:
            folder_path = root / folder_name
            if not folder_path.exists():
                print(f"  Warning: Folder does not exist: {folder_path}")
                continue
            
            process_folder(folder_path, protein_id, ref_pdb_path, root)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

