#!/usr/bin/env python
"""
Remove LIG residues from docking PDB files in yuel_design, diffsbdd, and pmdm folders.
Generates new receptor PDB files without ligands.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdb_utils import Structure
from tqdm import tqdm


def remove_lig_from_pdb(input_pdb: Path, output_pdb: Path):
    """Remove LIG residues from PDB file and save receptor structure.
    
    Args:
        input_pdb: Input PDB file path
        output_pdb: Output PDB file path (receptor only, no LIG)
    
    Returns:
        True if successful, False otherwise
    """
    structure = Structure()
    try:
        structure.read(str(input_pdb), skip_hetatm=False, skip_water=False)
    except Exception as e:
        print(f"    Warning: Failed to load PDB {input_pdb}: {e}")
        return False
    
    # Create new structure without LIG residues
    from src.pdb_utils import Model, Chain, Residue, Atom
    
    new_structure = Structure()
    
    for model in structure.models:
        new_model = Model(model.model_id)
        
        for chain in model.chains:
            new_chain = Chain(chain.chain_id)
            
            for residue in chain.residues:
                # Skip LIG residues
                if residue.res_name.strip() == "LIG":
                    continue
                
                # Copy residue to new chain
                new_residue = Residue(
                    residue.res_name,
                    residue.res_id,
                    chain.chain_id,
                    residue.insertion
                )
                
                for atom in residue.atoms:
                    new_atom = Atom(
                        record=atom.record,
                        atom_id=atom.atom_id,
                        atom_name=atom.atom_name,
                        alt_loc=atom.alt_loc,
                        res_name=atom.res_name,
                        chain_id=chain.chain_id,
                        res_id=residue.res_id,
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
            
            if len(new_chain.residues) > 0:
                new_model.add_chain(new_chain)
        
        if len(new_model.chains) > 0:
            new_structure.models.append(new_model)
    
    # Write receptor structure
    try:
        new_structure.write(str(output_pdb))
        return True
    except Exception as e:
        print(f"    Warning: Failed to write receptor PDB {output_pdb}: {e}")
        return False




def get_dock_pattern(folder_name: str, protein_id: str) -> str:
    """Get file pattern for dock PDB files based on folder name.
    
    Args:
        folder_name: Folder name (yuel_design, diffsbdd, pmdm)
        protein_id: Protein ID (3jqa, 6cyb)
    
    Returns:
        File pattern string
    """
    if folder_name == "yuel_design":
        return f"{protein_id}_yueldesign_*_*_lig_dock.pdb"
    else:
        return f"{protein_id}_{folder_name}_*_*_dock.pdb"


def process_folder(folder_path: Path, folder_name: str):
    """Process all dock PDB files in a folder and remove LIG residues.
    
    Args:
        folder_path: Path to folder containing dock PDB files
        folder_name: Name of the folder (yuel_design, diffsbdd, pmdm)
    
    Returns:
        Tuple of (success_count, total_count)
    """
    if not folder_path.exists():
        print(f"  Warning: Folder does not exist: {folder_path}")
        return (0, 0)
    
    protein_ids = ["3jqa", "6cyb"]
    all_files = []
    
    for protein_id in protein_ids:
        pattern = get_dock_pattern(folder_name, protein_id)
        files = sorted(folder_path.glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        print(f"  No docking PDB files found in {folder_name}")
        return (0, 0)
    
    print(f"  Found {len(all_files)} docking PDB files in {folder_name}")
    
    success_count = 0
    for dock_pdb in tqdm(all_files, desc=f"  {folder_name}", unit="file", leave=False):
        # Output filename: {original_name}_receptor.pdb
        output_name = dock_pdb.stem + "_receptor.pdb"
        output_pdb = folder_path / output_name
        
        if remove_lig_from_pdb(dock_pdb, output_pdb):
            success_count += 1
    
    return (success_count, len(all_files))


def main():
    parser = argparse.ArgumentParser(
        description="Remove LIG residues from docking PDB files in yuel_design, diffsbdd, and pmdm folders."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["yuel_design", "diffsbdd", "pmdm"],
        help="Folders to process (default: yuel_design diffsbdd pmdm)",
    )
    args = parser.parse_args()
    
    root = Path(__file__).parent
    
    print(f"Processing folders: {', '.join(args.folders)}")
    print(f"{'='*60}")
    
    total_success = 0
    total_files = 0
    
    for folder_name in args.folders:
        folder_path = root / folder_name
        print(f"\nProcessing {folder_name}...")
        
        success_count, file_count = process_folder(folder_path, folder_name)
        total_success += success_count
        total_files += file_count
        
        if file_count > 0:
            print(f"  Successfully processed: {success_count}/{file_count} files")
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total successfully processed: {total_success}/{total_files} files")
    print(f"Receptor PDB files saved to respective folders")


if __name__ == "__main__":
    main()

