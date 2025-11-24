#!/usr/bin/env python
"""
Extract UNK residues from all _dock.pdb files in yuel_design, pmdm, and diffsbdd folders.
Save as mol2 files using PyMOL.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import pymol
    from pymol import cmd
except ImportError:
    print("Error: PyMOL is not installed. Please install it first.")
    print("You can install PyMOL using: conda install -c conda-forge pymol")
    sys.exit(1)

def extract_unk_from_pdb(pdb_file, output_mol2):
    """Extract UNK residue from PDB file and save as mol2."""
    try:
        # Load PDB file
        cmd.load(str(pdb_file))
        
        # Select UNK residue
        cmd.select("ligand", "resn UNK")
        
        # Check if ligand exists
        count = cmd.count_atoms("ligand")
        if count == 0:
            print(f"  Warning: No UNK residue found in {pdb_file.name}")
            cmd.delete("all")
            return False
        
        # Save as mol2
        cmd.save(str(output_mol2), "ligand")
        
        # Clean up
        cmd.delete("all")
        
        return True
    except Exception as e:
        print(f"  Error processing {pdb_file.name}: {e}")
        try:
            cmd.delete("all")
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract UNK residues from _dock.pdb files and save as mol2 files.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force conversion even if target mol2 file already exists (default: False)')
    args = parser.parse_args()
    
    # Initialize PyMOL once at the start
    pymol.finish_launching(['pymol', '-c', '-Q'])
    
    script_dir = Path(__file__).parent
    
    # Folders to process
    folders = ['yuel_design', 'pmdm', 'diffsbdd']
    
    total_success = 0
    total_fail = 0
    total_skipped = 0
    
    for folder_name in folders:
        folder_dir = script_dir / folder_name
        
        if not folder_dir.exists():
            print(f"Warning: Directory {folder_dir} does not exist, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*60}")
        
        # Find all _dock.pdb files
        dock_files = sorted(folder_dir.glob("*_dock.pdb"))
        
        if not dock_files:
            print(f"No _dock.pdb files found in {folder_dir}")
            continue
        
        print(f"Found {len(dock_files)} _dock.pdb files")
        print(f"Output directory: {folder_dir}")
        if args.force:
            print("Force mode: Will overwrite existing mol2 files")
        print()
        
        success_count = 0
        fail_count = 0
        skipped_count = 0
        
        for dock_file in dock_files:
            # Generate output filename: replace _dock.pdb with _dock_unk.mol2
            output_mol2 = folder_dir / f"{dock_file.stem}_unk.mol2"
            
            # Skip if output file already exists (unless force mode)
            if not args.force and output_mol2.exists():
                print(f"Skipping: {dock_file.name} -> {output_mol2.name} (already exists)")
                skipped_count += 1
                continue
            
            print(f"Processing: {dock_file.name} -> {output_mol2.name}")
            
            if extract_unk_from_pdb(dock_file, output_mol2):
                success_count += 1
                print(f"  Success")
            else:
                fail_count += 1
                print(f"  Failed")
        
        print()
        print(f"Summary for {folder_name}:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total: {len(dock_files)}")
        
        total_success += success_count
        total_fail += fail_count
        total_skipped += skipped_count
    
    print()
    print(f"{'='*60}")
    print(f"Overall Summary")
    print(f"{'='*60}")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_fail}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Total: {total_success + total_fail + total_skipped}")

if __name__ == "__main__":
    main()

