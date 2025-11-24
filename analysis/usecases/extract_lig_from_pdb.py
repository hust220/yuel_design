#!/usr/bin/env python
"""
Extract LIG residues from all PDB files in yuel_design folder and save as mol2 files.
Uses PyMOL for conversion.
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

def extract_lig_from_pdb(pdb_file, output_mol2):
    """Extract LIG residue from PDB file and save as mol2."""
    try:
        # Load PDB file
        cmd.load(str(pdb_file))
        
        # Select LIG residue
        cmd.select("ligand", "resn LIG")
        
        # Check if ligand exists
        count = cmd.count_atoms("ligand")
        if count == 0:
            print(f"  Warning: No LIG residue found in {pdb_file.name}")
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
    parser = argparse.ArgumentParser(description='Extract LIG residues from PDB files and save as mol2 files.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force conversion even if target mol2 file already exists (default: False)')
    args = parser.parse_args()
    
    # Initialize PyMOL once at the start
    pymol.finish_launching(['pymol', '-c', '-Q'])
    
    script_dir = Path(__file__).parent
    yuel_design_dir = script_dir / "yuel_design"
    
    if not yuel_design_dir.exists():
        print(f"Error: Directory {yuel_design_dir} does not exist")
        sys.exit(1)
    
    # Find all PDB files
    pdb_files = sorted(yuel_design_dir.glob("*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {yuel_design_dir}")
        sys.exit(0)
    
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Output directory: {yuel_design_dir}")
    if args.force:
        print("Force mode: Will overwrite existing mol2 files")
    print()
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for pdb_file in pdb_files:
        output_mol2 = yuel_design_dir / f"{pdb_file.stem}_lig.mol2"
        
        # Skip if output file already exists (unless force mode)
        if not args.force and output_mol2.exists():
            print(f"Skipping: {pdb_file.name} -> {output_mol2.name} (already exists)")
            skipped_count += 1
            continue
        
        print(f"Processing: {pdb_file.name} -> {output_mol2.name}")
        
        if extract_lig_from_pdb(pdb_file, output_mol2):
            success_count += 1
            print(f"  Success")
        else:
            fail_count += 1
            print(f"  Failed")
    
    print()
    print(f"Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(pdb_files)}")

if __name__ == "__main__":
    main()

