#!/usr/bin/env python
"""
Convert all SDF files in pmdm folder to mol2 files.
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

def convert_sdf_to_mol2(sdf_file, output_mol2):
    """Convert SDF file to mol2 format."""
    try:
        # Load SDF file
        cmd.load(str(sdf_file))
        
        # Get the object name (usually the filename without extension)
        obj_name = cmd.get_object_list()[0]
        
        # Save as mol2
        cmd.save(str(output_mol2), obj_name)
        
        # Clean up
        cmd.delete("all")
        
        return True
    except Exception as e:
        print(f"  Error processing {sdf_file.name}: {e}")
        try:
            cmd.delete("all")
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert SDF files in pmdm folder to mol2 files.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force conversion even if target mol2 file already exists (default: False)')
    args = parser.parse_args()
    
    # Initialize PyMOL once at the start
    pymol.finish_launching(['pymol', '-c', '-Q'])
    
    script_dir = Path(__file__).parent
    pmdm_dir = script_dir / "pmdm"
    
    if not pmdm_dir.exists():
        print(f"Error: Directory {pmdm_dir} does not exist")
        sys.exit(1)
    
    # Find all SDF files
    sdf_files = sorted(pmdm_dir.glob("*.sdf"))
    
    if not sdf_files:
        print(f"No SDF files found in {pmdm_dir}")
        sys.exit(0)
    
    print(f"Found {len(sdf_files)} SDF files")
    print(f"Output directory: {pmdm_dir}")
    if args.force:
        print("Force mode: Will overwrite existing mol2 files")
    print()
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for sdf_file in sdf_files:
        output_mol2 = pmdm_dir / f"{sdf_file.stem}.mol2"
        
        # Skip if output file already exists (unless force mode)
        if not args.force and output_mol2.exists():
            print(f"Skipping: {sdf_file.name} -> {output_mol2.name} (already exists)")
            skipped_count += 1
            continue
        
        print(f"Processing: {sdf_file.name} -> {output_mol2.name}")
        
        if convert_sdf_to_mol2(sdf_file, output_mol2):
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
    print(f"  Total: {len(sdf_files)}")

if __name__ == "__main__":
    main()

