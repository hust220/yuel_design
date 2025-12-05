#!/usr/bin/env python

import argparse
from pathlib import Path
import pymol
from pymol import cmd


def extract_ligand_from_pdb(pdb_path: Path, sdf_path: Path):
    """
    Extract ligand from PDB file and save as SDF.
    The ligand is identified by residue name 'LIG'.
    """
    try:
        # Load PDB file
        cmd.load(str(pdb_path), 'structure')
        
        # Select ligand by residue name
        cmd.select('ligand', 'resn LIG')
        
        # Check if ligand was found
        if cmd.count_atoms('ligand') == 0:
            print(f"[WARN] No ligand found (resn LIG) in {pdb_path}")
            cmd.delete('all')
            return False
        
        # Save ligand as SDF
        cmd.save(str(sdf_path), 'ligand', format='sdf')
        cmd.delete('all')
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract ligand from {pdb_path}: {e}")
        cmd.delete('all')
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract ligand from YuelDesign PDB file and convert to SDF")
    parser.add_argument("pdb_file", type=str, help="Input PDB file path")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output SDF file path (default: same name as input with .sdf extension)")
    args = parser.parse_args()
    
    pdb_path = Path(args.pdb_file)
    if not pdb_path.is_file():
        print(f"[ERROR] PDB file not found: {pdb_path}")
        return 1
    
    if args.output:
        sdf_path = Path(args.output)
    else:
        sdf_path = pdb_path.with_suffix('.sdf')
    
    # Initialize PyMOL in quiet mode
    pymol.finish_launching(['pymol', '-Q', '-c'])
    
    if extract_ligand_from_pdb(pdb_path, sdf_path):
        print(f"[OK] Successfully extracted ligand from {pdb_path} to {sdf_path}")
        return 0
    else:
        print(f"[ERROR] Failed to extract ligand from {pdb_path}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

