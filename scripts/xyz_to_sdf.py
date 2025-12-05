#!/usr/bin/env python

import argparse
from pathlib import Path
import pymol
from pymol import cmd


def xyz_to_sdf(xyz_path: Path, sdf_path: Path):
    """
    Convert XYZ file to SDF using PyMOL.
    PyMOL will automatically infer bonds based on atomic distances.
    """
    try:
        # Load XYZ file
        cmd.load(str(xyz_path), 'molecule')
        
        # Check if molecule was loaded successfully
        if cmd.count_atoms('molecule') == 0:
            print(f"[WARN] No atoms found in {xyz_path}")
            cmd.delete('all')
            return False
        
        # Save as SDF (PyMOL will infer bonds automatically)
        cmd.save(str(sdf_path), 'molecule', format='sdf')
        cmd.delete('all')
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert {xyz_path}: {e}")
        cmd.delete('all')
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert XYZ file to SDF format")
    parser.add_argument("xyz_file", type=str, help="Input XYZ file path")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Output SDF file path (default: same name as input with .sdf extension)")
    args = parser.parse_args()
    
    xyz_path = Path(args.xyz_file)
    if not xyz_path.is_file():
        print(f"[ERROR] XYZ file not found: {xyz_path}")
        return 1
    
    if args.output:
        sdf_path = Path(args.output)
    else:
        sdf_path = xyz_path.with_suffix('.sdf')
    
    # Initialize PyMOL in quiet mode
    pymol.finish_launching(['pymol', '-Q', '-c'])
    
    if xyz_to_sdf(xyz_path, sdf_path):
        print(f"[OK] Successfully converted {xyz_path} to {sdf_path}")
        return 0
    else:
        print(f"[ERROR] Failed to convert {xyz_path}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

