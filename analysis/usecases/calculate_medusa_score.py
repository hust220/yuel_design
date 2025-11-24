#!/usr/bin/env python
"""
Calculate MedusaScore for all docking PDB and ligand mol2 pairs.
Uses complex_interEnergy_charge from MEDUSA toolkit.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_medusa_score(output: str) -> float:
    """Extract MedusaScore from complex_interEnergy_charge output.
    
    Expected format:
    MedusaScore-VDWR= -16.0276 E-Charge= 0
    
    Returns:
        MedusaScore value as float, or None if not found
    """
    # Look for the last line matching the pattern
    lines = output.strip().split('\n')
    for line in reversed(lines):
        # Match pattern: MedusaScore-VDWR= <number> E-Charge= <number>
        match = re.search(r'MedusaScore-VDWR=\s*([-\d.]+)', line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def calculate_medusa_score(pdb_path: Path, mol2_path: Path, medusa_param: str = None) -> float:
    """Calculate MedusaScore for a PDB and mol2 pair.
    
    Args:
        pdb_path: Path to docking PDB file
        mol2_path: Path to ligand mol2 file
        medusa_param: Path to MEDUSA parameter file (uses $MEDUSA_PARAMETER if None)
    
    Returns:
        MedusaScore value, or None if calculation failed
    """
    cmd = ['complex_interEnergy_charge']
    
    if medusa_param:
        param_path = medusa_param
    else:
        # Use environment variable
        param_path = os.environ.get('MEDUSA_PARAMETER')
        if not param_path:
            print(f"    Warning: MEDUSA_PARAMETER environment variable not set")
            return None
    
    cmd.extend(['-p', param_path])
    cmd.extend(['-i', str(pdb_path)])
    cmd.extend(['-m', str(mol2_path)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=pdb_path.parent,
        )
        
        if result.returncode != 0:
            print(f"    Warning: complex_interEnergy_charge failed for {pdb_path.name}: {result.stderr}")
            return None
        
        score = extract_medusa_score(result.stdout)
        if score is None:
            print(f"    Warning: Could not extract MedusaScore from output for {pdb_path.name}")
            return None
        
        return score
    
    except Exception as e:
        print(f"    Warning: Exception running complex_interEnergy_charge for {pdb_path.name}: {e}")
        return None


def find_matching_files(folder: Path, protein_id: str, folder_name: str):
    """Find all matching PDB and mol2 file pairs in a folder.
    
    Args:
        folder: Folder to search
        protein_id: Protein ID (e.g., "6cyb")
        folder_name: Folder name (e.g., "yuel_design", "diffsbdd", "pmdm")
    
    Yields:
        Tuple of (pdb_path, mol2_path, i, j)
    """
    if folder_name == "yuel_design":
        # Pattern: {protein_id}_yueldesign_{i}_{j}_lig_dock.pdb
        #          {protein_id}_yueldesign_{i}_{j}_lig_dock_unk.mol2
        pattern_pdb = f"{protein_id}_yueldesign_*_*_lig_dock.pdb"
        pattern_mol2 = f"{protein_id}_yueldesign_*_*_lig_dock_unk.mol2"
    else:
        # Pattern: {protein_id}_{folder_name}_{i}_{j}_dock.pdb
        #          {protein_id}_{folder_name}_{i}_{j}_dock_unk.mol2
        pattern_pdb = f"{protein_id}_{folder_name}_*_*_dock.pdb"
        pattern_mol2 = f"{protein_id}_{folder_name}_*_*_dock_unk.mol2"
    
    pdb_files = sorted(folder.glob(pattern_pdb))
    
    for pdb_path in pdb_files:
        # Determine corresponding mol2 file
        if folder_name == "yuel_design":
            mol2_name = pdb_path.name.replace("_lig_dock.pdb", "_lig_dock_unk.mol2")
        else:
            mol2_name = pdb_path.name.replace("_dock.pdb", "_dock_unk.mol2")
        
        mol2_path = folder / mol2_name
        
        if not mol2_path.exists():
            print(f"    Warning: Mol2 file not found for {pdb_path.name}: {mol2_name}")
            continue
        
        # Extract i and j from filename
        parts = pdb_path.stem.split('_')
        try:
            if folder_name == "yuel_design":
                # {protein_id}_yueldesign_{i}_{j}_lig_dock
                i = int(parts[2])
                j = int(parts[3])
            else:
                # {protein_id}_{folder_name}_{i}_{j}_dock
                i = int(parts[2])
                j = int(parts[3])
        except (ValueError, IndexError):
            i, j = None, None
        
        yield (pdb_path, mol2_path, i, j)


def process_folder(folder: Path, protein_id: str, folder_name: str, medusa_param: str = None, results: list = None):
    """Process all docking files in a folder and calculate MedusaScore.
    
    Args:
        folder: Folder containing docking files
        protein_id: Protein ID
        folder_name: Folder name (yuel_design, diffsbdd, pmdm)
        medusa_param: MEDUSA parameter file path
        results: List to append results to
    """
    if results is None:
        results = []
    
    if not folder.exists():
        print(f"  Warning: Folder does not exist: {folder}")
        return results
    
    file_pairs = list(find_matching_files(folder, protein_id, folder_name))
    
    if not file_pairs:
        print(f"  No matching file pairs found in {folder.name}")
        return results
    
    print(f"  Processing {len(file_pairs)} file pairs in {folder.name}...")
    
    for pdb_path, mol2_path, i, j in tqdm(file_pairs, desc=f"  {folder.name}", unit="pair"):
        score = calculate_medusa_score(pdb_path, mol2_path, medusa_param)
        
        if score is not None:
            results.append({
                "protein_id": protein_id,
                "folder": folder_name,
                "batch": i,
                "design": j,
                "pdb_file": pdb_path.name,
                "mol2_file": mol2_path.name,
                "medusa_score": score,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate MedusaScore for all docking PDB and mol2 pairs."
    )
    parser.add_argument(
        "--protein-id",
        nargs="+",
        default=["3jqa", "6cyb"],
        choices=["3jqa", "6cyb"],
        help="Protein ID(s) to process (default: both 3jqa and 6cyb)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["yuel_design", "diffsbdd", "pmdm"],
        help="Folders to process (default: yuel_design diffsbdd pmdm)",
    )
    parser.add_argument(
        "--medusa-param",
        help="Path to MEDUSA parameter file (uses $MEDUSA_PARAMETER if not specified)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="medusa_score_results.csv",
        help="Output CSV file (default: medusa_score_results.csv)",
    )
    args = parser.parse_args()
    
    # Check if MEDUSA is available
    try:
        result = subprocess.run(
            ['which', 'complex_interEnergy_charge'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("Warning: complex_interEnergy_charge not found. Please run 'module load medusa' first.")
            sys.exit(1)
    except Exception:
        print("Warning: Could not check for complex_interEnergy_charge. Please ensure MEDUSA is loaded.")
    
    # Check MEDUSA_PARAMETER if not provided
    if not args.medusa_param and 'MEDUSA_PARAMETER' not in os.environ:
        print("Warning: MEDUSA_PARAMETER environment variable not set.")
        print("Please run 'module load medusa' or provide --medusa-param.")
        sys.exit(1)
    
    root = Path(__file__).parent
    
    # Ensure protein_id is a list
    if isinstance(args.protein_id, str):
        protein_ids = [args.protein_id]
    else:
        protein_ids = args.protein_id
    
    all_results = []
    
    # Process each protein
    for protein_id in protein_ids:
        print(f"\n{'='*60}")
        print(f"Processing {protein_id.upper()}")
        print(f"{'='*60}")
        
        # Process each folder
        for folder_name in args.folders:
            folder_path = root / folder_name
            process_folder(folder_path, protein_id, folder_name, args.medusa_param, all_results)
    
    if not all_results:
        print("\nNo MedusaScore values computed.")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = root / args.output
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total calculations: {len(df)}")
    print(f"Results saved to: {output_path}")
    
    # Print statistics
    if len(df) > 0:
        print(f"\nMedusaScore statistics:")
        print(f"  Mean: {df['medusa_score'].mean():.4f}")
        print(f"  Median: {df['medusa_score'].median():.4f}")
        print(f"  Min: {df['medusa_score'].min():.4f}")
        print(f"  Max: {df['medusa_score'].max():.4f}")
        print(f"  Std: {df['medusa_score'].std():.4f}")
        
        print(f"\nBy protein:")
        for protein_id in protein_ids:
            protein_df = df[df['protein_id'] == protein_id]
            if len(protein_df) > 0:
                print(f"  {protein_id}: {len(protein_df)} values, mean: {protein_df['medusa_score'].mean():.4f}")
        
        print(f"\nBy folder:")
        for folder_name in args.folders:
            folder_df = df[df['folder'] == folder_name]
            if len(folder_df) > 0:
                print(f"  {folder_name}: {len(folder_df)} values, mean: {folder_df['medusa_score'].mean():.4f}")


if __name__ == "__main__":
    main()

