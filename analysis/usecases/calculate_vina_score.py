#!/usr/bin/env python
"""
Extract Vina scores from .vina_score files in yuel_design, diffsbdd, and pmdm folders.
Reads Affinity line and extracts the score value.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_vina_score(file_path: Path) -> float:
    """Extract Vina score from a .vina_score file.
    
    Looks for line with "Affinity" and extracts the second field.
    Example: "Affinity: -8.5 kcal/mol" -> -8.5
    Or: "Affinity -8.5 kcal/mol" -> -8.5
    
    Args:
        file_path: Path to .vina_score file
    
    Returns:
        Vina score as float, or None if not found
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Affinity' in line:
                    # Split by whitespace and get second field (index 1)
                    parts = line.split()
                    if len(parts) >= 2:
                        # Try to extract number from second field
                        # Handle cases like "-8.5" or "-8.5," or "(-8.5)"
                        value_str = parts[1].strip(',;()[]')
                        try:
                            return float(value_str)
                        except ValueError:
                            continue
                    # Fallback: try regex pattern
                    match = re.search(r'Affinity\s*:?\s*(-?\d+\.?\d*)', line)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            continue
        return None
    except Exception as e:
        return None  # Silent failure, will be counted in statistics


def get_vina_score_pattern(folder_name: str, protein_id: str) -> str:
    """Get file pattern for vina_score files based on folder name.
    
    Args:
        folder_name: Folder name (yuel_design, diffsbdd, pmdm)
        protein_id: Protein ID (3jqa, 6cyb)
    
    Returns:
        File pattern string
    """
    if folder_name == "yuel_design":
        return f"{protein_id}_yueldesign_*_*.vina_score"
    else:
        return f"{protein_id}_{folder_name}_*_*.vina_score"


def process_folder(folder_path: Path, folder_name: str, protein_id: str, results: list):
    """Process all vina_score files in a folder and extract scores.
    
    Args:
        folder_path: Path to folder containing vina_score files
        folder_name: Name of the folder (yuel_design, diffsbdd, pmdm)
        protein_id: Protein ID
        results: List to append results to
    
    Returns:
        Number of successfully processed files
    """
    if not folder_path.exists():
        return 0
    
    pattern = get_vina_score_pattern(folder_name, protein_id)
    score_files = sorted(folder_path.glob(pattern))
    
    if not score_files:
        return 0
    
    success_count = 0
    for score_file in tqdm(score_files, desc=f"  {folder_name}", unit="file", leave=False):
        vina_score = extract_vina_score(score_file)
        
        if vina_score is not None:
            # Extract batch and design numbers from filename
            parts = score_file.stem.split('_')
            try:
                if folder_name == "yuel_design":
                    # {protein_id}_yueldesign_{i}_{j}.vina_score
                    batch = int(parts[2])
                    design = int(parts[3])
                else:
                    # {protein_id}_{folder_name}_{i}_{j}.vina_score
                    batch = int(parts[2])
                    design = int(parts[3])
            except (ValueError, IndexError):
                batch = None
                design = None
            
            results.append({
                "protein_id": protein_id,
                "folder": folder_name,
                "batch": batch,
                "design": design,
                "score_file": score_file.name,
                "vina_score": vina_score,
            })
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract Vina scores from .vina_score files in yuel_design, diffsbdd, and pmdm folders."
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
        "--output",
        "-o",
        default="vina_score_results.csv",
        help="Output CSV file (default: vina_score_results.csv)",
    )
    args = parser.parse_args()
    
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
            if not folder_path.exists():
                print(f"  Warning: Folder does not exist: {folder_path}")
                continue
            
            print(f"  Processing {folder_name}...")
            count = process_folder(folder_path, folder_name, protein_id, all_results)
            if count > 0:
                print(f"    Extracted {count} Vina scores")
    
    if not all_results:
        print("\nNo Vina scores found.")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = root / args.output
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total scores extracted: {len(df)}")
    print(f"Results saved to: {output_path}")
    
    # Print statistics
    if len(df) > 0:
        print(f"\nVina score statistics:")
        print(f"  Mean: {df['vina_score'].mean():.4f}")
        print(f"  Median: {df['vina_score'].median():.4f}")
        print(f"  Min: {df['vina_score'].min():.4f}")
        print(f"  Max: {df['vina_score'].max():.4f}")
        print(f"  Std: {df['vina_score'].std():.4f}")
        
        print(f"\nBy protein:")
        for protein_id in protein_ids:
            protein_df = df[df['protein_id'] == protein_id]
            if len(protein_df) > 0:
                print(f"  {protein_id}: {len(protein_df)} values, mean: {protein_df['vina_score'].mean():.4f}")
        
        print(f"\nBy folder:")
        for folder_name in args.folders:
            folder_df = df[df['folder'] == folder_name]
            if len(folder_df) > 0:
                print(f"  {folder_name}: {len(folder_df)} values, mean: {folder_df['vina_score'].mean():.4f}")


if __name__ == "__main__":
    main()

