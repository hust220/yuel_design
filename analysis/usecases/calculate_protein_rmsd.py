#!/usr/bin/env python
"""
Compute backbone RMSD between designed receptors and reference pocket structures.
For each protein target, the reference CA centroid is subtracted from both the
reference coordinates and the designed coordinates before RMSD is calculated.
No alignment is applied.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdb_utils import Structure


def load_structure_coords(pdb_path: Path, center: np.ndarray = None, exclude_residue="LIG"):
    if not pdb_path.exists():
        print(f"    Warning: PDB missing: {pdb_path}")
        return None

    structure = Structure()
    try:
        structure.read(str(pdb_path), skip_hetatm=False, skip_water=False)
    except FileNotFoundError:
        print(f"    Warning: Failed to parse PDB: {pdb_path}")
        return None

    coords = []
    for model in structure.models:
        for chain in model.chains:
            for residue in chain.residues:
                if residue.res_name.strip() == exclude_residue:
                    continue
                for atom in residue.atoms:
                    coords.append(atom.get_coord())

    if not coords:
        print(f"    Warning: No atoms found in {pdb_path}")
        return None

    coords = np.array(coords)
    if center is not None:
        return coords - center
    return coords


def compute_ca_centroid(pdb_path: Path):
    structure = Structure()
    try:
        structure.read(str(pdb_path), skip_hetatm=False, skip_water=False)
    except FileNotFoundError:
        print(f"    Warning: Failed to parse pocket PDB: {pdb_path}")
        return None

    ca_coords = []
    for model in structure.models:
        for chain in model.chains:
            for residue in chain.residues:
                for atom in residue.atoms:
                    if atom.atom_name.strip() == "CA":
                        ca_coords.append(atom.get_coord())

    if not ca_coords:
        print(f"    Warning: No CA atoms found in pocket: {pdb_path}")
        return None

    ca_coords = np.array(ca_coords)
    return ca_coords.mean(axis=0)


def compute_rmsd(coords_a: np.ndarray, coords_b: np.ndarray):
    if coords_a.shape != coords_b.shape:
        print(f"    Warning: Coordinate count mismatch {coords_a.shape} vs {coords_b.shape}")
        return None
    diff = coords_a - coords_b
    return float(np.sqrt((diff ** 2).mean()))


def process_yuel_design(root: Path, results: list):
    folder = root / "yuel_design"
    for protein_id in ["3jqa", "6cyb"]:
        pocket_pdb = root / f"{protein_id}_pocket_1.pdb"
        pocket_center = compute_ca_centroid(pocket_pdb)
        if pocket_center is None:
            continue

        pocket_coords = load_structure_coords(pocket_pdb, center=pocket_center)
        if pocket_coords is None:
            continue

        for i in range(1, 11):
            for j in range(1, 21):
                design_pdb = folder / f"{protein_id}_yuel_design_{i}_{j}.pdb"
                if not design_pdb.exists():
                    design_pdb = folder / f"{protein_id}_yueldesign_{i}_{j}.pdb"
                if not design_pdb.exists():
                    print(f"    Warning: Design PDB missing: {design_pdb}")
                    continue

                design_coords = load_structure_coords(design_pdb, center=pocket_center)
                if design_coords is None:
                    continue

                rmsd = compute_rmsd(pocket_coords, design_coords)
                if rmsd is not None:
                    results.append({
                        "protein_id": protein_id,
                        "design_name": f"{protein_id}_yueldesign_{i}_{j}",
                        "design_file": design_pdb.name,
                        "pocket_file": pocket_pdb.name,
                        "rmsd": rmsd,
                    })


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSD between designed receptor structures and reference pockets."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="protein_rmsd_results.csv",
        help="Output CSV filename (default: protein_rmsd_results.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    results = []

    print("Processing YuelDesign receptors ...")
    process_yuel_design(root, results)

    if not results:
        print("No RMSD values computed.")
        return

    df = pd.DataFrame(results)
    output_path = root / args.output
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} RMSD values to {output_path}")
    print(df.groupby("protein_id")["rmsd"].describe())


if __name__ == "__main__":
    main()

