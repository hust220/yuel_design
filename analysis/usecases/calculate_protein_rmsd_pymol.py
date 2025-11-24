#!/usr/bin/env python
"""
Compute protein RMSD between reference pocket structures and YuelDesign receptors
using PyMOL. Ligand residues (resn LIG) are excluded, objects are aligned with
PyMOL's built-in `align`, and the reported RMSD is taken directly from PyMOL.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import pymol2
except ImportError:
    print("Error: pymol-open-source is required. Install via `pip install pymol-open-source`.")
    sys.exit(1)


def process_yuel_design(root: Path, results: list):
    folder = root / "yuel_design"
    protein_ids = ["3jqa", "6cyb"]

    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        for protein_id in protein_ids:
            pocket_pdb = root / f"{protein_id}_pocket_1.pdb"
            if not pocket_pdb.exists():
                print(f"    Warning: pocket PDB missing: {pocket_pdb}")
                continue

            for i in range(1, 11):
                design_ids = list(range(1, 21))
                for j in tqdm(design_ids, desc=f"{protein_id} batch {i}", unit="design"):
                    design_pdb = folder / f"{protein_id}_yuel_design_{i}_{j}.pdb"
                    if not design_pdb.exists():
                        design_pdb = folder / f"{protein_id}_yueldesign_{i}_{j}.pdb"
                    if not design_pdb.exists():
                        print(f"    Warning: design PDB missing: {design_pdb}")
                        continue

                    cmd.reinitialize()
                    cmd.load(str(pocket_pdb), "ref")
                    cmd.remove("ref and resn LIG")
                    cmd.load(str(design_pdb), "design")
                    cmd.remove("design and resn LIG")

                    try:
                        rmsd, *_ = cmd.align("design and polymer.protein", "ref and polymer.protein", cycles=0)
                    except Exception as exc:
                        print(f"    Warning: PyMOL align failed for {design_pdb}: {exc}")
                        continue

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
        description="Compute receptor RMSD via PyMOL (ligands excluded, no alignment)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="protein_rmsd_pymol.csv",
        help="Output CSV path (default: protein_rmsd_pymol.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    results = []

    print("Processing YuelDesign receptors with PyMOL ...")
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

