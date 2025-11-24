#!/usr/bin/env python
"""
Compute RMSD between YuelDesign receptor PDBs and their corresponding docked
structures using PyMOL alignment. Ligand residues (resn LIG) are removed before
alignment so only protein atoms contribute to the RMSD reported by PyMOL.
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
            for i in range(1, 11):
                design_indices = list(range(1, 21))
                for j in tqdm(design_indices, desc=f"{protein_id} batch {i}", unit="design"):
                    base_name = f"{protein_id}_yueldesign_{i}_{j}"
                    design_pdb = folder / (base_name.replace("yueldesign", "yuel_design") + ".pdb")
                    if not design_pdb.exists():
                        design_pdb = folder / f"{base_name}.pdb"

                    docking_pdb = folder / f"{base_name}_lig_dock.pdb"
                    if not design_pdb.exists():
                        print(f"    Warning: design PDB missing: {design_pdb}")
                        continue
                    if not docking_pdb.exists():
                        print(f"    Warning: docking PDB missing: {docking_pdb}")
                        continue

                    cmd.reinitialize()
                    cmd.load(str(docking_pdb), "dock")
                    cmd.remove("dock and resn LIG")
                    cmd.load(str(design_pdb), "design")
                    cmd.remove("design and resn LIG")

                    try:
                        rmsd, *_ = cmd.align("design and polymer.protein", "dock and polymer.protein", cycles=0)
                    except Exception as exc:
                        print(f"    Warning: PyMOL align failed for {design_pdb}: {exc}")
                        continue

                    if rmsd is not None:
                        results.append({
                            "protein_id": protein_id,
                            "design_name": base_name,
                            "design_file": design_pdb.name,
                            "docking_file": docking_pdb.name,
                            "rmsd": rmsd,
                        })


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSD between designed PDBs and docking PDBs via PyMOL."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="design_vs_docking_rmsd_pymol.csv",
        help="Output CSV path (default: design_vs_docking_rmsd_pymol.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    results = []

    print("Processing YuelDesign vs docking RMSDs with PyMOL ...")
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

