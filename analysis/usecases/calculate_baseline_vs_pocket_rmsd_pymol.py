#!/usr/bin/env python
"""
Compute RMSD between baseline docking receptors (DiffSBDD/PMDM) and pocket PDBs
using PyMOL alignment. Ligand residues (resn LIG) are removed before alignment,
and PyMOL's reported RMSD is recorded.
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


def process_folder(root: Path, folder_name: str, results: list):
    folder = root / folder_name
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
                for j in tqdm(design_ids, desc=f"{folder_name} {protein_id} batch {i}", unit="design"):
                    base_name = f"{protein_id}_{folder_name}_{i}_{j}"
                    docking_pdb = folder / f"{base_name}_dock.pdb"
                    if not docking_pdb.exists():
                        docking_pdb = folder / f"{base_name}_dock_unk.pdb"
                    if not docking_pdb.exists():
                        docking_pdb = folder / f"{base_name}_lig_dock.pdb"

                    if not docking_pdb.exists():
                        print(f"    Warning: docking PDB missing: {docking_pdb}")
                        continue

                    cmd.reinitialize()
                    cmd.load(str(pocket_pdb), "pocket")
                    cmd.remove("pocket and resn LIG")
                    cmd.load(str(docking_pdb), "dock")
                    cmd.remove("dock and resn LIG")

                    try:
                        rmsd, *_ = cmd.align("dock and polymer.protein", "pocket and polymer.protein", cycles=0)
                    except Exception as exc:
                        print(f"    Warning: PyMOL align failed for {docking_pdb}: {exc}")
                        continue

                    if rmsd is not None:
                        results.append({
                            "method": folder_name,
                            "protein_id": protein_id,
                            "sample_name": base_name,
                            "pocket_file": pocket_pdb.name,
                            "docking_file": docking_pdb.name,
                            "rmsd": rmsd,
                        })


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSD between baseline docking receptors and pocket PDBs using PyMOL."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="baseline_vs_pocket_rmsd_pymol.csv",
        help="Output CSV path (default: baseline_vs_pocket_rmsd_pymol.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    results = []

    print("Processing DiffSBDD vs pocket RMSDs ...")
    process_folder(root, "diffsbdd", results)

    print("Processing PMDM vs pocket RMSDs ...")
    process_folder(root, "pmdm", results)

    if not results:
        print("No RMSD values computed.")
        return

    df = pd.DataFrame(results)
    output_path = root / args.output
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} RMSD values to {output_path}")
    print(df.groupby(["method", "protein_id"])["rmsd"].describe())


if __name__ == "__main__":
    main()

