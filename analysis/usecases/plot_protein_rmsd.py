#!/usr/bin/env python
"""
Plot protein RMSD distributions from protein_rmsd_pymol.csv.
Generates separate KDE plots per protein ID.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rmsd(df: pd.DataFrame, output_dir: Path):
    protein_ids = sorted(df["protein_id"].dropna().unique())
    if not protein_ids:
        print("No protein IDs found in CSV.")
        return

    for protein_id in protein_ids:
        protein_df = df[df["protein_id"] == protein_id]
        values = protein_df["rmsd"].dropna()
        if values.empty:
            print(f"Warning: no RMSD values for {protein_id}, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(2.5, 3))
        sns.boxplot(
            y=values,
            color="white",
            ax=ax,
            width=0.4,
            fliersize=0,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        sns.swarmplot(y=values, color="#8e7fb8", size=3, ax=ax)
        ax.set_ylabel("PyMOL RMSD (Å)", fontsize=12)
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        output_file = output_dir / f"protein_rmsd_{protein_id}.svg"
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot protein RMSD distributions from CSV.")
    parser.add_argument(
        "--input",
        "-i",
        default="protein_rmsd_pymol.csv",
        help="Input CSV file (default: protein_rmsd_pymol.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to save plots (default: current directory)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "protein_id" not in df.columns or "rmsd" not in df.columns:
        print("Error: CSV must contain 'protein_id' and 'rmsd' columns.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_rmsd(df, output_dir)


if __name__ == "__main__":
    main()

