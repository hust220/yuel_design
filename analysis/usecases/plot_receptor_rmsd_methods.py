#!/usr/bin/env python
"""
Compare receptor RMSD distributions across YuelDesign, DiffSBDD, and PMDM.
Requires:
 - protein_rmsd_pymol.csv (YuelDesign vs pocket)
 - baseline_vs_pocket_rmsd_pymol.csv (DiffSBDD/PMDM vs pocket)
Generates a combined violin+swarm plot per protein ID.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(yuel_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    if not yuel_csv.exists():
        raise FileNotFoundError(f"Missing YuelDesign CSV: {yuel_csv}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_csv}")

    yuel_df = pd.read_csv(yuel_csv)
    if "protein_id" not in yuel_df.columns or "rmsd" not in yuel_df.columns:
        raise ValueError("YuelDesign CSV must contain 'protein_id' and 'rmsd'")
    yuel_df = yuel_df.assign(method="yuel_design")

    base_df = pd.read_csv(baseline_csv)
    if not {"protein_id", "rmsd", "method"}.issubset(base_df.columns):
        raise ValueError("Baseline CSV must contain 'protein_id', 'rmsd', 'method'")

    combined = pd.concat([yuel_df[["protein_id", "rmsd", "method"]], base_df[["protein_id", "rmsd", "method"]]], ignore_index=True)
    combined = combined.dropna(subset=["protein_id", "rmsd", "method"])
    return combined


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    palette = {
        "yuel_design": "#8e7fb8",
        "diffsbdd": "#a2c9ae",
        "pmdm": "#e6b8a2",
    }
    protein_ids = sorted(df["protein_id"].unique())

    for protein_id in protein_ids:
        subset = df[df["protein_id"] == protein_id]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.boxplot(
            data=subset,
            x="method",
            y="rmsd",
            hue="method",
            palette=palette,
            ax=ax,
            width=0.5,
            fliersize=0,
            boxprops=dict(edgecolor="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        leg = ax.get_legend()
        if leg:
            leg.remove()
        sns.swarmplot(
            data=subset,
            x="method",
            y="rmsd",
            hue="method",
            palette=palette,
            dodge=False,
            size=2,
            ax=ax,
        )
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.set_xlabel("")
        ax.set_ylabel("PyMOL RMSD (Å)", fontsize=11)
        ax.set_title(protein_id.upper(), fontsize=12)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_xticklabels([tick.get_text().replace("_", "\n") for tick in ax.get_xticklabels()])
        plt.tight_layout()

        output_file = output_dir / f"receptor_rmsd_{protein_id}.svg"
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot receptor RMSD comparisons for YuelDesign, DiffSBDD, PMDM.")
    parser.add_argument("--yuel-csv", default="protein_rmsd_pymol.csv", help="YuelDesign vs pocket CSV path")
    parser.add_argument("--baseline-csv", default="baseline_vs_pocket_rmsd_pymol.csv", help="Baseline vs pocket CSV path")
    parser.add_argument("--output-dir", "-o", default=".", help="Directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.yuel_csv), Path(args.baseline_csv))
    plot_comparison(df, output_dir)


if __name__ == "__main__":
    main()

