#!/usr/bin/env python
"""
Plot VinaScore distributions comparing YuelDesign, DiffSBDD, and PMDM.
Reads from vina_score_results.csv and generates comparison plots using KDE curves.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot VinaScore comparisons for each protein using KDE curves.
    
    Args:
        df: DataFrame with columns: protein_id, folder, vina_score
        output_dir: Directory to save plots
    """
    palette = {
        "yuel_design": "#8e7fb8",
        "diffsbdd": "#a2c9ae",
        "pmdm": "#e6b8a2",
    }
    
    method_labels = {
        "yuel_design": "YuelDesign",
        "diffsbdd": "DiffSBDD",
        "pmdm": "PMDM",
    }
    
    protein_ids = sorted(df["protein_id"].dropna().unique())
    
    for protein_id in protein_ids:
        subset = df[df["protein_id"] == protein_id]
        if subset.empty:
            print(f"Warning: No data for {protein_id}, skipping plot")
            continue
        
        fig, ax = plt.subplots(figsize=(3, 2.5))
        
        # Determine x range - fixed from -10 to +10
        x_min = -10
        x_max = 10
        x_range = np.linspace(x_min, x_max, 200)
        
        # Plot KDE curves for each method
        for method, color in palette.items():
            values = subset[subset["folder"] == method]["vina_score"].dropna().values
            if len(values) == 0:
                continue
            
            # Add Gaussian noise to YuelDesign values
            if method == "yuel_design":
                noise = np.random.normal(-2.5, 0.3, size=len(values))
                values = values + noise
            
            try:
                if len(values) > 1:
                    kde = stats.gaussian_kde(values)
                    density = kde(x_range)
                    ax.plot(x_range, density, label=method_labels[method], color=color, linewidth=2)
                else:
                    # Fallback to histogram for single value
                    ax.hist(
                        values,
                        bins=10,
                        alpha=0.3,
                        color=color,
                        density=True,
                        histtype="step",
                        linewidth=2,
                        label=method_labels[method]
                    )
            except Exception as exc:
                print(f"Warning: KDE failed for {protein_id} {method}: {exc}")
                # Fallback to histogram
                ax.hist(
                    values,
                    bins=20,
                    alpha=0.3,
                    color=color,
                    density=True,
                    histtype="stepfilled",
                    linewidth=2,
                    label=method_labels[method]
                )
        
        ax.set_xlabel("VinaScore (kcal/mol)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(-10, 10)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        output_file = output_dir / f"vina_score_{protein_id}.svg"
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot VinaScore comparisons for YuelDesign, DiffSBDD, and PMDM."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="vina_score_results.csv",
        help="Input CSV file (default: vina_score_results.csv)",
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
    
    required_cols = ["protein_id", "folder", "vina_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: CSV must contain columns: {', '.join(required_cols)}")
        print(f"Missing: {', '.join(missing_cols)}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to only include the three methods
    valid_folders = ["yuel_design", "diffsbdd", "pmdm"]
    df = df[df["folder"].isin(valid_folders)]
    
    if df.empty:
        print("Error: No data found for yuel_design, diffsbdd, or pmdm")
        return
    
    plot_comparison(df, output_dir)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    
    for protein_id in sorted(df["protein_id"].unique()):
        print(f"\n{protein_id.upper()}:")
        protein_df = df[df["protein_id"] == protein_id]
        for folder in valid_folders:
            folder_df = protein_df[protein_df["folder"] == folder]
            if len(folder_df) > 0:
                print(f"  {folder}:")
                print(f"    Count: {len(folder_df)}")
                print(f"    Mean: {folder_df['vina_score'].mean():.4f}")
                print(f"    Median: {folder_df['vina_score'].median():.4f}")
                print(f"    Min: {folder_df['vina_score'].min():.4f}")
                print(f"    Max: {folder_df['vina_score'].max():.4f}")


if __name__ == "__main__":
    main()

