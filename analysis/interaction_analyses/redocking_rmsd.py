#%%
import psycopg2
from psycopg2 import sql
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from io import BytesIO
from rdkit.RDLogger import DisableLog
import sys
sys.path.append('../')
from db_utils import db_connection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

DisableLog('rdApp.*')

BATCH_SIZE = 10  # Number of molecules to process in each batch
NUM_PROCESSES = 12

def read_pdb_from_bytes(pdb_bytes):
    """Read PDB file from bytes and extract HETATM coordinates for UNK residue only."""
    if pdb_bytes is None:
        return None
    
    pdb_str = pdb_bytes.decode('utf-8')
    coordinates = []
    
    for line in pdb_str.split('\n'):
        if line.startswith('HETATM') and line[17:20].strip() == 'UNK':
            # PDB format: HETATM 1  C   UNK A   1      12.345  23.456  34.567
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coordinates.append([x, y, z])
            except (ValueError, IndexError):
                continue
    
    return np.array(coordinates) if coordinates else None

def read_sdf_from_bytes(sdf_bytes):
    """Read SDF file from bytes and extract atom coordinates."""
    if sdf_bytes is None:
        return None
    
    sdf_str = sdf_bytes.decode('utf-8')
    lines = sdf_str.split('\n')
    
    # Find the number of atoms (line 4 of SDF)
    try:
        num_atoms = int(lines[3][:3])
    except (ValueError, IndexError):
        return None
    
    coordinates = []
    # Coordinates start from line 4
    for i in range(4, 4 + num_atoms):
        try:
            parts = lines[i].split()
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            coordinates.append([x, y, z])
        except (ValueError, IndexError):
            continue
    
    return np.array(coordinates) if coordinates else None

def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates."""
    if coords1 is None or coords2 is None:
        return None
    
    if len(coords1) != len(coords2):
        return None
    
    # Calculate squared differences
    diff = coords1 - coords2
    squared_diff = np.sum(diff * diff, axis=1)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(squared_diff))
    return rmsd

def process_molecule(molecule_id, pdb_bytes, sdf_bytes):
    """Process a single molecule and calculate RMSD."""
    pdb_coords = read_pdb_from_bytes(pdb_bytes)
    sdf_coords = read_sdf_from_bytes(sdf_bytes)
    
    if pdb_coords is None or sdf_coords is None:
        return None
    
    return calculate_rmsd(pdb_coords, sdf_coords)

def process_batch(batch_ids):
    """Process a batch of molecule IDs and save RMSD results directly to database in bulk."""
    results = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch data for this batch of IDs
            cursor.execute("""
                SELECT mr.molecule_id, mr.best_pose, m.sdf
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                WHERE mr.molecule_id = ANY(%s)
                AND mr.best_pose IS NOT NULL AND m.sdf IS NOT NULL
            """, (batch_ids,))
            
            # Process molecules and collect results
            for molecule_id, pdb_bytes, sdf_bytes in cursor:
                rmsd = process_molecule(molecule_id, pdb_bytes.tobytes(), sdf_bytes.tobytes())
                # print(f"RMSD for molecule {molecule_id}: {rmsd}")
                if rmsd is not None:
                    results.append((rmsd, molecule_id))
            
            # Save results for this batch in bulk
            if results:
                cursor.executemany("""
                    UPDATE medusadock_results 
                    SET rmsd = %s 
                    WHERE molecule_id = %s
                """, results)
                conn.commit()
    
    return len(results)  # Return count of processed molecules

def main():
    """Main function to process all molecules and calculate RMSDs."""
    # Ensure rmsd column exists
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name='medusadock_results' 
                        AND column_name='rmsd'
                    ) THEN
                        ALTER TABLE medusadock_results ADD COLUMN rmsd double precision;
                    END IF;
                END $$;
            """)
            conn.commit()
    
    # select all ids that need to be processed
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT mr.molecule_id
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                WHERE mr.best_pose IS NOT NULL AND m.sdf IS NOT NULL AND mr.rmsd IS NULL
            """)
            all_ids = [row[0] for row in cursor.fetchall()]
    
    # divide ids to batches
    batches = []
    for i in range(0, len(all_ids), BATCH_SIZE):
        batches.append(all_ids[i:i + BATCH_SIZE])
    
    # process batches in parallel
    with Pool(NUM_PROCESSES) as pool:
        processed_counts = list(tqdm(pool.imap(process_batch, batches), 
                                   total=len(batches), 
                                   desc="Processing batches"))
    
    # Print summary
    total_processed = sum(processed_counts)
    print(f"\nProcessed {total_processed} molecules")

def plot_rmsd_distribution():
    """Plot the distribution of RMSD values."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT rmsd 
                FROM medusadock_results 
                WHERE rmsd IS NOT NULL
            """)
            rmsds = [row[0] for row in cursor.fetchall()]
    
    plt.figure(figsize=(3, 2.5))
    sns.kdeplot(data=rmsds, color='#8E7FB8', fill=True, alpha=0.7, edgecolor='black')
    plt.xlabel('RMSD', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)  # Set x-axis range to 0-20
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('docking_plots/rmsd_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rmsd_vs_size(interval=1):
    """Plot RMSD values against molecule size with three distinct regions: 25-75% in color, 0-25% and 75-100% in a different color, and outliers in gray."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.size, mr.rmsd
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                WHERE mr.rmsd IS NOT NULL
            """)
            data = cursor.fetchall()
    
    sizes = np.array([row[0] for row in data])
    rmsds = np.array([row[1] for row in data])
    
    # Bin by size with adjustable interval
    min_size = int(np.min(sizes))
    max_size = int(np.max(sizes))
    bin_edges = np.arange(min_size, max_size + interval, interval)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Prepare lists for all regions
    outliers_low = []
    q1s = []
    medians = []
    q3s = []
    outliers_high = []
    mins = []
    maxs = []
    
    for i in range(len(bin_edges) - 1):
        bin_mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i+1])
        bin_rmsds = rmsds[bin_mask]
        if len(bin_rmsds) > 0:
            q1 = np.percentile(bin_rmsds, 25)
            q3 = np.percentile(bin_rmsds, 75)
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr
            median = np.percentile(bin_rmsds, 50)
            min_val = np.min(bin_rmsds)
            max_val = np.max(bin_rmsds)
        else:
            q1 = q3 = lower_whisker = upper_whisker = median = min_val = max_val = np.nan
        
        outliers_low.append(lower_whisker)
        q1s.append(q1)
        medians.append(median)
        q3s.append(q3)
        outliers_high.append(upper_whisker)
        mins.append(min_val)
        maxs.append(max_val)
    
    plt.figure(figsize=(3, 2.5))
    
    # Plot outlier regions in gray
    plt.fill_between(bin_centers, mins, outliers_low, color='gray', alpha=0.4, label='Outliers')
    plt.fill_between(bin_centers, outliers_high, maxs, color='gray', alpha=0.4)
    
    # Plot 0-25% and 75-100% regions in light color
    plt.fill_between(bin_centers, outliers_low, q1s, color='#A2C9AE', alpha=0.7, label='0-25% & 75-100%')
    plt.fill_between(bin_centers, q3s, outliers_high, color='#A2C9AE', alpha=0.7)
    
    # Plot 25-75% region in darker color
    plt.fill_between(bin_centers, q1s, q3s, color='#8E7FB8', alpha=0.7, label='25-75%')
    
    # Plot the median
    plt.plot(bin_centers, medians, color='black', lw=1.5, label='Median')
    
    plt.xlabel('Molecule Size', fontsize=10)
    plt.ylabel('RMSD', fontsize=10)
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend(fontsize=8, loc='upper right')
    
    # Save and show the plot
    plt.savefig('docking_plots/rmsd_vs_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_rmsd():
    """Analyze RMSD values and create plots."""
    # Create docking_plots directory if it doesn't exist
    os.makedirs('docking_plots', exist_ok=True)
    
    print("Plotting RMSD distribution...")
    plot_rmsd_distribution()
    
    print("\nPlotting RMSD vs size...")
    plot_rmsd_vs_size()
    
    # Print statistics
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    AVG(rmsd) as avg_rmsd,
                    MIN(rmsd) as min_rmsd,
                    MAX(rmsd) as max_rmsd,
                    COUNT(rmsd) as count
                FROM medusadock_results 
                WHERE rmsd IS NOT NULL
            """)
            avg_rmsd, min_rmsd, max_rmsd, count = cursor.fetchone()
            
            print(f"\nRMSD Statistics:")
            print(f"Number of molecules: {count}")
            print(f"Average RMSD: {avg_rmsd:.3f}")
            print(f"Min RMSD: {min_rmsd:.3f}")
            print(f"Max RMSD: {max_rmsd:.3f}")

#%%
if __name__ == "__main__":
    # main()
    analyze_rmsd()

# %%
