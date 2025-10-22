#%%
from rdkit import Chem
from rdkit.Chem import QED
import networkx as nx
import sys
sys.path.append('../')
from db_utils import db_connection
from src import sascorer, const
from rdkit.RDLogger import DisableLog
DisableLog('rdApp.*')
from io import BytesIO
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Lipinski, Descriptors
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from analysis.analyze_bond_prediction import convert_sdf_to_mol, recalculate_bonds_with_distances, recalculate_bonds, replace_bonds_with_single

def find_better_yuelbond_molecules():
    """
    Find molecules where:
    1. yuelbond2 has higher QED than yuelbond1 by at least 0.2
    2. yuelbond1 has higher QED than other bond types by at least 0.2
    Results are ordered by the QED difference (yuelbond2 - yuelbond1) in descending order
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        query = """
        WITH molecule_qed AS (
            SELECT 
                molecule_id,
                bond_type,
                qed
            FROM molecule_metrics
            WHERE bond_type IN ('yuelbond1', 'yuelbond2', 'single_bond', 'recalculated_bond', 'distance_bond')
        ),
        other_bonds_max AS (
            SELECT 
                molecule_id,
                MAX(qed) as max_other_qed
            FROM molecule_qed
            WHERE bond_type NOT IN ('yuelbond1', 'yuelbond2')
            GROUP BY molecule_id
        ),
        molecule_comparison AS (
            SELECT 
                m1.molecule_id,
                m1.qed as yuelbond1_qed,
                m2.qed as yuelbond2_qed,
                o.max_other_qed,
                (m2.qed - m1.qed) as qed_difference
            FROM molecule_qed m1
            JOIN molecule_qed m2 ON m1.molecule_id = m2.molecule_id AND m2.bond_type = 'yuelbond2'
            JOIN other_bonds_max o ON m1.molecule_id = o.molecule_id
            WHERE m1.bond_type = 'yuelbond1'
        )
        SELECT 
            molecule_id,
            yuelbond1_qed,
            yuelbond2_qed,
            max_other_qed,
            qed_difference
        FROM molecule_comparison
        WHERE yuelbond2_qed > yuelbond1_qed + 0.2  -- yuelbond2 is at least 0.2 higher than yuelbond1
        AND yuelbond1_qed > max_other_qed + 0.2    -- yuelbond1 is at least 0.2 higher than others
        ORDER BY qed_difference DESC;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert results to a pandas DataFrame for easier analysis
        df = pd.DataFrame(results, columns=['molecule_id', 'yuelbond1_qed', 'yuelbond2_qed', 'max_other_qed', 'qed_difference'])
        
        cursor.close()
        conn.close()
        
        return df

def save_molecule_sdfs(molecule_id, mol, mol2, output_dir, qed_values=None):
    """
    Save a molecule with all its bond type variants as SDF files.
    
    Args:
        molecule_id: ID of the molecule
        mol: Original molecule (yuelbond1)
        mol2: Second molecule (yuelbond2)
        output_dir: Directory to save the SDF files
        qed_values: Dictionary containing QED values for each bond type (optional)
    """
    if mol is None:
        print("Failed to convert SDF data to molecule")
        return
    
    # Create different bond variants
    bond_types = ['yuelbond1', 'yuelbond2', 'single_bond', 'recalculated_bond', 'distance_bond']
    molecules = {
        'yuelbond1': mol,
        'yuelbond2': mol2,
        'single_bond': replace_bonds_with_single(mol),
        'recalculated_bond': recalculate_bonds(mol),
        'distance_bond': recalculate_bonds_with_distances(mol)
    }
    
    # Save all variants to SDF files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each variant
    for bond_type, mol_variant in molecules.items():
        if mol_variant is not None:
            # Create filename with QED value if available
            if qed_values and bond_type in qed_values:
                qed_str = f"_qed{qed_values[bond_type]:.3f}"
            else:
                qed_str = ""
            
            filename = f"{output_dir}/molecule_{molecule_id}_{bond_type}{qed_str}.sdf"
            Chem.MolToMolFile(mol_variant, filename, kekulize=False)
    
    print(f"\nSaved all bond variants to {output_dir}/ directory")

def save_top_molecule_sdfs(better_molecules_df, output_dir, top_k=1):
    """
    Get the top-k molecules from the better_molecules DataFrame and save their different bond type variants as SDF files.
    
    Args:
        better_molecules_df: DataFrame containing the results from find_better_yuelbond_molecules()
        output_dir: Directory to save the SDF files
        top_k: Number of top molecules to save (default: 1)
    """
    if better_molecules_df.empty:
        print("No molecules found matching the criteria")
        return
    
    # Get the top-k molecules
    top_molecules = better_molecules_df.head(top_k)
    
    for idx, top_molecule in top_molecules.iterrows():
        molecule_id = top_molecule['molecule_id']
        
        print(f"\nProcessing molecule {idx + 1}/{top_k}")
        print(f"Molecule ID: {molecule_id}")
        print(f"QED values:")
        print(f"YuelBond2: {top_molecule['yuelbond2_qed']:.3f}")
        print(f"YuelBond1: {top_molecule['yuelbond1_qed']:.3f}")
        print(f"Max other bonds: {top_molecule['max_other_qed']:.3f}")
        print(f"QED difference: {top_molecule['qed_difference']:.3f}")
        
        # Get the SDF data for this molecule
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sdf, sdf2 
                FROM molecules 
                WHERE id = %s
            """, (molecule_id,))
            sdf_data = cursor.fetchone()
            
            if not sdf_data:
                print(f"No SDF data found for molecule {molecule_id}")
                continue
            
            # Convert SDF data to molecules
            mol = convert_sdf_to_mol(sdf_data[0].tobytes())
            mol2 = convert_sdf_to_mol(sdf_data[1].tobytes())
            
            # Prepare QED values for filename
            qed_values = {
                'yuelbond1': top_molecule['yuelbond1_qed'],
                'yuelbond2': top_molecule['yuelbond2_qed'],
                'single_bond': top_molecule['max_other_qed'],  # Using max_other_qed as an approximation
                'recalculated_bond': top_molecule['max_other_qed'],  # Using max_other_qed as an approximation
                'distance_bond': top_molecule['max_other_qed']  # Using max_other_qed as an approximation
            }
            
            # Save all variants
            save_molecule_sdfs(molecule_id, mol, mol2, output_dir, qed_values)

output_dir = "best_prediction_examples"
better_molecules = find_better_yuelbond_molecules()
print(f"Found {len(better_molecules)} molecules where yuelbond2 > yuelbond1 > max(other bonds)")
print("\nTop 5 molecules with largest QED improvement from yuelbond1 to yuelbond2:")
print(better_molecules.head())

# Save the top 5 molecules' SDF files
save_top_molecule_sdfs(better_molecules, output_dir, top_k=5)

# %%
