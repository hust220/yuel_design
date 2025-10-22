#%%
from rdkit import Chem
from rdkit.Chem import QED
import networkx as nx
import sys
sys.path.append('../..')
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

def is_valid(mol):
    # mol.RemoveAllConformers()
    # AllChem.Compute2DCoords(mol)
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def is_connected(rdkit_mol):
    G = nx.Graph()
    
    for atom in rdkit_mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in rdkit_mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    return nx.is_connected(G)

def calculate_qed(mol):
    try:
        qed_score = QED.default(mol)
        return qed_score
    except Exception as e:
        return None

def calculate_sas(mol):
    try:
        sas_score = sascorer.calculateScore(mol)
        return sas_score
    except Exception as e:
        return None

def calculate_lipinski(mol):
    try:
        passes_ro5 = all([
            Descriptors.MolWt(mol) <= 500,
            Descriptors.MolLogP(mol) <= 5,
            Lipinski.NumHDonors(mol) <= 5,
            Lipinski.NumHAcceptors(mol) <= 10
        ])
        return passes_ro5
    except Exception as e:
        return None

def get_existing_metrics(molecule_id):
    """Get existing metrics for a molecule from the database."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT bond_type 
                FROM molecule_metrics 
                WHERE molecule_id = %s
            """, (molecule_id,))
            return {row[0] for row in cur.fetchall()}

def get_molecules_from_db(limit=None):
    """Fetch molecules from database that don't have all metrics yet."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            # Get all molecule IDs
            cur.execute("SELECT id FROM molecules")
            all_ids = [row[0] for row in cur.fetchall()]
            
            # Get all existing metrics in one query
            cur.execute("""
                SELECT molecule_id, bond_type 
                FROM molecule_metrics 
                WHERE molecule_id = ANY(%s)
            """, (all_ids,))
            
            # Create a dictionary of existing metrics for each molecule
            existing_metrics = {}
            for mol_id, bond_type in cur.fetchall():
                if mol_id not in existing_metrics:
                    existing_metrics[mol_id] = set()
                existing_metrics[mol_id].add(bond_type)
            
            # Filter out molecules that already have all metrics
            needed_ids = []
            for mol_id in all_ids:
                if mol_id not in existing_metrics or len(existing_metrics[mol_id]) < 5:  # We expect 5 types of metrics
                    needed_ids.append(mol_id)
                    if limit is not None and len(needed_ids) >= limit:
                        break
            
            if limit is not None:
                return [(id,) for id in needed_ids[:limit]]
            return [(id,) for id in needed_ids]

def convert_sdf_to_mol(sdf_bytes):
    """Convert SDF bytes to RDKit molecule."""
    return next(Chem.ForwardSDMolSupplier(BytesIO(sdf_bytes), sanitize=False, strictParsing=False))

def replace_bonds_with_single(mol):
    """Replace all bonds with single bonds."""
    rwmol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        rwmol.RemoveBond(begin_idx, end_idx)
        rwmol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
    return rwmol.GetMol()

def get_sdf_bytes(mol_id, column='sdf'):
    """Get SDF bytes for a molecule from the database."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {column} FROM molecules WHERE id = %s", (mol_id,))
            sdf_bytes = cur.fetchone()[0]
    return sdf_bytes.tobytes()

def recalculate_bonds(mol):
    """Recalculate bonds using RDKit's DetermineBonds."""
    # Create a new molecule with just the atoms and coordinates
    try:
        new_mol = Chem.RWMol()
        
        # Add atoms
        for atom in mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_mol.AddAtom(new_atom)
        
        # Add coordinates
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        new_mol.AddConformer(new_conf)
        
        # Determine bonds
        rdDetermineBonds.DetermineBonds(new_mol, charge=0)
    except Exception as e:
        # print(f"Error recalculating bonds: {e}")
        return None
    
    return new_mol.GetMol()

def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def recalculate_bonds_with_distances(mol):
    """Recalculate bonds using distance-based bond order determination."""
    try:
        new_mol = Chem.RWMol()
        
        # Add atoms
        for atom in mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_mol.AddAtom(new_atom)
        
        # Add coordinates
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        new_mol.AddConformer(new_conf)
        
        # Calculate bonds based on distances
        for i in range(mol.GetNumAtoms()):
            atom1 = mol.GetAtomWithIdx(i)
            pos1 = conf.GetAtomPosition(i)
            for j in range(i + 1, mol.GetNumAtoms()):
                atom2 = mol.GetAtomWithIdx(j)
                pos2 = conf.GetAtomPosition(j)
                
                # Calculate distance
                distance = pos1.Distance(pos2)
                
                # Get bond order
                bond_order = get_bond_order(
                    atom1.GetSymbol(),
                    atom2.GetSymbol(),
                    distance
                )
                
                # Add bond if order > 0
                if bond_order > 0:
                    new_mol.AddBond(i, j, Chem.BondType(bond_order))
        
        return new_mol.GetMol()
    except Exception as e:
        # print(f"Error recalculating bonds with distances: {e}")
        return None


def create_metrics_table():
    """Create a table to store molecule metrics if it doesn't exist."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bond_reconstruction (
                    id SERIAL PRIMARY KEY,
                    molecule_id INTEGER,
                    bond_type TEXT NOT NULL,
                    is_valid BOOLEAN,
                    qed FLOAT,
                    sas FLOAT,
                    lipinski BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(molecule_id, bond_type)
                )
            """)
            conn.commit()

def save_metrics_to_db(molecule_id, bond_type, metrics):
    """Save molecule metrics to the database."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO molecule_metrics 
                    (molecule_id, bond_type, is_valid, qed, sas, lipinski)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (molecule_id, bond_type) 
                DO UPDATE SET
                    is_valid = EXCLUDED.is_valid,
                    qed = EXCLUDED.qed,
                    sas = EXCLUDED.sas,
                    lipinski = EXCLUDED.lipinski,
                    created_at = CURRENT_TIMESTAMP
            """, (
                molecule_id,
                bond_type,
                metrics['valid'],
                metrics['qed'],
                metrics['sas'],
                metrics['lipinski']
            ))
            conn.commit()

def calculate_metrics(mol):
    if mol is None:
        return {'valid': False, 'qed': None, 'sas': None, 'lipinski': None}
    valid = is_valid(mol)
    qed = calculate_qed(mol)
    sas = calculate_sas(mol)
    lipinski = calculate_lipinski(mol)
    return {
        'valid': valid,
        'qed': qed,
        'sas': sas,
        'lipinski': lipinski
    }

def process_molecule_batch(mol_ids, create_table=True):
    """Process a batch of molecules."""
    if create_table:
        create_metrics_table()
    
    # Get all SDF data for the batch in one query
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, sdf, sdf2 
                FROM molecules 
                WHERE id = ANY(%s)
            """, (mol_ids,))
            batch_data = cur.fetchall()
    
    # Prepare batch data for database update
    batch_metrics = []
    
    # Process each molecule in the batch
    for mol_id, sdf_bytes, sdf2_bytes in batch_data:
        # Convert SDF data to bytes
        sdf_bytes = sdf_bytes.tobytes()
        sdf2_bytes = sdf2_bytes.tobytes()
        
        mol = convert_sdf_to_mol(sdf_bytes)
        mol2 = convert_sdf_to_mol(sdf2_bytes)
        if mol is None:
            continue
        
        single_bond_mol = replace_bonds_with_single(mol)
        recalculated_mol = recalculate_bonds(mol)
        distance_mol = recalculate_bonds_with_distances(mol)

        # Calculate metrics for all variants
        yuelbond1_metrics = calculate_metrics(mol)
        yuelbond2_metrics = calculate_metrics(mol2)
        single_bond_metrics = calculate_metrics(single_bond_mol)
        recalculated_metrics = calculate_metrics(recalculated_mol)
        distance_metrics = calculate_metrics(distance_mol)
        
        # Add all metrics to batch
        for bond_type, metrics in [
            ('yuelbond1', yuelbond1_metrics),
            ('yuelbond2', yuelbond2_metrics),
            ('single_bond', single_bond_metrics),
            ('recalculated_bond', recalculated_metrics),
            ('distance_bond', distance_metrics)
        ]:
            batch_metrics.append((
                mol_id,
                bond_type,
                metrics['valid'],
                metrics['qed'],
                metrics['sas'],
                metrics['lipinski']
            ))
    
    # Save all metrics in one transaction
    if batch_metrics:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO bond_reconstruction 
                        (molecule_id, bond_type, is_valid, qed, sas, lipinski)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (molecule_id, bond_type) 
                    DO UPDATE SET
                        is_valid = EXCLUDED.is_valid,
                        qed = EXCLUDED.qed,
                        sas = EXCLUDED.sas,
                        lipinski = EXCLUDED.lipinski,
                        created_at = CURRENT_TIMESTAMP
                """, batch_metrics)
            conn.commit()

# Common group order, labels, and colors for all plots
GROUP_ORDER = ['distance_bond', 'recalculated_bond', 'single_bond', 'yuelbond1', 'yuelbond2']
GROUP_LABELS = ['Distance', 'RDKit', 'Single Bond', 'YuelBond', 'YuelBond*']
GROUP_COLORS = ['grey', '#e6b8a2', 'black', '#8e7fb8', '#a2c9ae']

def remove_legend(ax):
    """Remove the legend from a matplotlib Axes if it exists."""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

def plot_qed_distribution():
    with db_connection() as conn:
        query = """
            SELECT bond_type, qed
            FROM bond_reconstruction
            WHERE qed IS NOT NULL and is_valid = True
        """
        df_qed = pd.read_sql_query(query, conn)
    df_qed = df_qed[df_qed['bond_type'].isin(GROUP_ORDER)]
    df_qed['bond_type'] = pd.Categorical(df_qed['bond_type'], categories=GROUP_ORDER, ordered=True)
    plt.figure(figsize=(4, 3))
    sns.kdeplot(data=df_qed, x='qed', hue='bond_type', hue_order=GROUP_ORDER, palette=GROUP_COLORS, common_norm=True)
    plt.xlabel('QED Score')
    plt.ylabel('Density')
    ax = plt.gca()
    remove_legend(ax)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('bond_effects_plots/qed_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print("\nQED Statistics by Bond Type:")
    print(df_qed.groupby('bond_type')['qed'].describe())

def plot_sas_distribution():
    with db_connection() as conn:
        query = """
            SELECT bond_type, sas
            FROM bond_reconstruction
            WHERE sas IS NOT NULL and is_valid = True
        """
        df_sas = pd.read_sql_query(query, conn)
    df_sas = df_sas[df_sas['bond_type'].isin(GROUP_ORDER)]
    df_sas['bond_type'] = pd.Categorical(df_sas['bond_type'], categories=GROUP_ORDER, ordered=True)
    plt.figure(figsize=(4, 3))
    sns.kdeplot(data=df_sas, x='sas', hue='bond_type', hue_order=GROUP_ORDER, palette=GROUP_COLORS, common_norm=True)
    plt.xlabel('SAS Score')
    plt.ylabel('Density')
    ax = plt.gca()
    remove_legend(ax)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('bond_effects_plots/sas_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print("\nSAS Statistics by Bond Type:")
    print(df_sas.groupby('bond_type')['sas'].describe())

def plot_lipinski_distribution():
    with db_connection() as conn:
        query = """
            SELECT bond_type, lipinski
            FROM bond_reconstruction
            WHERE lipinski IS NOT NULL and is_valid = True
        """
        df_lipinski = pd.read_sql_query(query, conn)
    df_lipinski = df_lipinski[df_lipinski['bond_type'].isin(GROUP_ORDER)]
    df_lipinski['bond_type'] = pd.Categorical(df_lipinski['bond_type'], categories=GROUP_ORDER, ordered=True)
    lipinski_counts = df_lipinski.groupby(['bond_type', 'lipinski']).size().unstack().reindex(index=GROUP_ORDER).fillna(0)
    group_totals = lipinski_counts.sum(axis=1)
    max_total = group_totals.max() if len(group_totals) > 0 else 1
    lipinski_frac = lipinski_counts / max_total
    x = np.arange(len(GROUP_ORDER))
    width = 0.4
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    bars_true = ax.bar(x - width/2, lipinski_frac[True], width,
                       label='Passes',
                       color=GROUP_COLORS,
                       edgecolor='black',
                       linewidth=1.5)
    bars_false = ax.bar(x + width/2, lipinski_frac[False], width,
                        label='Not Passes',
                        color='none',
                        edgecolor=GROUP_COLORS,
                        linewidth=1.5,
                        hatch=None)
    # Show fractions above bars
    for i, rect in enumerate(bars_true):
        frac = lipinski_frac.iloc[i].get(True, 0)
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01, f"{frac:.2f}",
                ha='center', va='bottom', fontsize=8, color='black')
    for i, rect in enumerate(bars_false):
        frac = lipinski_frac.iloc[i].get(False, 0)
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01, f"{frac:.2f}",
                ha='center', va='bottom', fontsize=8, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, rotation=45)
    ax.set_ylabel('Fraction')
    ax.set_xlabel('')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('bond_effects_plots/lipinski_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print("\nLipinski Rule Compliance:")
    print(lipinski_counts)

def plot_validity_distribution():
    with db_connection() as conn:
        query = """
            SELECT bond_type, is_valid
            FROM bond_reconstruction
            WHERE is_valid IS NOT NULL
        """
        df_validity = pd.read_sql_query(query, conn)
    df_validity = df_validity[df_validity['bond_type'].isin(GROUP_ORDER)]
    df_validity['bond_type'] = pd.Categorical(df_validity['bond_type'], categories=GROUP_ORDER, ordered=True)
    validity_counts = df_validity.groupby(['bond_type', 'is_valid']).size().unstack().reindex(index=GROUP_ORDER).fillna(0)
    group_totals_valid = validity_counts.sum(axis=1)
    max_total_valid = group_totals_valid.max() if len(group_totals_valid) > 0 else 1
    validity_frac = validity_counts / max_total_valid
    x = np.arange(len(GROUP_ORDER))
    width = 0.4
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    bars_true = ax.bar(x - width/2, validity_frac[True], width,
                       label='Valid',
                       color=GROUP_COLORS,
                       edgecolor='black',
                       linewidth=1.5)
    bars_false = ax.bar(x + width/2, validity_frac[False], width,
                        label='Not Valid',
                        color='none',
                        edgecolor=GROUP_COLORS,
                        linewidth=1.5,
                        hatch=None)
    # Show fractions above bars
    for i, rect in enumerate(bars_true):
        frac = validity_frac.iloc[i].get(True, 0)
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01, f"{frac:.2f}",
                ha='center', va='bottom', fontsize=8, color='black')
    for i, rect in enumerate(bars_false):
        frac = validity_frac.iloc[i].get(False, 0)
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01, f"{frac:.2f}",
                ha='center', va='bottom', fontsize=8, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, rotation=45)
    ax.set_ylabel('Fraction')
    ax.set_xlabel('')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig('bond_effects_plots/validity_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print("\nValidity Statistics:")
    print(validity_counts)

def analyze_metric_distributions():
    plot_qed_distribution()
    plot_sas_distribution()
    plot_lipinski_distribution()
    plot_validity_distribution()

def run_calculation():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze molecule bonds and calculate metrics')
    parser.add_argument('--workers', type=int, default=11,
                      help='Number of worker processes (default: 1)')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of molecules to process (default: None)')
    parser.add_argument('--batch', type=int, default=1,
                      help='Number of molecules to process in each batch (default: 10)')
    args = parser.parse_args()
    
    # Create metrics table if it doesn't exist
    create_metrics_table()
    
    # Get molecules from database that need metrics
    ids = get_molecules_from_db(limit=args.limit)
    
    if not ids:
        print("No molecules need processing.")
        return
    
    # Prepare batches
    mol_ids = [mol_id[0] for mol_id in ids]
    batches = [mol_ids[i:i + args.batch] for i in range(0, len(mol_ids), args.batch)]
    
    # Set up parallel processing
    pool = mp.Pool(processes=args.workers)
    
    # Process batches in parallel with progress bar
    process_func = partial(process_molecule_batch, create_table=False)
    list(tqdm(
        pool.imap(process_func, batches),
        total=len(batches),
        desc="Processing batches",
        unit="batch"
    ))
    
    # Clean up
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Set start method for multiprocessing
    # mp.set_start_method('spawn', force=True)
    # run_calculation()
    analyze_metric_distributions()

# %%
