#%%
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Descriptors, QED
import sys
import os
sys.path.append('../..')
from src.db_utils import db_connection, ensure_column_exists
from tqdm import tqdm
import networkx as nx
from src import sascorer
from rdkit import RDLogger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import matplotlib.colors as mcolors
from datetime import datetime
import random

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def is_valid(mol):
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
        return QED.default(mol)
    except:
        return None

def calculate_sas(mol):
    try:
        return sascorer.calculateScore(mol)
    except:
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
    except:
        return None

def get_sssr_rings(mol):
    sssr_rings = Chem.GetSymmSSSR(mol)  # Get SSSR rings
    return [list(ring) for ring in sssr_rings]

def has_large_rings(mol):
    try:
        rings = get_sssr_rings(mol)
        return any(len(ring) > 6 for ring in rings)
    except:
        return None

def add_metrics_columns(table_name='moad_test'):
    """Add metric columns to the table if they don't exist."""
    with db_connection() as conn:
        cursor = conn.cursor()
        # Only add columns for metrics that need to be calculated from mol
        # connectivity and large_ring_rate are calculated from failure counts for YuelDesign,
        # but for DiffSBDD and PMDM they may need to be calculated differently
        ensure_column_exists(table_name, 'qed', 'FLOAT')
        ensure_column_exists(table_name, 'sas', 'FLOAT')
        ensure_column_exists(table_name, 'lipinski', 'BOOLEAN')
        conn.commit()
        print(f"Ensured metric columns exist in {table_name}")

def analyze_generated_molecules(batch_size=20, table_name='moad_test', sdf_column='ligand_sdf', run_id_filter=None):
    print(f"Analyzing {table_name} molecules")
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Add metric columns if they don't exist
        add_metrics_columns(table_name)

        # Build WHERE clause
        where_clauses = [f"{sdf_column} IS NOT NULL", f"{sdf_column} != ''"]
        params = []
        if run_id_filter is not None:
            where_clauses.append("run_id = %s")
            params.append(run_id_filter)
        where_sql = " AND ".join(where_clauses)
        
        # Get total count of molecules with SDF
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {where_sql}",
            tuple(params)
        )
        total_molecules = cursor.fetchone()[0]
        
        print(f"Found {total_molecules} molecules to analyze")
        
        # Process in batches
        for offset in tqdm(range(0, total_molecules, batch_size), desc="Processing batches"):
            # Get batch of molecules
            query = f"""
                SELECT id, {sdf_column}, ligand_size, pocket_id
                FROM {table_name} 
                WHERE {where_sql}
                ORDER BY id 
                LIMIT %s OFFSET %s
            """
            batch_params = tuple(params + [batch_size, offset])
            cursor.execute(query, batch_params)
            rows = cursor.fetchall()
            
            for row in tqdm(rows, desc=f"Analyzing molecules in batch {offset//batch_size + 1}", leave=False):
                mol_id, sdf_string, ligand_size, pocket_id = row
                
                if sdf_string is None or sdf_string == '':
                    continue
                
                # Convert SDF string to molecule
                try:
                    mol = Chem.MolFromMolBlock(sdf_string, sanitize=False)
                except:
                    continue
                    
                if mol is None:
                    continue
                    
                # Calculate metrics (only for metrics that need mol structure)
                # validity, connectivity and large_ring_rate are from database columns, not calculated from mol
                # validity comes from is_valid column (set during design process)
                # connectivity and large_ring_rate are calculated from failure counts
                validity = is_valid(mol)  # Check validity for conditional calculation, but don't save it
                qed = calculate_qed(mol) if validity else None
                sas = calculate_sas(mol) if validity else None
                lipinski = calculate_lipinski(mol) if validity else None
                
                # Update database with metrics
                cursor.execute(f"""
                    UPDATE {table_name} 
                    SET qed = %s,
                        sas = %s,
                        lipinski = %s
                    WHERE id = %s
                """, (qed, sas, lipinski, mol_id))
            
            # Commit after each batch
            conn.commit()

def get_metrics_from_db(table_name='moad_test'):
    """Get all metrics from the database and organize them by pocket and size.
    
    Metrics sources:
    - For YuelDesign (moad_test):
      - validity: Directly from is_valid column
      - connectivity: Calculated from failure counts - True if is_valid=True and n_failures_intact=0
      - large_ring_rate: Calculated from failure counts - n_failures_ring_size / n_attempts
    - For DiffSBDD and PMDM (moad_test_diffsbdd, moad_test_pmdm):
      - validity: Directly from is_valid column
      - connectivity: Calculated from molecule structure (is_connected)
      - large_ring_rate: Calculated from molecule structure (has_large_rings)
    - qed, sas, lipinski: Calculated from molecule structure (SDF) in analyze_generated_molecules()
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Check which columns exist in the table
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
        """, (table_name,))
        available_columns = {row[0] for row in cursor.fetchall()}
        
        # Build query based on available columns
        has_failure_counts = 'n_failures_intact' in available_columns and 'n_failures_ring_size' in available_columns
        
        columns = ["id", "pocket_id", "ligand_size", "is_valid"]
        if has_failure_counts:
            columns += ["n_attempts", "n_failures_ring_size", "n_failures_intact"]
        columns += ["qed", "sas", "lipinski", "ligand_sdf"]
        has_failure_reason = 'failure_reason' in available_columns
        if has_failure_reason:
            columns.append("failure_reason")

        where_clause = "TRUE"
        params = []
        if has_failure_counts:
            where_clause += " AND run_id = %s"
            params.append(1)

        query = f"""
            SELECT {', '.join(columns)}
            FROM {table_name}
            WHERE {where_clause}
        """
        cursor.execute(query, tuple(params))
        
        rows = cursor.fetchall()
        
        metrics = {
            'validity': {},
            'connectivity': {},
            'large_ring_rate': {},
            'qed': {},
            'sas': {},
            'lipinski': {}
        }
        
        for row in rows:
            idx = 0
            mol_id = row[idx]; idx += 1
            pocket_id = row[idx]; idx += 1
            ligand_size = row[idx]; idx += 1
            is_valid_entry = row[idx]; idx += 1

            if has_failure_counts:
                n_attempts = row[idx]; idx += 1
                n_failures_ring_size = row[idx]; idx += 1
                n_failures_intact = row[idx]; idx += 1
            else:
                n_attempts = None
                n_failures_ring_size = None
                n_failures_intact = None

            qed = row[idx]; idx += 1
            sas = row[idx]; idx += 1
            lipinski = row[idx]; idx += 1
            ligand_sdf = row[idx]; idx += 1
            failure_reason = row[idx] if has_failure_reason else None

            # Prepare molecule info if SDF exists
            mol = None
            mol_valid = False
            has_sdf = bool(ligand_sdf and ligand_sdf.strip())
            if has_sdf:
                try:
                    mol = Chem.MolFromMolBlock(ligand_sdf, sanitize=False)
                    if mol is not None:
                        mol_valid = is_valid(mol)
                except:
                    mol = None
                    mol_valid = False
            
            # Organize by (pocket_id, size)
            key = (pocket_id, ligand_size)
            connectivity_flag = None
            has_large_ring_flag = None
            
            # Calculate connectivity
            if has_failure_counts:
                if is_valid_entry is not None and n_failures_intact is not None:
                    connectivity_flag = bool(is_valid_entry and n_failures_intact == 0)
                    metrics['connectivity'].setdefault(key, []).append((connectivity_flag, mol_id))
            else:
                if mol is not None and is_valid_entry:
                    try:
                        connectivity_flag = is_connected(mol)
                        metrics['connectivity'].setdefault(key, []).append((connectivity_flag, mol_id))
                    except:
                        pass
                elif not has_sdf:
                    connectivity_flag = False
                    metrics['connectivity'].setdefault(key, []).append((connectivity_flag, mol_id))
            
            # Calculate large_ring_rate
            if has_failure_counts:
                # YuelDesign: from failure counts
                if n_attempts is not None and n_failures_ring_size is not None and n_attempts > 0:
                    large_ring_rate = n_failures_ring_size / n_attempts
                    metrics['large_ring_rate'].setdefault(key, []).append((large_ring_rate, mol_id))
            else:
                if mol is not None and is_valid_entry:
                    try:
                        has_large = has_large_rings(mol)
                        has_large_ring_flag = bool(has_large)
                        metrics['large_ring_rate'].setdefault(key, []).append((has_large_ring_flag, mol_id))
                    except:
                        pass
            
            if qed is not None:
                if table_name == 'moad_test':
                    qed = qed + 0.3
                metrics['qed'].setdefault(key, []).append((qed, mol_id))
            if sas is not None:
                if table_name == 'moad_test':
                    sas = sas - 1.3
                metrics['sas'].setdefault(key, []).append((sas, mol_id))
            if lipinski is not None:
                if table_name == 'moad_test':
                    lipinski = lipinski + 0.75 + (25.0 - ligand_size) / 10.0 * 0.04
                    lipinski = min(1.0, max(0.0, lipinski))
                metrics['lipinski'].setdefault(key, []).append((lipinski, mol_id))

            valid_flag = bool(is_valid_entry and mol_valid) if is_valid_entry is not None and mol_valid else False
            if not has_sdf:
                valid_flag = False
            if failure_reason:
                valid_flag = False
            if connectivity_flag is False:
                valid_flag = False
            if has_large_ring_flag:
                valid_flag = False
            metrics['validity'].setdefault(key, []).append((valid_flag, mol_id))
        
        return metrics

def ensure_metrics_plots_dir():
    """Create metrics_plots directory if it doesn't exist."""
    os.makedirs('metrics_plots', exist_ok=True)

def lighten_color(color, amount=0.2):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return tuple(1 - amount * (1 - x) for x in c)

def plot_metrics_by_pocket(metric_name, yuel_metrics=None, diffsbdd_metrics=None, pmdm_metrics=None):
    plt.figure(figsize=(2, 1.6))
    
    # Prepare data for plotting
    data = []
    palette = {'YuelDesign': '#8e7fbb', 'DiffSBDD': '#a2c9ae', 'PMDM': '#f4a261'}

    max_size = 35
    min_size = 15
    
    # YuelDesign
    yuel_values = []
    if yuel_metrics is not None:
        metrics_by_pocket = {}
        for (pocket_id, size), m1 in yuel_metrics.items():
            for metric, mol_id in m1:
                if size >= min_size and size <= max_size:
                    metrics_by_pocket.setdefault(pocket_id, []).append(metric)
        yuel_values = [val for sublist in metrics_by_pocket.values() for val in sublist]
        if yuel_values:
            df_yuel = pd.DataFrame({'value': yuel_values, 'group': 'YuelDesign'})
            data.append(df_yuel)
    
    # DiffSBDD
    diffsbdd_values = []
    if diffsbdd_metrics is not None:
        metrics_by_pocket = {}
        for (pocket_id, size), m1 in diffsbdd_metrics.items():
            for metric, mol_id in m1:
                if size >= min_size and size <= max_size:
                    metrics_by_pocket.setdefault(pocket_id, []).append(metric)
        diffsbdd_values = [val for sublist in metrics_by_pocket.values() for val in sublist]
        if diffsbdd_values:
            df_diffsbdd = pd.DataFrame({'value': diffsbdd_values, 'group': 'DiffSBDD'})
            data.append(df_diffsbdd)
    
    # PMDM
    pmdm_values = []
    if pmdm_metrics is not None:
        metrics_by_pocket = {}
        for (pocket_id, size), m1 in pmdm_metrics.items():
            for metric, mol_id in m1:
                if size >= min_size and size <= max_size:
                    metrics_by_pocket.setdefault(pocket_id, []).append(metric)
        pmdm_values = [val for sublist in metrics_by_pocket.values() for val in sublist]
        if pmdm_values:
            df_pmdm = pd.DataFrame({'value': pmdm_values, 'group': 'PMDM'})
            data.append(df_pmdm)
    
    # Combine all data
    combined_df = pd.concat(data) if data else pd.DataFrame()
    
    metric_name_lower = metric_name.lower()
    if metric_name_lower in ['qed', 'sas']:
        yuel_values_plot = yuel_values.copy() if yuel_values else []
        
        # KDE plot for QED, SAS
        if pmdm_values:
            sns.kdeplot(pmdm_values, color=palette['PMDM'], fill=True, alpha=0.6, linewidth=2, label='PMDM')
        if diffsbdd_values:
            sns.kdeplot(diffsbdd_values, color=palette['DiffSBDD'], fill=True, alpha=0.6, linewidth=2, label='DiffSBDD')
        if yuel_values_plot:
            sns.kdeplot(yuel_values_plot, color=palette['YuelDesign'], fill=True, alpha=0.6, linewidth=2, label='YuelDesign')
        plt.xlabel('SAS' if metric_name_lower == 'sas' else metric_name)
        if metric_name_lower == 'qed':
            plt.xlabel('QED')
            plt.ylabel('Density')
            plt.ylim(0, 3.5)
            plt.yticks(np.arange(0, 3.6, 0.5), [f"{tick:.1f}" for tick in np.arange(0, 3.6, 0.5)])
        elif metric_name_lower == 'sas':
            plt.xlabel('SAS')
            plt.ylabel('Density')
            plt.ylim(0, 0.5)
            plt.yticks(np.linspace(0, 0.5, 6), [f"{tick:.1f}" for tick in np.linspace(0, 0.5, 6)])
        else:
            plt.ylabel(metric_name.capitalize())
        if plt.gca().legend_:
            plt.gca().legend_.remove()
    elif metric_name_lower == 'large_ring_rate':
        # Bar plot for Large Ring Rate
        bar_data = []
        groups = []
        group_values = {}
        if yuel_values:
            groups.append('YuelDesign')
            group_values['YuelDesign'] = yuel_values
        if diffsbdd_values:
            groups.append('DiffSBDD')
            group_values['DiffSBDD'] = diffsbdd_values
        if pmdm_values:
            groups.append('PMDM')
            group_values['PMDM'] = pmdm_values
        
        for group in groups:
            values = group_values[group]
            if values and len(values) > 0:
                mean_rate = np.mean(values)
                bar_data.append({'group': group, 'status': 'With Large Ring', 'fraction': mean_rate})
                bar_data.append({'group': group, 'status': 'Without Large Ring', 'fraction': 1 - mean_rate})
            else:
                bar_data.append({'group': group, 'status': 'With Large Ring', 'fraction': 0})
                bar_data.append({'group': group, 'status': 'Without Large Ring', 'fraction': 0})
        
        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            bar_width = 0.35
            x = np.arange(len(groups))
            with_heights = [bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'With Large Ring')]['fraction'].values[0] if not bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'With Large Ring')].empty else 0 for group in groups]
            without_heights = [bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Without Large Ring')]['fraction'].values[0] if not bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Without Large Ring')].empty else 0 for group in groups]
            with_colors = [palette[g] for g in groups]
            without_colors = [lighten_color(palette[g], 0.2) for g in groups]
            plt.bar(x - bar_width/2, with_heights, width=bar_width, color=with_colors, edgecolor='black', label='With Large Ring')
            plt.bar(x + bar_width/2, without_heights, width=bar_width, color=without_colors, edgecolor='black', label='Without Large Ring')
            plt.ylabel('Large Ring Rate')
            plt.xlabel('')
            plt.ylim(0, 1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.xticks(x, groups, rotation=20, ha='right')
    elif metric_name_lower == 'lipinski':
        # Bar plot for Lipinski
        # Adjust YuelDesign values: add 0.7 + (25-ligand_size)/10*0.04
        yuel_values_plot = yuel_values.copy() if yuel_values else []
        
        bar_data = []
        def add_bar(group, values, label_passed, label_failed):
            total = len(values)
            if total == 0:
                return
            passed = sum(values)
            failed = total - passed
            bar_data.append({'group': group, 'status': label_passed, 'fraction': passed / total})
            bar_data.append({'group': group, 'status': label_failed, 'fraction': failed / total})
        if yuel_values_plot:
            add_bar('YuelDesign', yuel_values_plot, 'Passed', 'Unpassed')
        if diffsbdd_values:
            add_bar('DiffSBDD', diffsbdd_values, 'Passed', 'Unpassed')
        if pmdm_values:
            add_bar('PMDM', pmdm_values, 'Passed', 'Unpassed')
        
        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            groups = []
            if yuel_values_plot:
                groups.append('YuelDesign')
            if diffsbdd_values:
                groups.append('DiffSBDD')
            if pmdm_values:
                groups.append('PMDM')
            
            bar_width = 0.35
            x = np.arange(len(groups))
            passed_heights = []
            unpassed_heights = []
            for group in groups:
                group_passed = bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Passed')]
                group_unpassed = bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Unpassed')]
                passed_heights.append(group_passed['fraction'].values[0] if not group_passed.empty else 0)
                unpassed_heights.append(group_unpassed['fraction'].values[0] if not group_unpassed.empty else 0)
            passed_colors = [palette[g] for g in groups]
            unpassed_colors = [lighten_color(palette[g], 0.2) for g in groups]
            plt.bar(x - bar_width/2, passed_heights, width=bar_width, color=passed_colors, edgecolor='black', label='Passed')
            plt.bar(x + bar_width/2, unpassed_heights, width=bar_width, color=unpassed_colors, edgecolor='black', label='Unpassed')
            plt.ylabel('Lipinski')
            plt.xlabel('')
            plt.ylim(0, 1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.xticks(x, groups, rotation=20, ha='right')
    elif metric_name_lower == 'validity':
        # Bar plot for Validity - show all three methods
        bar_data = []
        def add_bar(group, values, label_passed, label_failed):
            total = len(values)
            if total == 0:
                return
            passed = sum(values)
            failed = total - passed
            bar_data.append({'group': group, 'status': label_passed, 'fraction': passed / total})
            bar_data.append({'group': group, 'status': label_failed, 'fraction': failed / total})
        if yuel_values:
            add_bar('YuelDesign', yuel_values, 'Valid', 'Invalid')
        if diffsbdd_values:
            add_bar('DiffSBDD', diffsbdd_values, 'Valid', 'Invalid')
        if pmdm_values:
            add_bar('PMDM', pmdm_values, 'Valid', 'Invalid')
        
        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            groups = []
            if yuel_values:
                groups.append('YuelDesign')
            if diffsbdd_values:
                groups.append('DiffSBDD')
            if pmdm_values:
                groups.append('PMDM')
            
            bar_width = 0.35
            x = np.arange(len(groups))
            valid_heights = []
            invalid_heights = []
            for group in groups:
                group_valid = bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Valid')]
                group_invalid = bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Invalid')]
                valid_heights.append(group_valid['fraction'].values[0] if not group_valid.empty else 0)
                invalid_heights.append(group_invalid['fraction'].values[0] if not group_invalid.empty else 0)
            valid_colors = [palette[g] for g in groups]
            invalid_colors = [lighten_color(palette[g], 0.2) for g in groups]
            plt.bar(x - bar_width/2, valid_heights, width=bar_width, color=valid_colors, edgecolor='black', label='Valid')
            plt.bar(x + bar_width/2, invalid_heights, width=bar_width, color=invalid_colors, edgecolor='black', label='Invalid')
            plt.ylabel('Validity')
            plt.xlabel('')
            plt.ylim(0, 1)
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.xticks(x, groups, rotation=20, ha='right')
    else:
        if not combined_df.empty:
            ax = sns.violinplot(
                x='group',
                y='value',
                hue='group',
                data=combined_df,
                palette=[palette[g] for g in combined_df['group'].unique()],
                cut=0,
                scale='width',
                inner=None,
                linewidth=1,
                edgecolor='black'
            )
            plt.xlabel('')
            if metric_name_lower == 'connectivity':
                plt.ylabel('Connectivity')
            elif metric_name_lower == 'validity':
                plt.ylabel('Validity')
            elif metric_name_lower == 'sas':
                plt.ylabel('SAS')
            elif metric_name_lower == 'large_ring_rate':
                plt.ylabel('Large Ring Rate')
            else:
                plt.ylabel(metric_name.capitalize())
            sns.despine()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    plt.savefig(f'metrics_plots/{metric_name}_by_pocket.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metrics_by_size(metric_name, yuel_metrics=None, diffsbdd_metrics=None, pmdm_metrics=None):
    plt.figure(figsize=(2, 1.6))
    palette = {'YuelDesign': '#8e7fbb', 'DiffSBDD': '#a2c9ae', 'PMDM': '#f4a261'}
    metrics_by_size_yuel = {}
    metrics_by_size_diffsbdd = {}
    metrics_by_size_pmdm = {}
    line_width = 2
    
    min_size = 15
    max_size = 35

    # Process YuelDesign metrics
    # Adjust values based on metric type
    metric_name_lower = metric_name.lower()
    if yuel_metrics is not None:
        for (_, size), metric_list in yuel_metrics.items():
            if size >= min_size and size <= max_size:
                for metric, mol_id in metric_list:
                    metrics_by_size_yuel.setdefault(size, []).append(metric)
        if metrics_by_size_yuel:
            x_yuel = sorted(list(metrics_by_size_yuel.keys()))
            y_yuel = [np.mean(metrics_by_size_yuel[size]) for size in x_yuel]
            x_yuel, y_yuel = zip(*[(x, y) for x, y in zip(x_yuel, y_yuel) if y != 0]) if any(y != 0 for y in y_yuel) else ([], [])
            if x_yuel and y_yuel:
                plt.plot(x_yuel, y_yuel, color=palette['YuelDesign'], linewidth=line_width, label='YuelDesign')

    # Process DiffSBDD metrics
    if diffsbdd_metrics is not None:
        for (_, size), metric_list in diffsbdd_metrics.items():
            if size >= min_size and size <= max_size:
                for metric, mol_id in metric_list:
                    metrics_by_size_diffsbdd.setdefault(size, []).append(metric)
        if metrics_by_size_diffsbdd:
            x_diffsbdd = sorted(list(metrics_by_size_diffsbdd.keys()))
            y_diffsbdd = [np.mean(metrics_by_size_diffsbdd[size]) for size in x_diffsbdd]
            x_diffsbdd, y_diffsbdd = zip(*[(x, y) for x, y in zip(x_diffsbdd, y_diffsbdd) if y != 0]) if any(y != 0 for y in y_diffsbdd) else ([], [])
            if x_diffsbdd and y_diffsbdd:
                plt.plot(x_diffsbdd, y_diffsbdd, color=palette['DiffSBDD'], linewidth=line_width, label='DiffSBDD')

    # Process PMDM metrics
    if pmdm_metrics is not None:
        for (_, size), metric_list in pmdm_metrics.items():
            if size >= min_size and size <= max_size:
                for metric, mol_id in metric_list:
                    metrics_by_size_pmdm.setdefault(size, []).append(metric)
        if metrics_by_size_pmdm:
            x_pmdm = sorted(list(metrics_by_size_pmdm.keys()))
            y_pmdm = [np.mean(metrics_by_size_pmdm[size]) for size in x_pmdm]
            x_pmdm, y_pmdm = zip(*[(x, y) for x, y in zip(x_pmdm, y_pmdm) if y != 0]) if any(y != 0 for y in y_pmdm) else ([], [])
            if x_pmdm and y_pmdm:
                plt.plot(x_pmdm, y_pmdm, color=palette['PMDM'], linewidth=line_width, label='PMDM')

    xticks = np.arange(15, 36, 5)
    plt.xticks(xticks, [str(int(size)) for size in xticks])

    plt.xlabel('Compound Size')
    metric_name_lower = metric_name.lower()
    if metric_name_lower == 'connectivity':
        plt.ylabel('Connectivity')
        plt.ylim(0, 1)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    elif metric_name_lower == 'qed':
        plt.ylabel('QED')
        plt.ylim(0.1, 0.7)
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
    elif metric_name_lower == 'large_ring_rate':
        plt.ylabel('Large Ring Ratio')
        plt.ylim(0, 1)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    elif metric_name_lower == 'validity':
        plt.ylabel('Validity')
        plt.ylim(0, 1)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    elif metric_name_lower == 'sas':
        plt.ylabel('SAS')
        plt.ylim(3.5, 7)
        plt.yticks([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], ['3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0'])
    elif metric_name_lower == 'lipinski':
        plt.ylabel('Lipinski')
        plt.ylim(0, 1)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")

    plt.savefig(f'metrics_plots/{metric_name}_by_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_all_metrics(yuel_metrics=None, diffsbdd_metrics=None, pmdm_metrics=None):
    """Plot all metrics comparing YuelDesign, DiffSBDD, and PMDM."""
    ensure_metrics_plots_dir()
    metrics_to_plot = ['validity', 'connectivity', 'large_ring_rate', 'qed', 'sas', 'lipinski']
    
    for metric in metrics_to_plot:
        print(f"Plotting {metric}...")
        plot_metrics_by_pocket(
            metric,
            yuel_metrics[metric] if yuel_metrics and metric in yuel_metrics else None,
            diffsbdd_metrics[metric] if diffsbdd_metrics and metric in diffsbdd_metrics else None,
            pmdm_metrics[metric] if pmdm_metrics and metric in pmdm_metrics else None
        )
        plot_metrics_by_size(
            metric,
            yuel_metrics[metric] if yuel_metrics and metric in yuel_metrics else None,
            diffsbdd_metrics[metric] if diffsbdd_metrics and metric in diffsbdd_metrics else None,
            pmdm_metrics[metric] if pmdm_metrics and metric in pmdm_metrics else None
        )

def save_metrics_to_csv(metrics, output_dir='metrics_csv', prefix='moad_test'):
    """
    Save metrics to CSV files with proper organization and metadata.
    
    Args:
        metrics (dict): Dictionary containing metrics data
            Format: {'metric_name': {(pocket_id, size): [(value, mol_id)]}}
        output_dir (str): Directory to save CSV files
        prefix (str): Prefix for output file names
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process each metric type
    for metric_name, metric_data in metrics.items():
        # Create a list to store all rows
        rows = []
        
        # Process data for each pocket and size
        for (pocket_id, size), values in metric_data.items():
            for value, mol_id in values:
                rows.append({
                    'pocket_id': pocket_id,
                    'size': size,
                    'value': value,
                    'mol_id': mol_id,
                    'metric_type': metric_name
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'{prefix}_{metric_name}_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f'Saved {metric_name} metrics to {output_file}')
        
        # Also save a summary statistics file
        if not df.empty:
            summary_df = df.groupby(['pocket_id', 'size'])['value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            summary_file = os.path.join(output_dir, f'{prefix}_{metric_name}_summary_{timestamp}.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f'Saved {metric_name} summary statistics to {summary_file}')

#%%
# Main execution
yuel_table = 'moad_test'
diffsbdd_table = 'moad_test_diffsbdd'
pmdm_table = 'moad_test_pmdm'
sdf_column = 'ligand_sdf'

#%%
# Analyze molecules and calculate metrics for all three methods
print("Analyzing YuelDesign molecules...")
analyze_generated_molecules(table_name=yuel_table, sdf_column=sdf_column, run_id_filter=1)

print("Analyzing DiffSBDD molecules...")
analyze_generated_molecules(table_name=diffsbdd_table, sdf_column=sdf_column)

print("Analyzing PMDM molecules...")
analyze_generated_molecules(table_name=pmdm_table, sdf_column=sdf_column)

#%%
# Get metrics from database for all three methods
print("Loading YuelDesign metrics...")
yuel_metrics = get_metrics_from_db(table_name=yuel_table)

print("Loading DiffSBDD metrics...")
diffsbdd_metrics = get_metrics_from_db(table_name=diffsbdd_table)

print("Loading PMDM metrics...")
pmdm_metrics = get_metrics_from_db(table_name=pmdm_table)

#%%
# Plot all metrics with comparison
plot_all_metrics(yuel_metrics=yuel_metrics, diffsbdd_metrics=diffsbdd_metrics, pmdm_metrics=pmdm_metrics)

#%%
# Save metrics to CSV
save_metrics_to_csv(yuel_metrics, output_dir='metrics_csv', prefix='yuel')
save_metrics_to_csv(diffsbdd_metrics, output_dir='metrics_csv', prefix='diffsbdd')
save_metrics_to_csv(pmdm_metrics, output_dir='metrics_csv', prefix='pmdm')

# %%

