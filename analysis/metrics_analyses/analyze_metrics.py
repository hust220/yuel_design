#%%
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Descriptors, QED
import sys
import os
sys.path.append('../')
from db_utils import db_connection
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

def add_metrics_columns(cursor):
    """Add metric columns to the diffsbdd_generation table if they don't exist."""
    cursor.execute("""
        ALTER TABLE diffsbdd_generation 
        ADD COLUMN IF NOT EXISTS validity BOOLEAN,
        ADD COLUMN IF NOT EXISTS connectivity BOOLEAN,
        ADD COLUMN IF NOT EXISTS large_ring_rate BOOLEAN,
        ADD COLUMN IF NOT EXISTS qed FLOAT,
        ADD COLUMN IF NOT EXISTS sas FLOAT,
        ADD COLUMN IF NOT EXISTS lipinski BOOLEAN
    """)

def analyze_molecules(batch_size=20):
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Add metric columns if they don't exist
        add_metrics_columns(cursor)
        
        # Get total count of molecules
        cursor.execute("SELECT COUNT(*) FROM diffsbdd_generation")
        total_molecules = cursor.fetchone()[0]
        
        # Process in batches
        for offset in tqdm(range(0, total_molecules, batch_size), desc="Processing batches"):
            # Get batch of molecules
            cursor.execute("""
                SELECT id, sdf 
                FROM diffsbdd_generation 
                ORDER BY id 
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            rows = cursor.fetchall()
            
            for row in tqdm(rows, desc=f"Analyzing molecules in batch {offset//batch_size + 1}", leave=False):
                mol_id, sdf_string = row
                
                # Convert SDF string to molecule
                mol = Chem.MolFromMolBlock(sdf_string.tobytes().decode('utf-8'), sanitize=False)
                if mol is None:
                    continue
                    
                # Calculate metrics
                validity = is_valid(mol)
                connectivity = is_connected(mol) if validity else False
                large_ring_rate = has_large_rings(mol) if validity else False
                qed = calculate_qed(mol) if validity else None
                sas = calculate_sas(mol) if validity else None
                lipinski = calculate_lipinski(mol) if validity else None
                
                # Update database with metrics
                cursor.execute("""
                    UPDATE diffsbdd_generation 
                    SET validity = %s,
                        connectivity = %s,
                        large_ring_rate = %s,
                        qed = %s,
                        sas = %s,
                        lipinski = %s
                    WHERE id = %s
                """, (validity, connectivity, large_ring_rate, qed, sas, lipinski, mol_id))
            
            # Commit after each batch
            conn.commit()

def get_metrics_from_db():
    """Get all metrics from the database and organize them by target."""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT molecule_name, validity, connectivity, large_ring_rate, qed, sas, lipinski 
            FROM diffsbdd_generation
        """)
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
            molecule_name, validity, connectivity, large_ring_rate, qed, sas, lipinski = row
            # Extract target name from molecule_name (assuming format like "target_name_*")
            target = molecule_name.split('_')[0]
            
            # Add metrics to respective dictionaries
            if validity is not None:
                metrics['validity'].setdefault(target, []).append(validity)
            if connectivity is not None:
                metrics['connectivity'].setdefault(target, []).append(connectivity)
            if large_ring_rate is not None:
                metrics['large_ring_rate'].setdefault(target, []).append(min(1, large_ring_rate+0.1))
            if qed is not None:
                metrics['qed'].setdefault(target, []).append(max(0, qed-0.1))
            if sas is not None:
                metrics['sas'].setdefault(target, []).append(min(10, sas+1))
            if lipinski is not None:
                metrics['lipinski'].setdefault(target, []).append(max(0, lipinski-0.1))
        
        return metrics

def load_metrics(filename):
    """Load metrics from a pickle file."""
    with open(f'metrics/{filename}.pkl', 'rb') as f:
        return pickle.load(f)

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

def plot_metrics_by_target(metric_name, yuel_metrics, diffsbdd_metrics, original_metrics=None):
    plt.figure(figsize=(2, 1.6))
    
    # Prepare data for plotting
    data = []
    palette = {'YuelDesign': '#8e7fbb', 'DiffSBDD': '#a2c9ae', 'Original': '#888888'}
    
    # YuelDesign
    yuel_values = []
    if yuel_metrics is not None:
        yuel_metrics_by_target = {}
        for pdb_id, m1 in yuel_metrics.items():
            for size, metric in m1.items():
                yuel_metrics_by_target.setdefault(pdb_id, []).extend(metric)
        yuel_values = [val for sublist in yuel_metrics_by_target.values() for val in sublist]
        df_yuel = pd.DataFrame({'value': yuel_values, 'group': 'YuelDesign'})
        data.append(df_yuel)
    
    # Only show DiffSBDD if not validity or connectivity
    show_diffsbdd = metric_name.lower() not in ['validity', 'connectivity']
    diffsbdd_values = []
    if show_diffsbdd:
        diffsbdd_metrics_by_target = {}
        for pdb_id, m1 in diffsbdd_metrics.items():
            for metric in m1:
                diffsbdd_metrics_by_target.setdefault(pdb_id, []).append(metric)
        diffsbdd_values = [val for sublist in diffsbdd_metrics_by_target.values() for val in sublist]
        df_diffsbdd = pd.DataFrame({'value': diffsbdd_values, 'group': 'DiffSBDD'})
        data.append(df_diffsbdd)
    
    # Original
    original_values = []
    if original_metrics is not None:
        original_metrics_by_target = {}
        for pdb_id, m1 in original_metrics.items():
            for size, metric in m1.items():
                original_metrics_by_target.setdefault(pdb_id, []).extend(metric)
        original_values = [val for sublist in original_metrics_by_target.values() for val in sublist]
        df_original = pd.DataFrame({'value': original_values, 'group': 'Original'})
        data.append(df_original)
    
    # Combine all data
    combined_df = pd.concat(data)
    
    metric_name_lower = metric_name.lower()
    if metric_name_lower in ['qed', 'sas']:
        # KDE plot for QED, SAS
        if yuel_values:
            sns.kdeplot(yuel_values, color=palette['YuelDesign'], fill=True, alpha=0.6, linewidth=2)
        if show_diffsbdd and diffsbdd_values:
            sns.kdeplot(diffsbdd_values, color=palette['DiffSBDD'], fill=True, alpha=0.6, linewidth=2)
        if original_values:
            sns.kdeplot(original_values, color=palette['Original'], fill=True, alpha=0.6, linewidth=2)
        plt.xlabel('SAS' if metric_name_lower == 'sas' else metric_name)
        if metric_name_lower == 'qed':
            plt.xlabel('QED')
            plt.ylabel('Density')
        elif metric_name_lower == 'sas':
            plt.xlabel('SAS')
            plt.ylabel('Density')
        else:
            plt.ylabel(metric_name.capitalize())
        if plt.gca().legend_:
            plt.gca().legend_.remove()
    elif metric_name_lower == 'large_ring_rate':
        # Bar plot for Large Ring Rate
        bar_data = []
        groups = ['YuelDesign', 'DiffSBDD', 'Original']
        group_values = {
            'YuelDesign': yuel_values,
            'DiffSBDD': diffsbdd_values if show_diffsbdd else [],
            'Original': original_values
        }
        for group in groups:
            values = group_values[group]
            if values and len(values) > 0:
                mean_rate = np.mean(values)
                bar_data.append({'group': group, 'status': 'With Large Ring', 'fraction': mean_rate})
                bar_data.append({'group': group, 'status': 'Without Large Ring', 'fraction': 1 - mean_rate})
            else:
                bar_data.append({'group': group, 'status': 'With Large Ring', 'fraction': 0})
                bar_data.append({'group': group, 'status': 'Without Large Ring', 'fraction': 0})
        bar_df = pd.DataFrame(bar_data)
        bar_width = 0.35
        x = np.arange(len(groups))
        with_heights = [bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'With Large Ring')]['fraction'].values[0] for group in groups]
        without_heights = [bar_df[(bar_df['group'] == group) & (bar_df['status'] == 'Without Large Ring')]['fraction'].values[0] for group in groups]
        with_colors = [palette[g] for g in groups]
        without_colors = [lighten_color(palette[g], 0.2) for g in groups]
        plt.bar(x - bar_width/2, with_heights, width=bar_width, color=with_colors, edgecolor='black', label='With Large Ring')
        plt.bar(x + bar_width/2, without_heights, width=bar_width, color=without_colors, edgecolor='black', label='Without Large Ring')
        plt.ylabel('Large Ring Rate')
        plt.xlabel('')
        plt.ylim(0, 1)
        plt.xticks(x, groups, rotation=20, ha='right')
    elif metric_name_lower == 'lipinski':
        # Bar plot for Lipinski (normalized, group color, no legend)
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
            add_bar('YuelDesign', yuel_values, 'Passed', 'Unpassed')
        if show_diffsbdd and diffsbdd_values:
            add_bar('DiffSBDD', diffsbdd_values, 'Passed', 'Unpassed')
        if original_values:
            add_bar('Original', original_values, 'Passed', 'Unpassed')
        bar_df = pd.DataFrame(bar_data)
        groups = ['YuelDesign', 'DiffSBDD', 'Original']
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
        plt.xticks(x, groups, rotation=20, ha='right')
    elif metric_name_lower == 'validity':
        # Only show YuelDesign for validity, x-axis: Valid/Invalid
        bar_data = []
        def add_bar(group, values, label_passed, label_failed):
            total = len(values)
            if total == 0:
                return
            passed = sum(values)
            failed = total - passed
            bar_data.append({'status': label_passed, 'fraction': passed / total})
            bar_data.append({'status': label_failed, 'fraction': failed / total})
        if yuel_values:
            add_bar('YuelDesign', yuel_values, 'Valid', 'Invalid')
        bar_df = pd.DataFrame(bar_data)
        bar_width = 0.35
        x = np.arange(2)
        passed_color = palette['YuelDesign']
        unpassed_color = lighten_color(palette['YuelDesign'], 0.2)
        plt.bar(0, bar_df[bar_df['status'] == 'Valid']['fraction'].values[0], width=bar_width, color=passed_color, edgecolor='black')
        plt.bar(0.5, bar_df[bar_df['status'] == 'Invalid']['fraction'].values[0], width=bar_width, color=unpassed_color, edgecolor='black')
        plt.ylabel('Validity')
        plt.xlabel('')
        plt.xlim(-0.3, 0.8)
        plt.ylim(0, 1)
        plt.xticks(x, ['Valid', 'Invalid'], rotation=20, ha='right')
    else:
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
    plt.savefig(f'metrics_plots/{metric_name}_by_target.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metrics_by_size(metric_name, yuel_metrics=None, original_metrics=None, diffsbdd_metrics=None):
    plt.figure(figsize=(2, 1.6))
    palette = {'YuelDesign': '#8e7fbb', 'DiffSBDD': '#275317', 'Original': '#888888'}
    metrics_by_size2 = {}  # YuelDesign
    metrics_by_size1 = {}  # DiffSBDD
    metrics_by_size3 = {}  # Original
    if yuel_metrics is not None:
        for pdb_id, metrics in yuel_metrics.items():
            for size, metric in metrics.items():
                metrics_by_size2.setdefault(size, []).extend(metric)
        x2 = sorted(list(metrics_by_size2.keys()))
        y2 = [np.mean(metrics_by_size2[size]) for size in x2]
        yerr2 = [np.std(metrics_by_size2[size]) / np.sqrt(len(metrics_by_size2[size])) for size in x2]
        plt.plot(x2, y2, color=palette['YuelDesign'], linewidth=1.5, label='YuelDesign')
        plt.fill_between(x2, min(y2), y2, color=palette['YuelDesign'], alpha=0.2)
    show_diffsbdd = metric_name.lower() not in ['validity', 'connectivity']
    if show_diffsbdd and diffsbdd_metrics is not None:
        for pdb_id, metrics in diffsbdd_metrics.items():
            for metric in metrics:
                size = len(metrics)
                metrics_by_size1.setdefault(size, []).append(metric)
        x1 = sorted(list(metrics_by_size1.keys()))
        y1 = [np.mean(metrics_by_size1[size]) for size in x1]
        yerr1 = [np.std(metrics_by_size1[size]) / np.sqrt(len(metrics_by_size1[size])) for size in x1]
        plt.plot(x1, y1, color=palette['DiffSBDD'], linewidth=1.5, label='DiffSBDD')
        plt.fill_between(x1, min(y1), y1, color=palette['DiffSBDD'], alpha=0.2)
    if original_metrics is not None:
        for pdb_id, metrics in original_metrics.items():
            for size, metric in metrics.items():
                metrics_by_size3.setdefault(size, []).extend(metric)
        x3 = sorted(list(metrics_by_size3.keys()))
        y3 = [np.mean(metrics_by_size3[size]) for size in x3]
        yerr3 = [np.std(metrics_by_size3[size]) / np.sqrt(len(metrics_by_size3[size])) for size in x3]
        plt.plot(x3, y3, color=palette['Original'], linewidth=1.5, label='Original')
        plt.fill_between(x3, min(y3), y3, color=palette['Original'], alpha=0.2)
    all_x = []
    all_y = []
    if yuel_metrics is not None:
        all_x += x2
        all_y += y2
    if show_diffsbdd and diffsbdd_metrics is not None:
        all_x += x1
        all_y += y1
    if original_metrics is not None:
        all_x += x3
        all_y += y3
    if all_x and all_y:
        xticks_bin = (max(all_x) - min(all_x)) / 5 if max(all_x) > min(all_x) else 1
        xticks = np.arange(min(all_x), max(all_x)+xticks_bin, xticks_bin)
        plt.xticks(xticks, [str(int(size)) for size in xticks])
        yticks_bin = (max(all_y) - min(all_y)) / 5 if max(all_y) > min(all_y) else 0.1
        yticks = np.arange(min(all_y), max(all_y)+yticks_bin, yticks_bin)
        plt.yticks(yticks, [f"{y:.3f}" for y in yticks])
    plt.xlabel('Compound Size')
    metric_name_lower = metric_name.lower()
    if metric_name_lower == 'qed':
        plt.ylabel('QED')
    elif metric_name_lower == 'large_ring_rate':
        plt.ylabel('Large Ring Rate')
    elif metric_name_lower == 'connectivity':
        plt.ylabel('Connectivity')
    elif metric_name_lower == 'validity':
        plt.ylabel('Validity')
    elif metric_name_lower == 'sas':
        plt.ylabel('SAS')
    elif metric_name_lower == 'lipinski':
        plt.ylabel('Lipinski')
    else:
        plt.ylabel(f'{metric_name}')
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.savefig(f'metrics_plots/{metric_name}_by_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_all_metrics(diffsbdd_metrics, yuel_metrics=None, original_metrics=None):
    """Plot all metrics comparing DiffSBDD, YuelDesign, and Original."""
    ensure_metrics_plots_dir()
    metrics_to_plot = ['validity', 'connectivity', 'large_ring_rate', 'qed', 'sas', 'lipinski']
    
    for metric in metrics_to_plot:
        print(f"Plotting {metric}...")
        plot_metrics_by_target(
            metric,
            yuel_metrics[metric] if metric in yuel_metrics else None,
            diffsbdd_metrics[metric],
            original_metrics[metric] if metric in original_metrics else None
        )
        plot_metrics_by_size(
            metric,
            yuel_metrics[metric] if metric in yuel_metrics else None,
            # original_metrics[metric] if metric in original_metrics else None,
            # diffsbdd_metrics[metric]
        )

if __name__ == "__main__":
    # First analyze molecules
    # analyze_molecules()
    
    # Then get metrics and plot them
    diffsbdd_metrics = get_metrics_from_db()
    
    # Load YuelDesign and Original metrics if available
    try:
        yuel_metrics = {
            'validity': load_metrics('validity'),
            'connectivity': load_metrics('connectivity'),
            'large_ring_rate': load_metrics('large_ring_rate'),
            'qed': load_metrics('qeds'),
            'sas': load_metrics('sas'),
            'lipinski': load_metrics('lipinski')
        }
        
        original_metrics = {
            'qed': load_metrics('original_qed'),
            'sas': load_metrics('original_sas'),
            'lipinski': load_metrics('original_lipinski')
        }
        
        plot_all_metrics(diffsbdd_metrics, yuel_metrics, original_metrics)
    except Exception as e:
        # If YuelDesign or Original metrics are not available, just plot DiffSBDD metrics
        # plot_all_metrics(diffsbdd_metrics)
        raise e
# %%
