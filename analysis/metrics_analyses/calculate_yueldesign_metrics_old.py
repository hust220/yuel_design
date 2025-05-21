#%%
# scan all the .sdf files in the test_results/MOAD_test folder
# calculate the validity of the molecules
import os
from rdkit import Chem
from rdkit.Chem import AllChem
# from src.metrics import is_valid
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
import random
from rdkit import rdBase, RDLogger
import networkx as nx
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors
import pickle
from src import sascorer
from rdkit.Chem import Lipinski, Descriptors
import seaborn as sns
# Disable all RDKit warnings
RDLogger.DisableLog('rdApp.*')

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
    # Default QED calculation (uses all 8 descriptors)
    qed_score = QED.default(mol)

    # Weighted QED (custom weights possible)
    # weights = QED.weights_max  # Predefined weights from the paper
    # qed_weighted = QED.qed(mol, weights)

    return qed_score

def calculate_sas(mol):
    sas_score = sascorer.calculateScore(mol)
    return sas_score

def scan_folder(folder_path):
    sdf_files = [file for file in os.listdir(folder_path) if file.endswith(".sdf")]
    print('Found', len(sdf_files), 'sdf files')
    for sdf_file in tqdm(sdf_files, total=len(sdf_files)):
        # if name is like 1z8d_1_size10_n10_s25183.sdf
        m = re.match(r"(.+)_size(\d+)_n(\d+)_s(\d+)\.sdf", sdf_file)
        if m:
            pdb_id = m.group(1)
            size = int(m.group(2))
            nsamples = int( m.group(3))
            seed = int(m.group(4))
            # print(f"Processing {sdf_file} with pdb_id {pdb_id}, size {size}, nsamples {nsamples}, seed {seed}")

            suppl = Chem.SDMolSupplier(os.path.join(folder_path, sdf_file), sanitize=False, strictParsing=False)
            mol_count = 0
            for mol in suppl:
                if mol is None:
                    continue
                yield mol, pdb_id, size, nsamples

def scan_sdf(sdf_file, nsamples=1):
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, strictParsing=False)
    for mol in suppl:
        if mol is None:
            continue
        try:
            mol = Chem.RemoveAllHs(mol)
            name = mol.GetProp('_Name')
            size = mol.GetNumAtoms()
        except:
            continue
        yield mol, name, size, nsamples

def calculate_metrics(molecules, callback=None, max_molecules=None, ignore_errors=True):
    # read the sdf file
    metrics = {}
    molecules_processed = 0
    bar = tqdm(desc='Processing molecules')
    for mol, name, size, nsamples in molecules:
        if callback is not None:
            try:
                metric = callback(mol)
                metrics.setdefault(name, {}).setdefault(size, []).append(metric)
            except Exception as e:
                if not ignore_errors:
                    raise e
        bar.update(1)
        molecules_processed += 1
        if max_molecules is not None and molecules_processed >= max_molecules:
            break
    return metrics

def calculate_lipinski(mol):
    passes_ro5 = all([
        Descriptors.MolWt(mol) <= 500,
        Descriptors.MolLogP(mol) <= 5,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10
    ])
    return passes_ro5

def plot_metrics(metric_name, metrics1, metrics2=None):
    plot_metrics_by_target(metric_name, metrics1, metrics2)
    plot_metrics_by_size(metric_name, metrics1, metrics2)

def plot_metrics_by_target(metric_name, metrics1, metrics2=None):
    plt.figure(figsize=(2, 1.3))
    
    # Prepare data for violin plot
    data = []
    
    # Process first set of metrics
    metrics_by_target1 = {}
    for pdb_id, m1 in metrics1.items():
        for size, m2 in m1.items():
            for metric in m2:
                metrics_by_target1.setdefault(pdb_id, []).append(metric)
    
    # Create DataFrame for first set
    df1 = pd.DataFrame({
        'value': [val for sublist in metrics_by_target1.values() for val in sublist],
        'group': 'Prediction'
    })
    data.append(df1)
    
    # Process second set of metrics if provided
    if metrics2 is not None:
        metrics_by_target2 = {}
        for pdb_id, m1 in metrics2.items():
            for size, m2 in m1.items():
                for metric in m2:
                    metrics_by_target2.setdefault(pdb_id, []).append(metric)
        
        # Create DataFrame for second set
        df2 = pd.DataFrame({
            'value': [val for sublist in metrics_by_target2.values() for val in sublist],
            'group': 'Original'
        })
        data.append(df2)
    
    # Combine all data
    combined_df = pd.concat(data)
    
    # Create violin plot
    ax = sns.violinplot(
        x='group' if metrics2 is not None else None,
        y='value',
        hue='group' if metrics2 is not None else None,
        data=combined_df,
        palette=['#8e7fbb', '#a2c9ae'] if metrics2 is not None else ['#8e7fbb'],
        cut=0,
        scale='width',
        inner=None,
        linewidth=1,
        edgecolor='black'
    )
    
    # Customize plot appearance
    plt.xlabel('')
    plt.ylabel(metric_name)
    
    # Remove top and right spines
    sns.despine()
    
    # plt.tight_layout()
    plt.show()
    plt.clf()

def plot_metrics_by_size(metric_name, metrics1, metrics2=None):
    # plot the average validity vs the size
    plt.figure(figsize=(2,1.3))

    metrics_by_size1 = {}
    metrics_by_size2 = {}

    for pdb_id, sizes in metrics1.items():
        for size, metrics in sizes.items():
            for metric in metrics:
                metrics_by_size1.setdefault(size, []).append(metric)
    # sort the sizes
    x1 = sorted(list(metrics_by_size1.keys()))
    y1 = [np.mean(metrics_by_size1[size]) for size in x1]
    yerr1 = [np.std(metrics_by_size1[size]) / np.sqrt(len(metrics_by_size1[size])) for size in x1]
    # plot line with error bars
    plt.plot(x1, y1, color='#6957a3', linewidth=1.5)
    # #8e7fbb between x axis and the line

    if metrics2 is not None:
        for pdb_id, sizes in metrics2.items():
            for size, metrics in sizes.items():
                for metric in metrics:
                    metrics_by_size2.setdefault(size, []).append(metric)

        x2 = sorted(list(metrics_by_size2.keys()))
        y2 = [np.mean(metrics_by_size2[size]) for size in x2]
        yerr2 = [np.std(metrics_by_size2[size]) / np.sqrt(len(metrics_by_size2[size])) for size in x2]
        plt.plot(x2, y2, color='#275317', linewidth=1.5)

    xmin = min(min(x1), min(x2)) if metrics2 is not None else min(x1)
    xmax = max(max(x1), max(x2)) if metrics2 is not None else max(x1)
    ymin = min(min(y1), min(y2)) if metrics2 is not None else min(y1)
    ymax = max(max(y1), max(y2)) if metrics2 is not None else max(y1)

    plt.fill_between(x1, ymin, y1, color='#8e7fbb', alpha=0.5)
    if metrics2 is not None:
        plt.fill_between(x2, ymin, y2, color='#a2c9ae', alpha=0.5)

    # set x ticks to be the sizes, and there are no periods in the sizes
    xticks_bin = (xmax - xmin) / 5
    xticks = np.arange(xmin, xmax+xticks_bin, xticks_bin)
    plt.xticks(xticks, [str(int(size)) for size in xticks])

    yticks_bin = (ymax - ymin) / 5
    yticks = np.arange(ymin, ymax+yticks_bin, yticks_bin)
    plt.yticks(yticks, [f"{y:.3f}" for y in yticks])

    plt.xlabel('Compound Size')
    plt.ylabel(f'{metric_name}')
    # plt.tight_layout()
    plt.show()
    plt.clf()

def save_metrics(metrics, filename):
    with open(f'analysis/{filename}.pkl', 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(filename):
    with open(f'analysis/{filename}.pkl', 'rb') as f:
        return pickle.load(f)

def save_metrics_to_csv(metrics, output_dir='analysis/metrics_csv'):
    """
    Save metrics to CSV files with proper organization and metadata.
    
    Args:
        metrics (dict): Dictionary containing metrics data
        output_dir (str): Directory to save CSV files
    """
    import os
    import pandas as pd
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process each metric type
    for metric_name, metric_data in metrics.items():
        # Create a list to store all rows
        rows = []
        
        # Process data for each target and size
        for target, size_data in metric_data.items():
            for size, values in size_data.items():
                for value in values:
                    rows.append({
                        'target': target,
                        'size': size,
                        'value': value,
                        'metric_type': metric_name
                    })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'{metric_name}_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f'Saved {metric_name} metrics to {output_file}')
        
        # Also save a summary statistics file
        summary_df = df.groupby(['target', 'size'])['value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        summary_file = os.path.join(output_dir, f'{metric_name}_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f'Saved {metric_name} summary statistics to {summary_file}')

# Example usage:
# save_metrics_to_csv({
#     'validity': validity_metrics,
#     'qed': qed_metrics,
#     'sas': sas_metrics,
#     'lipinski': lipinski_metrics
# })

#%%
folder_path = "test_results/MOAD_test"
# qeds = calculate_metrics(folder_path, callback=calculate_qed)
# sas = calculate_metrics(folder_path, callback=calculate_sas)
# sas = calculate_metrics(folder_path, callback=calculate_sas, max_files=10, ignore_errors=False)
lipinski = calculate_metrics(scan_folder(folder_path), callback=calculate_lipinski)
# lipinski = calculate_metrics(folder_path, callback=calculate_lipinski, max_files=100, ignore_errors=False)
#%%
# make a `analysis` folder
# save qeds with pickle
os.makedirs('analysis', exist_ok=True)

save_metrics(qeds, 'qeds')
plot_metrics(qeds, 'QED')

save_metrics(sas, 'sas')
plot_metrics(sas, 'SAS')

save_metrics(lipinski, 'lipinski')
plot_metrics(lipinski, 'Lipinski')

# plot_metrics(connects, 'Connectivity')
# %%
import pandas as pd

folder_path = "test_results/MOAD_test"
summary_files = [file for file in os.listdir(folder_path) if file.endswith(".csv") and file.startswith("summary")]
print('Found', len(summary_files), 'summary files')
valid_metrics = {}
connect_metrics = {}
large_rings_metrics = {}
for summary_file in tqdm(summary_files, total=len(summary_files)):
    # read the summary file
    # summary_size13_n10_s24411.csv
    m = re.match(r"summary_size(\d+)_n(\d+)_s(\d+)\.csv", summary_file)
    if m:
        size = int(m.group(1))
        nsamples = int(m.group(2))
        seed = int(m.group(3))
        df = pd.read_csv(os.path.join(folder_path, summary_file))
        # get the line that starts with "Total valid"
        for _, row in df.iterrows():
            name = row['name']
            invalid = row['invalid']
            unconnected = row['unconnected']
            large_rings = row['large_rings']
            attempts = row['attempts']
            valid_rate = 1 - invalid/attempts
            connect_rate = 1 - unconnected/attempts
            large_rings_rate = large_rings/attempts
            valid_metrics.setdefault(name, {}).setdefault(size, []).append(valid_rate)
            connect_metrics.setdefault(name, {}).setdefault(size, []).append(connect_rate)
            large_rings_metrics.setdefault(name, {}).setdefault(size, []).append(large_rings_rate)
# %%
save_metrics(valid_metrics, 'validity')
# plot_metrics(valid_metrics, 'Validity')

save_metrics(connect_metrics, 'connectivity')
# plot_metrics(connect_metrics, 'Connectivity')

save_metrics(large_rings_metrics, 'large_ring_rate')
# plot_metrics(large_rings_metrics, 'Large Ring Rate')
# %%

# read datasets/MOAD_test.pt with torch, get the names to a list
import torch
data = torch.load('datasets/MOAD_test.pt', map_location=torch.device('cpu'))
names = [row['name'] for row in data]
print(len(names), names[:10])

#%%
# read data/MOAD/processed/conformers.sdf, extract sdf blocks with the names
import rdkit.Chem as Chem

suppl = Chem.SDMolSupplier('data/MOAD/processed/conformers.sdf', sanitize=False, strictParsing=False)
outfile = 'analysis/MOAD_test_original.sdf'
with open(outfile, 'w') as f:
    writer = Chem.SDWriter(f)
    for mol in suppl:
        if mol is None:
            continue
        name = mol.GetProp('_Name')
        if name in names:
            writer.write(mol)
    writer.close()

# %%

# original_validity = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_validity)
# original_connectivity = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_connectivity)
# original_large_rings = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_large_rings)
original_qed = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_qed)
original_sas = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_sas)
original_lipinski = calculate_metrics(scan_sdf('analysis/MOAD_test_original.sdf'), callback=calculate_lipinski)
# %%
save_metrics(original_qed, 'original_qed')
save_metrics(original_sas, 'original_sas')
save_metrics(original_lipinski, 'original_lipinski')
# %%
qeds = load_metrics('qeds')
sas = load_metrics('sas')
lipinski = load_metrics('lipinski')
plot_metrics('QED', qeds, original_qed)
plot_metrics('SAS', sas, original_sas)
plot_metrics('Lipinski', lipinski, original_lipinski)

# %%
validity = load_metrics('validity')
connectivity = load_metrics('connectivity')
large_ring_rate = load_metrics('large_ring_rate')
plot_metrics('Validity', validity)
plot_metrics('Connectivity', connectivity)
plot_metrics('Large Ring Rate', large_ring_rate)
# %%
