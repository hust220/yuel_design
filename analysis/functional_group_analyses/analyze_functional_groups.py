#%%

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import psycopg2
from contextlib import contextmanager
import time, os
from multiprocessing import Pool
import io, sys
from typing import List, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool
import json
sys.path.append('../..')
from db_utils import db_connection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit.RDLogger import DisableLog
DisableLog('rdApp.*')

BATCH_SIZE = 20
NUM_PROCESSES = 16

# TABLE_NAME = 'ligands'
# ID_COLUMN = 'id'
# SDF_COLUMN = 'mol'
# GROUP_COLUMN = 'functional_groups'

TABLE_NAME = 'molecules'
ID_COLUMN = 'id'
SDF_COLUMN = 'sdf2'
GROUP_COLUMN = 'functional_groups'

# 1. Common functional groups SMARTS
functional_groups = {
    # 酸、醛、酮等 Acids, aldehydes, ketones, etc.
    'Carboxylic Acid': Chem.MolFromSmarts('C(=O)[OH]'),
    'Ester': Chem.MolFromSmarts('C(=O)O*'),
    'Amide': Chem.MolFromSmarts('C(=O)N'),
    'Ketone': Chem.MolFromSmarts('C(=O)C'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)C'),

    # 含氮官能团 Nitrogen functional groups
    'Amine (Primary/Secondary)': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
    'Amine (Tertiary)': Chem.MolFromSmarts('[NX3]([#6])([#6])[#6]'),
    'Nitrile': Chem.MolFromSmarts('[C;!R]#[N]'),

    # 含氧官能团 Oxygen functional groups
    'Alcohol': Chem.MolFromSmarts('[CX4][OH]'),
    # 'Phenol': Chem.MolFromSmarts('c[OH]'),
    'Phenol': Chem.MolFromSmarts('C1CCCCC1[OH]'), # Sanitized version
    'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'Epoxide': Chem.MolFromSmarts('[C;R]1[O][C;R]1'),
    
    # 含硫官能团 Sulfur functional groups
    'Thiol': Chem.MolFromSmarts('[#16H1]'),
    'Thioether': Chem.MolFromSmarts('[#16]([#6])[#6]'),
    'Sulfonamide': Chem.MolFromSmarts('S(=O)(=O)N'),

    # 卤素 Halogen
    'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),

    # 芳香环结构 Aromatic ring structures
    'Benzene': Chem.MolFromSmarts('C1CCCCC1'),
    'Pyridine': Chem.MolFromSmarts('N1CCCCC1'),
    'Pyrimidine': Chem.MolFromSmarts('C1CNCNC1'),
    'Imidazole': Chem.MolFromSmarts('C1CNC[NH]1'),
    'Indole': Chem.MolFromSmarts('C1CC2CCCC2[NH]1'),
    'Furan': Chem.MolFromSmarts('C1CCCO1'),
    'Thiophene': Chem.MolFromSmarts('C1CCSC1'),
    'Oxazole': Chem.MolFromSmarts('C1COCNC1'),

    # 特殊环类 Special ring structures
    'Cyclopropane': Chem.MolFromSmarts('C1CC1'),
    'Cyclobutane': Chem.MolFromSmarts('C1CCC1'),
}

# Add descriptions for each functional group
functional_group_descriptions = {
    'Carboxylic Acid': 'Organic compound containing a carboxyl group (-COOH), commonly found in amino acids and fatty acids',
    'Ester': 'Organic compound formed by the reaction of an acid with an alcohol, characterized by -COO- linkage',
    'Amide': 'Organic compound containing a carbonyl group (C=O) linked to a nitrogen atom, common in proteins and peptides',
    'Ketone': 'Organic compound containing a carbonyl group (C=O) bonded to two carbon atoms',
    'Aldehyde': 'Organic compound containing a carbonyl group (C=O) bonded to at least one hydrogen atom',
    'Amine (Primary/Secondary)': 'Organic compound containing nitrogen with one or two alkyl/aryl groups attached',
    'Amine (Tertiary)': 'Organic compound containing nitrogen with three alkyl/aryl groups attached',
    'Nitrile': 'Organic compound containing a cyano group (-C≡N)',
    'Alcohol': 'Organic compound containing a hydroxyl group (-OH) attached to a carbon atom',
    'Phenol': 'Aromatic compound containing a hydroxyl group (-OH) directly attached to a benzene ring',
    'Ether': 'Organic compound containing an oxygen atom connected to two alkyl or aryl groups',
    'Epoxide': 'Cyclic ether with a three-membered ring containing an oxygen atom',
    'Thiol': 'Organic compound containing a sulfhydryl group (-SH)',
    'Thioether': 'Organic compound containing a sulfur atom connected to two alkyl or aryl groups',
    'Sulfonamide': 'Organic compound containing a sulfonyl group (-SO2-) linked to an amine',
    'Halogen': 'Element from group 17 (F, Cl, Br, I) that can form single bonds with carbon',
    'Benzene': 'Aromatic hydrocarbon with a six-membered ring containing alternating double bonds',
    'Pyridine': 'Heterocyclic aromatic compound with a nitrogen atom in a six-membered ring',
    'Pyrimidine': 'Heterocyclic aromatic compound with two nitrogen atoms in a six-membered ring',
    'Imidazole': 'Heterocyclic aromatic compound with two nitrogen atoms in a five-membered ring',
    'Indole': 'Heterocyclic aromatic compound containing a benzene ring fused to a pyrrole ring',
    'Furan': 'Heterocyclic aromatic compound with an oxygen atom in a five-membered ring',
    'Thiophene': 'Heterocyclic aromatic compound with a sulfur atom in a five-membered ring',
    'Oxazole': 'Heterocyclic aromatic compound containing both oxygen and nitrogen in a five-membered ring',
    'Cyclopropane': 'Cyclic hydrocarbon with a three-membered carbon ring',
    'Cyclobutane': 'Cyclic hydrocarbon with a four-membered carbon ring'
}

def ensure_functional_groups_column_exists():
    """确保functional_groups列存在"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='{TABLE_NAME}' AND column_name='{GROUP_COLUMN}'
            """)
            if not cursor.fetchone():
                cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {GROUP_COLUMN} TEXT")
            conn.commit()

def get_unprocessed_molecule_ids() -> List[int]:
    """获取所有functional_groups为空的分子ID"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # cursor.execute("""
            #     SELECT id FROM molecules 
            #     WHERE functional_groups IS NULL 
            #     ORDER BY id 
            # """)
            cursor.execute(f"""
                SELECT {ID_COLUMN} FROM {TABLE_NAME}
            """)
            return [row[0] for row in cursor.fetchall()]

def process_molecule_batch(mol_ids: List[int]) -> List[Tuple[int, str]]:
    """处理一批分子并返回结果"""
    results = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # 获取这批分子的SDF数据
            cursor.execute(f"""
                SELECT {ID_COLUMN}, {SDF_COLUMN} FROM {TABLE_NAME} 
                WHERE {ID_COLUMN} = ANY(%s)
            """, (mol_ids,))
            
            for mol_id, sdf_data in cursor.fetchall():
                try:
                    # 从字节流读取SDF
                    sdf_io = io.BytesIO(sdf_data.tobytes())
                    suppl = Chem.ForwardSDMolSupplier(sdf_io, sanitize=False, strictParsing=False)
                    # suppl = Chem.ForwardSDMolSupplier(sdf_io, strictParsing=False)
                    mol = next(suppl, None)
                    
                    if not mol:
                        # raise Exception(f"Molecule {mol_id} is None")
                        continue
                        
                    # 检测功能基团
                    # print(Chem.MolToSmarts(mol))
                    # print smiles
                    # print(Chem.MolToSmiles(mol))
                    group_counts = {}
                    for name, pattern in functional_groups.items():
                        matches = mol.GetSubstructMatches(pattern)
                        group_counts[name] = len(matches)
                        
                    # 格式化结果字符串
                    result_str = ",".join(f"{k}:{v}" for k, v in group_counts.items() if v > 0)
                    results.append((mol_id, result_str))
                    
                    # 立即更新数据库
                    cursor.execute(
                        f"UPDATE {TABLE_NAME} SET {GROUP_COLUMN} = %s WHERE {ID_COLUMN} = %s",
                        (result_str, mol_id)
                    )
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error processing molecule {mol_id}: {str(e)}")
                    conn.rollback()
                    # raise e
                    continue
                    
    return results

def process_molecules_parallel() -> List[Tuple[int, str]]:
    all_mol_ids = get_unprocessed_molecule_ids()  # 预取10批量的ID
    total_molecules = len(all_mol_ids)
    
    if not total_molecules:
        print("没有需要处理的分子")
        return []
    
    print(f"共发现 {total_molecules} 个待处理分子")
    
    batches = [all_mol_ids[i:i + BATCH_SIZE] for i in range(0, total_molecules, BATCH_SIZE)]
    with Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(batches), desc="处理进度", unit="batch") as pbar:
            results = []
            for batch_result in pool.imap_unordered(process_molecule_batch, batches):
                results.extend(batch_result)
                pbar.update(1)  # 更新进度条
                
                pbar.set_postfix({
                    '已处理分子': len(results),
                    '剩余分子': total_molecules - len(results)
                })
    
    processed_count = len(results)
    if processed_count < total_molecules:
        print(f"警告: 只成功处理了 {processed_count}/{total_molecules} 个分子")
    else:
        print(f"成功处理了所有 {processed_count} 个分子")
    
    return results

def generate_statistics(results: List[Tuple[int, str]]):
    """生成统计信息"""
    summary = defaultdict(int)
    for _, functional_groups_str in results:
        if not functional_groups_str:
            continue

        # print(functional_groups_str)
            
        # 解析功能基团字符串
        parts = functional_groups_str.split(',')
        for part in parts:
            group, count = part.split(':')
            summary[group] += int(count)
            
    return summary

def visualize_statistics(summary: dict, total_molecules: int):
    """可视化统计结果"""
    df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
    df['Fraction'] = df['Count'] / total_molecules
    
    df.sort_values('Fraction', ascending=True).plot(
        kind='barh', 
        xlim=(0,1), 
        figsize=(4,3), 
        legend=False,
        color='#8e7fb8',
        width=0.8
    )
    plt.xlabel('Fraction of Molecules')
    plt.ylabel('Functional Group')
    plt.tight_layout()
    plt.savefig(f'figures/{TABLE_NAME}_functional_group_diversity.svg', format='svg')
    plt.show()

def get_frequency_summary():
    """Retrieve and summarize functional group data from the database"""
    group_counts = defaultdict(int)
    total_molecules = 0
    
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Get total number of molecules
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            total_molecules = cursor.fetchone()[0]
            
            # Get all functional group data
            cursor.execute(f"SELECT {GROUP_COLUMN} FROM {TABLE_NAME} WHERE {GROUP_COLUMN} IS NOT NULL")
            
            for row in cursor.fetchall():
                functional_groups_str = row[0]
                if not functional_groups_str:
                    continue
                    
                # Parse the functional groups string
                parts = functional_groups_str.split(',')
                for part in parts:
                    try:
                        group, count = part.split(':')
                        if int(count) > 0:
                            group_counts[group] += 1
                    except ValueError:
                        continue
    
    return group_counts, total_molecules

def print_frequency_summary(group_counts, total_molecules):
    """Print a formatted summary of functional group counts"""
    print("\nFunctional Group Summary:")
    print("=" * 50)
    print(f"{'Functional Group':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    
    sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
    
    for group, count in sorted_groups:
        percentage = (count / total_molecules) * 100
        print(f"{group:<30} {count:<10} {percentage:.2f}%")
    
    print("=" * 50)
    print(f"Total molecules analyzed: {total_molecules}\n")

def analyze_frequency():
    # Get the summary from the database
    group_counts, total_molecules = get_frequency_summary()
    
    # Print the summary
    print_frequency_summary(group_counts, total_molecules)
    
    # Save to JSON file
    with open(f'metrics/{TABLE_NAME}_functional_group_summary.json', 'w') as f:
        json.dump(group_counts, f, indent=4)
    
    # Visualize the results
    visualize_statistics(group_counts, total_molecules)


def calculate_functional_groups():
    # 确保列存在
    ensure_functional_groups_column_exists()
    
    # 并行处理分子
    results = process_molecules_parallel()
    print(f"Processed {len(results)} molecules successfully")
    
    # 生成统计信息
    summary = generate_statistics(results)
    sorted_summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    for group, count in sorted_summary:
        print(f"{group}: {count}")
    # beautiful print
    # print(json.dumps(summary, indent=4))
    # dump to file
    with open(f'metrics/{TABLE_NAME}_functional_group_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    # 可视化
    visualize_statistics(summary, len(results))

# data1 and data2 are obtained from the running resuls of analyze_frequency()
# Data from Table 1 (68047 molecules)
data1 = {
    "Functional Group": [
        "Alcohol", "Amine (Primary/Secondary)", "Ether", "Benzene", 
        "Amine (Tertiary)", "Pyrimidine", "Furan", "Imidazole", 
        "Halogen", "Pyridine", "Thioether", "Phenol", 
        "Thiol", "Thiophene", "Cyclopropane", "Epoxide", 
        "Cyclobutane", "Oxazole", "Indole"
    ],
    "Percentage (Table 1)": [
        88.31, 71.25, 54.28, 34.87, 34.85, 28.33, 20.68, 17.89,
        11.60, 11.27, 8.84, 7.91, 2.81, 2.09, 1.03, 0.27, 
        0.23, 0.07, 0.01
    ]
}

# Data from Table 2 (210000 molecules)
data2 = {
    "Functional Group": [
        "Alcohol", "Amine (Primary/Secondary)", "Ether", "Ketone", "Amide", "Benzene", "Amine (Tertiary)", "Pyridine",
        "Ester", "Cyclopropane", "Thioether", "Furan", "Phenol", "Pyrimidine", "Epoxide", "Thiol", "Halogen",
        "Cyclobutane", "Nitrile", "Thiophene", "Imidazole", "Oxazole", "Carboxylic Acid", "Aldehyde", "Sulfonamide", "Indole"
    ],
    "Percentage (Table 2)": [
        72.75, 55.26, 48.23, 46.79, 23.16, 20.36, 20.13, 17.76,
        12.12, 12.02, 10.70, 9.52, 7.29, 5.49, 5.43, 4.88, 4.67,
        2.99, 2.05, 2.03, 1.57, 1.43, 0.72, 0.65, 0.61, 0.25
    ]
}

def plot_functional_group_comparison():
    """Plot the functional group comparison between Table 1 and Table 2"""

    dir_path = 'functional_groups_plots'
    os.makedirs(dir_path, exist_ok=True)

    # Create dataframes
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Merge the two dataframes on "Functional Group"
    merged_df = pd.merge(df1, df2, on="Functional Group", how="inner")

    # Sort by Table 1 percentage (descending)
    merged_df = merged_df.sort_values(by="Percentage (Table 1)", ascending=False)

    # Control how many lowest ranked groups to show in the zoomed-in plot
    N_LOWEST_GROUPS = 9  # Change this value as needed

    # Plotting
    plt.figure(figsize=(5, 3))
    bar_width = 0.35
    index = np.arange(len(merged_df))

    bars1 = plt.bar(index, merged_df["Percentage (Table 1)"], bar_width, label="Original", color='#a2c9ae')
    bars2 = plt.bar(index + bar_width, merged_df["Percentage (Table 2)"], bar_width, label="YuelDesign", color='#8e7fb8')

    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(index + bar_width/2, merged_df["Functional Group"], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir_path}/functional_group_comparison.svg', format='svg')
    plt.show()

    # Plot for the last N_LOWEST_GROUPS ranked groups (lowest percentages in Table 1)
    lastN = merged_df.tail(N_LOWEST_GROUPS)
    plt.figure(figsize=(4, 3))
    indexN = np.arange(len(lastN))

    bars1_N = plt.bar(indexN, lastN["Percentage (Table 1)"], bar_width, label="Original", color='#a2c9ae')
    bars2_N = plt.bar(indexN + bar_width, lastN["Percentage (Table 2)"], bar_width, label="YuelDesign", color='#8e7fb8')

    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(indexN + bar_width/2, lastN["Functional Group"], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir_path}/functional_group_comparison_last{N_LOWEST_GROUPS}.svg', format='svg')
    plt.show()

def save_tables():
    """Save functional group analysis results to a TSV file using existing data1 and data2"""
    # Create DataFrames from existing data
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Merge the two dataframes on "Functional Group"
    merged_df = pd.merge(df1, df2, on="Functional Group", how="outer")
    
    # Rename columns to match required format
    merged_df = merged_df.rename(columns={
        "Percentage (Table 1)": "Percentage in Native Ligands",
        "Percentage (Table 2)": "Percentage in YuelDesign-generated molecules"
    })
    
    # Add SMARTS patterns and descriptions
    merged_df['SMARTS Pattern'] = merged_df['Functional Group'].apply(
        lambda x: Chem.MolToSmarts(functional_groups[x]) if x in functional_groups else ""
    )
    merged_df['Description'] = merged_df['Functional Group'].apply(
        lambda x: functional_group_descriptions.get(x, '')
    )
    
    # Format percentage columns
    for col in ['Percentage in Native Ligands', 'Percentage in YuelDesign-generated molecules']:
        merged_df[col] = merged_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
    
    # Sort by Functional Group
    merged_df = merged_df.sort_values('Functional Group')
    
    # Ensure tables directory exists
    os.makedirs('tables', exist_ok=True)
    
    # Save to TSV file
    merged_df.to_csv('tables/functional_groups.tsv', sep='\t', index=False)
    print(f"Functional group analysis results saved to tables/functional_groups.tsv")

# %%
# calculate_functional_groups()
# analyze_frequency()
plot_functional_group_comparison()
# save_tables()

# %%
