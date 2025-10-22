#%%
import sys
sys.path.append('..')
sys.path.append('../..')

from analysis.pick_best_generation import (
    analyze_similarity,
    print_cluster_info
)

import pandas as pd
from db_utils import db_connection
from rdkit import Chem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit import DataStructs
from io import BytesIO
import os
import time
import numpy as np
from sklearn.cluster import KMeans

def get_predicted_molecules(target_name):
    """
    获取docking表中指定target的所有generated molecules，并将native ligand从{target}_ligand.sdf文件读取，返回DataFrame。
    字段包括id、title、ligand（结构）、ligand_format、size、is_native、is_best_score。
    """
    from rdkit.Chem import SDMolSupplier

    with db_connection() as conn:
        # 获取所有generated molecules（不含native）
        gen_query = """
            SELECT id, title, ligand, ligand_format, ligand_size, best_score
            FROM docking
            WHERE title LIKE %s AND title != %s AND docking_status = 2
            ORDER BY best_score ASC
        """
        gen_df = pd.read_sql_query(gen_query, conn, params=(f"{target_name}%", f"{target_name}_native"))
        gen_df['is_native'] = False
        # 标记best_score最低的为is_best_score
        if not gen_df.empty:
            min_score_idx = gen_df['best_score'].idxmin()
            gen_df['is_best_score'] = False
            gen_df.loc[min_score_idx, 'is_best_score'] = True
        else:
            gen_df['is_best_score'] = False
        gen_df = gen_df.rename(columns={'ligand_size': 'size'})

    # 读取native ligand sdf文件
    sdf_path = f"{target_name}_ligand.sdf"
    if os.path.exists(sdf_path):
        with open(sdf_path, 'r') as f:
            mol_block = f.read()
        # 用RDKit解析，获取分子大小
        mol = Chem.MolFromMolBlock(mol_block, sanitize=False, strictParsing=False)
        size = mol.GetNumAtoms() if mol is not None else None
        native_df = pd.DataFrame([{
            'id': -1,
            'title': f'{target_name}_native',
            'ligand': mol_block,
            'ligand_format': 'sdf',
            'size': size,
            'is_native': True,
            'is_best_score': False,
            'best_score': None
        }])
    else:
        print(f"Warning: native ligand file {sdf_path} not found!")
        native_df = pd.DataFrame(columns=['id','title','ligand','ligand_format','size','is_native','is_best_score','best_score'])

    # 合并
    df = pd.concat([native_df, gen_df], ignore_index=True)
    return df

def analyze_pharmacophore_fingerprints(molecules_df):
    """
    Analyze 2D pharmacophore fingerprints from molecules (docking表，ligand字段为bytes或字符串，ligand_format区分mol2/sdf)。
    Returns a dictionary of molecule IDs and their fingerprints.
    """
    print(f"\nAnalyzing {len(molecules_df)} molecules...")
    fingerprints = {}
    for _, row in molecules_df.iterrows():
        try:
            ligand_data = row['ligand']
            ligand_format = row.get('ligand_format', 'sdf').lower() if hasattr(row, 'get') else row['ligand_format'].lower()
            if isinstance(ligand_data, memoryview):
                ligand_data = ligand_data.tobytes()
            if isinstance(ligand_data, bytes):
                mol_block = ligand_data.decode('utf-8')
            else:
                mol_block = ligand_data
            if ligand_format == 'mol2':
                mol = Chem.MolFromMol2Block(mol_block, sanitize=False, removeHs=False)
            else:
                mol = Chem.MolFromMolBlock(mol_block, sanitize=False, strictParsing=False)
            if mol is None:
                print(f"Warning: failed to parse molecule {row['id']} (format: {ligand_format})")
                continue
            fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
            fingerprints[row['id']] = {
                'fingerprint': fp,
                'size': row['size']
            }
            print(f"Processed molecule {row['id']} (format: {ligand_format})")
        except Exception as e:
            print(f"Error processing molecule {row['id']}: {str(e)}")
            continue
    return fingerprints

def cluster(similarity_df, n_clusters=10):
    """
    Cluster molecules based on similarity matrix using KMeans.
    Returns a dictionary mapping molecule IDs to cluster assignments.
    每次使用不同的随机种子。
    """
    print("\nClustering molecules using KMeans...")
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_df.values
    # 用当前时间戳作为随机种子
    seed = int(time.time() * 1000) % (2**32 - 1)
    print(f"KMeans random_state: {seed}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    clusters = dict(zip(similarity_df.index, cluster_labels))
    print(f"Created {n_clusters} clusters using KMeans")
    return clusters

def plot_clusters(clusters, similarities, molecules_df, output_file=None):
    """
    Plot clusters using PCA for dimensionality reduction.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial import ConvexHull
    import numpy as np
    
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarities.values
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    coords = pca.fit_transform(distance_matrix)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'cluster': [clusters[idx] for idx in similarities.index],
        'size': [molecules_df[molecules_df['id'] == idx]['size'].iloc[0] for idx in similarities.index],
        'is_native': [molecules_df[molecules_df['id'] == idx]['is_native'].iloc[0] for idx in similarities.index],
        'is_best_score': [molecules_df[molecules_df['id'] == idx]['is_best_score'].iloc[0] for idx in similarities.index]
    })
    
    # Create figure
    plt.figure(figsize=(4, 3))
    
    # Plot clusters with convex hulls
    for cluster_id in sorted(plot_df['cluster'].unique()):
        cluster_points = plot_df[plot_df['cluster'] == cluster_id]
        
        # Plot points
        plt.scatter(
            cluster_points['x'],
            cluster_points['y'],
            s=10,  # Reduced point size
            alpha=0.6,
            label=None,  # Remove cluster labels from legend
            edgecolor='none'  # Remove point edges
        )
        
        # Add convex hull with fill
        if len(cluster_points) >= 3:  # Need at least 3 points for a convex hull
            points = cluster_points[['x', 'y']].values
            hull = ConvexHull(points)
            
            # Plot filled convex hull
            plt.fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                alpha=0.1,  # Very transparent fill
                edgecolor='none'  # No edge for the fill
            )
    
    # Highlight native ligand and best molecule
    native_point = plot_df[plot_df['is_native']]
    best_point = plot_df[plot_df['is_best_score']]
    
    if not native_point.empty:
        plt.scatter(
            native_point['x'],
            native_point['y'],
            s=80,  # Reduced size
            marker='o',
            color='#a2c9ae',  # 绿色
            label='Native Ligand',
            edgecolor='#5b7c6a',  # Darker green
            linewidths=1.2
        )
    
    if not best_point.empty:
        plt.scatter(
            best_point['x'],
            best_point['y'],
            s=80,  # Reduced size
            marker='o',
            color='#8e7fb8',  # 紫色
            label='Best Score',
            edgecolor='#5c4a91',  # Darker purple
            linewidths=1.2
        )
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Move legend to top without frame
    plt.legend(
        bbox_to_anchor=(0.5, 1.15),
        loc='center',
        fontsize='small',
        ncol=2,
        frameon=False  # Remove legend frame
    )
    plt.tight_layout()
    
    # Save plot
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"\nCluster visualization saved as '{output_file}'")
    plt.show()
    plt.close()
    


#%%
# target = '7e2y'
target = '7ckz'
df = get_predicted_molecules(target)
print(f"\nFound {len(df)} molecules with target {target} (including native ligand)")
fingerprints = analyze_pharmacophore_fingerprints(df)
similarities = analyze_similarity(fingerprints)
clusters = cluster(similarities, n_clusters=5)
print_cluster_info(clusters, df)
plot_clusters(clusters, similarities, df, output_file=f'{target}_pharmacophore_clusters.svg')

# %%
