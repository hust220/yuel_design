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
import Bio.PDB
from io import StringIO
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit import DataStructs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

def get_better_medusadock_results():
    """
    Get molecules where medusadock score is better than random docking score.
    Returns a pandas DataFrame with the results.
    """
    query = """
    SELECT 
        m.molecule_id,
        m.medusadock_score,
        r.best_score,
        (r.best_score - m.medusadock_score) as score_difference,
        mol.ligand_name,
        r.receptor
    FROM medusadock_results m 
    JOIN molecules mol ON m.molecule_id = mol.id 
    JOIN random_docking r ON mol.ligand_name = r.pocket AND r.pocket = r.ligand 
    WHERE m.medusadock_score < r.best_score 
    ORDER BY score_difference DESC;
    """
    
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    return df

def save_top_results_files(results_df, top_n=5):
    """
    Save PDB and molecular files for the top N results.
    Args:
        results_df: DataFrame with the results
        top_n: Number of top results to process
    """
    # Create output directory
    output_dir = "best_generation_structures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    # Query to get all necessary files
    query = """
    SELECT 
        m.molecule_id,
        m.medusadock_score,
        r.best_score,
        m.best_pose as prediction_pose,
        r.best_pose as native_pose,
        p.pdb as receptor_pdb,
        l.mol as ligand_mol,
        p.name as receptor_name,
        l.name as ligand_name
    FROM medusadock_results m
    JOIN molecules mol ON m.molecule_id = mol.id
    JOIN random_docking r ON mol.ligand_name = r.pocket AND r.pocket = r.ligand
    JOIN proteins p ON r.receptor = p.name
    JOIN ligands l ON r.ligand = l.name AND r.pocket = l.name
    WHERE m.molecule_id = ANY(%s)
    """
    
    with db_connection() as conn:
        files_df = pd.read_sql_query(query, conn, params=(top_results['molecule_id'].tolist(),))
    
    # Save files
    for _, row in files_df.iterrows():
        # Create subdirectory for this result
        result_dir = os.path.join(output_dir, f"molecule_{row['molecule_id']}_medusa{row['medusadock_score']:.2f}_random{row['best_score']:.2f}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save receptor PDB
        with open(os.path.join(result_dir, f"receptor_{row['receptor_name']}.pdb"), 'wb') as f:
            f.write(row['receptor_pdb'])
        
        # Save ligand MOL
        with open(os.path.join(result_dir, f"ligand_{row['ligand_name']}.mol"), 'wb') as f:
            f.write(row['ligand_mol'])
        
        # Save prediction pose
        if row['prediction_pose'] is not None:
            with open(os.path.join(result_dir, "prediction_pose.pdb"), 'wb') as f:
                f.write(row['prediction_pose'])
        
        # Save native pose
        if row['native_pose'] is not None:
            with open(os.path.join(result_dir, "native_pose.pdb"), 'wb') as f:
                f.write(row['native_pose'])
    
    print(f"\nSaved files for top {top_n} results in directory: {output_dir}")

def get_low_rmsd_large_molecules():
    """
    Get molecules that have RMSD less than 1 and size greater than 15.
    Returns a pandas DataFrame with the results.
    """
    query = """
    WITH unique_molecules AS (
        SELECT DISTINCT ON (m.molecule_id)
            m.molecule_id,
            m.rmsd,
            mol.size,
            mol.ligand_name,
            r.receptor
        FROM medusadock_results m
        JOIN molecules mol ON m.molecule_id = mol.id
        JOIN random_docking r ON mol.ligand_name = r.pocket
        WHERE m.rmsd < 1 
        AND mol.size > 15
        ORDER BY m.molecule_id, m.rmsd ASC
    )
    SELECT * FROM unique_molecules
    ORDER BY rmsd ASC;
    """
    
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    return df

def save_low_rmsd_files(results_df, top_n=5):
    # Create output directory
    output_dir = "best_generation_structures/low_rmsd_structures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    # Query to get all necessary files
    query = """
    SELECT 
        m.molecule_id,
        m.rmsd,
        mol.size,
        m.best_pose as docking_pose,
        mol.sdf2 as generation_pose,
        p.pdb as receptor_pdb,
        l.mol as ligand_mol,
        p.name as receptor_name,
        l.name as ligand_name
    FROM medusadock_results m
    JOIN molecules mol ON m.molecule_id = mol.id
    JOIN ligands l ON mol.ligand_name = l.name
    JOIN proteins p ON l.protein_name = p.name
    WHERE m.molecule_id = ANY(%s)
    """
    
    with db_connection() as conn:
        files_df = pd.read_sql_query(query, conn, params=(top_results['molecule_id'].tolist(),))
    
    # Save files
    for _, row in files_df.iterrows():
        # Create subdirectory for this result
        result_dir = os.path.join(output_dir, f"molecule_{row['molecule_id']}_rmsd{row['rmsd']:.2f}_size{row['size']}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save receptor PDB
        with open(os.path.join(result_dir, f"receptor_{row['receptor_name']}.pdb"), 'wb') as f:
            f.write(row['receptor_pdb'])
        
        # Save ligand MOL
        with open(os.path.join(result_dir, f"ligand_{row['ligand_name']}.mol"), 'wb') as f:
            f.write(row['ligand_mol'])
        
        # Save docking pose
        if row['docking_pose'] is not None:
            with open(os.path.join(result_dir, "docking_pose.pdb"), 'wb') as f:
                f.write(row['docking_pose'])
        
        # Save generation pose
        if row['generation_pose'] is not None:
            with open(os.path.join(result_dir, "generation_pose.sdf"), 'wb') as f:
                f.write(row['generation_pose'])
    
    print(f"\nSaved files for top {top_n} results in directory: {output_dir}")

def get_high_similarity_molecules(similarity_threshold=0.8):
    """
    Get molecules that have similarity greater than threshold.
    The similarity column contains space-separated numbers, we use the first number.
    Returns a pandas DataFrame with the results.
    """
    query = """
    SELECT 
        m.id as molecule_id,
        m.similarity,
        m.size,
        m.ligand_name
    FROM molecules m
    WHERE CAST(SPLIT_PART(m.similarity, ' ', 1) AS float) > %s
    ORDER BY CAST(SPLIT_PART(m.similarity, ' ', 1) AS float) DESC;
    """
    
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(similarity_threshold,))
    
    return df

def save_high_similarity_files(results_df, top_n=5):
    # Create output directory
    output_dir = "best_generation_structures/high_similarity_structures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    # Query to get all necessary files
    query = """
    SELECT 
        m.id as molecule_id,
        m.similarity,
        m.size,
        m.sdf2 as generation_pose,
        p.pdb as receptor_pdb,
        l.mol as ligand_mol,
        p.name as receptor_name,
        l.name as ligand_name
    FROM molecules m
    JOIN ligands l ON m.ligand_name = l.name
    JOIN proteins p ON l.protein_name = p.name
    WHERE m.id = ANY(%s)
    """
    
    with db_connection() as conn:
        files_df = pd.read_sql_query(query, conn, params=(top_results['molecule_id'].tolist(),))
    
    # Save files
    for _, row in files_df.iterrows():
        # Get first similarity value and format to 2 decimal places
        first_similarity = float(row['similarity'].split()[0])
        similarity_str = f"{first_similarity:.2f}"
        
        # Create subdirectory for this result
        result_dir = os.path.join(output_dir, f"molecule_{row['molecule_id']}_similarity{similarity_str}_size{row['size']}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save receptor PDB
        with open(os.path.join(result_dir, f"receptor_{row['receptor_name']}.pdb"), 'wb') as f:
            f.write(row['receptor_pdb'])
        
        # Save ligand MOL
        with open(os.path.join(result_dir, f"ligand_{row['ligand_name']}.mol"), 'wb') as f:
            f.write(row['ligand_mol'])
        
        # Save generation pose
        if row['generation_pose'] is not None:
            with open(os.path.join(result_dir, "generation_pose.sdf"), 'wb') as f:
                f.write(row['generation_pose'])
    
    print(f"\nSaved files for top {top_n} results in directory: {output_dir}")

def get_predicted_molecules(ligand_name):
    """
    Get molecules for a specific ligand name, including the native ligand.
    Returns a pandas DataFrame with the results.
    """
    # Get the native ligand from ligands table
    native_query = """
    SELECT 
        id,
        name as ligand_name,
        mol as sdf2,
        size,
        TRUE as is_native
    FROM ligands
    WHERE name = %s;
    """
    
    # Get the generated molecules
    generated_query = """
    WITH best_molecule AS (
        SELECT m.id as best_id
        FROM molecules m
        JOIN medusadock_results md ON m.id = md.molecule_id
        WHERE m.ligand_name = %s
        ORDER BY md.medusadock_score ASC
        LIMIT 1
    )
    SELECT 
        m.id,
        m.size,
        m.ligand_name,
        m.sdf2,
        FALSE as is_native,
        (m.id = bm.best_id) as is_best_score
    FROM molecules m
    JOIN ligands l ON m.ligand_name = l.name
    CROSS JOIN best_molecule bm
    WHERE l.name = %s
    ORDER BY m.size DESC;
    """
    
    with db_connection() as conn:
        native_df = pd.read_sql_query(native_query, conn, params=(ligand_name,))
        generated_df = pd.read_sql_query(generated_query, conn, params=(ligand_name, ligand_name))
    
    # Add is_best_score column to native_df (it's never the best score)
    native_df['is_best_score'] = False
    
    # Combine the dataframes
    combined_df = pd.concat([native_df, generated_df], ignore_index=True)
    return combined_df

def analyze_pharmacophore_fingerprints(molecules_df):
    """
    Analyze 2D pharmacophore fingerprints from molecules.
    Returns a dictionary of molecule IDs and their fingerprints.
    """
    print(f"\nAnalyzing {len(molecules_df)} molecules...")
    fingerprints = {}
    
    for _, row in molecules_df.iterrows():
        try:
            # Load molecule from SDF data
            sdf_data = BytesIO(row['sdf2'].tobytes())
            mol = next(Chem.ForwardSDMolSupplier(sdf_data, sanitize=False, strictParsing=False))
            
            if mol is None:
                continue
            
            # Generate 2D pharmacophore fingerprint using Gobbi's method
            fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
            
            # Store fingerprint
            fingerprints[row['id']] = {
                'fingerprint': fp,
                'size': row['size']
            }
            
            print(f"Processed molecule {row['id']}")
            
        except Exception as e:
            print(f"Error processing molecule {row['id']}: {str(e)}")
            continue
    
    return fingerprints

def analyze_similarity(fingerprints):
    """
    Calculate similarity matrix between all fingerprints.
    Returns a DataFrame with similarity scores.
    """
    print("\nCalculating similarities...")
    mol_ids = list(fingerprints.keys())
    n_mols = len(mol_ids)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_mols, n_mols))
    
    # Calculate similarities
    for i in range(n_mols):
        for j in range(i+1, n_mols):
            fp1 = fingerprints[mol_ids[i]]['fingerprint']
            fp2 = fingerprints[mol_ids[j]]['fingerprint']
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarity_matrix[i,j] = similarity
            similarity_matrix[j,i] = similarity
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=mol_ids,
        columns=mol_ids
    )
    
    return similarity_df

def cluster(similarity_df, n_clusters=10):
    """
    Cluster molecules based on similarity matrix using KMeans.
    Returns a dictionary mapping molecule IDs to cluster assignments.
    """
    print("\nClustering molecules using KMeans...")
    
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_df.values
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # Create dictionary mapping molecule IDs to cluster assignments
    clusters = dict(zip(similarity_df.index, cluster_labels))
    
    print(f"Created {n_clusters} clusters using KMeans")
    return clusters

def print_cluster_info(clusters, molecules_df):
    """
    Print detailed information about each cluster.
    """
    cluster_members = {}
    for mol_id, cluster_id in clusters.items():
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(mol_id)

    for cluster_id, members in sorted(cluster_members.items()):
        print(f"\nCluster {cluster_id}:")
        print(f"Number of molecules: {len(members)}")
        print("Molecule IDs:", members)
        
        # Calculate average size for this cluster
        cluster_sizes = molecules_df[molecules_df['id'].isin(members)]['size']
        print(f"Average size: {cluster_sizes.mean():.1f}")
        print(f"Size range: {cluster_sizes.min()} - {cluster_sizes.max()}")

def get_native_ligand_and_best_molecule(ligand_name):
    """
    Get the native ligand and the molecule with the lowest medusadock score.
    Returns their molecule IDs.
    """
    # Get native ligand
    native_query = """
    SELECT id FROM ligands 
    WHERE name = %s;
    """
    
    # Get molecule with lowest medusadock score
    best_query = """
    SELECT m.id 
    FROM molecules m
    JOIN medusadock_results md ON m.id = md.molecule_id
    WHERE m.ligand_name = %s
    ORDER BY md.medusadock_score ASC
    LIMIT 1;
    """
    
    with db_connection() as conn:
        native_id = pd.read_sql_query(native_query, conn, params=(ligand_name,))['id'].iloc[0]
        best_id = pd.read_sql_query(best_query, conn, params=(ligand_name,))['id'].iloc[0]
    
    return native_id, best_id

def plot_clusters(clusters, similarities, molecules_df):
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
            s=cluster_points['size'],  # Reduced point size
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
            marker='*',
            color='#8e7fb8',
            label='Native Ligand',
            edgecolor='#5c4a91',  # Darker purple
            linewidths=1.2
        )
    
    if not best_point.empty:
        plt.scatter(
            best_point['x'],
            best_point['y'],
            s=80,  # Reduced size
            marker='*',
            color='#a2c9ae',
            label='Best Score',
            edgecolor='#5b7c6a',  # Darker green
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
    plt.savefig('molecule_clusters.svg', bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("\nCluster visualization saved as 'molecule_clusters.png'")

#%%

# results_df = get_better_medusadock_results()
# print(f"Found {len(results_df)} molecules where medusadock performed better than random docking")
# print("\nTop 10 best improvements:")
# print(results_df.head(10))
# save_top_results_files(results_df, top_n=5)

# low_rmsd_df = get_low_rmsd_large_molecules()
# print(f"\nFound {len(low_rmsd_df)} molecules with RMSD < 1 and size > 15")
# print("\nTop 10 lowest RMSD molecules:")
# print(low_rmsd_df.head(10))
# save_low_rmsd_files(low_rmsd_df, top_n=5)

# Get high similarity molecules
# high_similarity_df = get_high_similarity_molecules(0.4)
# print(f"\nFound {len(high_similarity_df)} molecules with similarity > 0.4")
# print("\nTop 10 highest similarity molecules:")
# print(high_similarity_df.head(10))
# save_high_similarity_files(high_similarity_df, top_n=5)

#%%
if __name__ == "__main__":
    _4lv9_df = get_predicted_molecules('4lv9_2')
    print(f"\nFound {len(_4lv9_df)} molecules with target 4lv9_2 (including native ligand)")
    fingerprints = analyze_pharmacophore_fingerprints(_4lv9_df)
    similarities = analyze_similarity(fingerprints)

    # Cluster using KMeans
    clusters = cluster(similarities, n_clusters=10)
    print_cluster_info(clusters, _4lv9_df)

    # Get native ligand and best molecule IDs for cluster analysis
    native_id = _4lv9_df[_4lv9_df['is_native']]['id'].iloc[0]
    best_id = _4lv9_df[_4lv9_df['is_best_score']]['id'].iloc[0]

    # Print which cluster the native ligand and best molecule belong to
    print(f"\nCluster Assignment:")
    print(f"Native ligand (ID: {native_id}) belongs to cluster {clusters[native_id]}")
    print(f"Best molecule (ID: {best_id}) belongs to cluster {clusters[best_id]}")

    # Calculate similarity between native ligand and best molecule
    native_fp = fingerprints[native_id]['fingerprint']
    best_fp = fingerprints[best_id]['fingerprint']
    similarity = DataStructs.TanimotoSimilarity(native_fp, best_fp)
    print(f"\nSimilarity between native ligand and best molecule: {similarity:.3f}")

    plot_clusters(clusters, similarities, _4lv9_df)

# %%
