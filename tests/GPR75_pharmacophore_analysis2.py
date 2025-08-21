
#%%
from tqdm import tqdm
import sys
sys.path.append('../')
from db_utils import db_connection
from rdkit import Chem
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from rdkit.Chem import Draw, AllChem, RDConfig
import json
import multiprocessing as mp
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from rdkit.Chem import DataStructs
from IPython.display import SVG, display
from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit.Chem.Draw import rdDepictor
from sklearn.cluster import KMeans


FEATURE_WEIGHTS = {
    'HBD': 1.0,  # Hydrogen bond donor
    'HBA': 1.0,  # Hydrogen bond acceptor
    'AR': 0.8,   # Aromatic ring
    'H': 0.6,    # Hydrophobic
    'PI': 0.7,   # Positive ionizable
    'NI': 0.7    # Negative ionizable
}

FEATURE_COLORS = {
    'HBD': (0, 0.9, 0),  # Green
    'HBA': (0.9, 0, 0),  # Red
    'AR': (0.9, 0.9, 0),  # Yellow
    'H': (0.5, 0.5, 0.5),  # Gray
    'PI': (0, 0, 0.9),  # Blue
    'NI': (0.9, 0, 0.9)  # Purple
}

RDKIT_TO_FEATURE_MAP = {
    'Donor': 'HBD',
    'Acceptor': 'HBA',
    'Aromatic': 'AR',
    'Hydrophobe': 'H',
    'PosIonizable': 'PI',
    'NegIonizable': 'NI'
}

def add_pharmacophore_columns():
    with db_connection() as conn:
        conn.cursor().execute("""
            ALTER TABLE gpr75_molecules 
            ADD COLUMN IF NOT EXISTS pharmacophore_features TEXT,
            ADD COLUMN IF NOT EXISTS pharmacophore_feature_count INTEGER,
            ADD COLUMN IF NOT EXISTS cluster_id INTEGER
        """)
        conn.commit()

def extract_pharmacophore_features(mol):
    try:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
        return [(RDKIT_TO_FEATURE_MAP[f.GetFamily()], f.GetPos()) 
                for f in feature_factory.GetFeaturesForMol(mol)
                if f.GetFamily() in RDKIT_TO_FEATURE_MAP]
    except Exception as e:
        print(f"Error extracting pharmacophore features: {str(e)}")
        return None

def calculate_molecule_pharmacophores():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM gpr75_molecules WHERE validity = true")
        total_molecules = cursor.fetchone()[0]
        
        batch_size = 100
        for offset in tqdm(range(0, total_molecules, batch_size), desc="Processing batches"):
            cursor.execute("""
                SELECT id, sdf FROM gpr75_molecules 
                WHERE validity = true ORDER BY id LIMIT %s OFFSET %s
            """, (batch_size, offset))
            
            for mol_id, sdf_bytes in tqdm(cursor.fetchall(), desc=f"Batch {offset//batch_size + 1}", leave=False):
                mol = Chem.MolFromMolBlock(sdf_bytes.tobytes().decode('utf-8'), sanitize=False)
                if not mol:
                    continue
                
                features = extract_pharmacophore_features(mol)
                if features:
                    features_json = [{'type': t, 'x': float(p.x), 'y': float(p.y), 'z': float(p.z)} 
                                   for t, p in features]
                    cursor.execute("""
                        UPDATE gpr75_molecules 
                        SET pharmacophore_features = %s, pharmacophore_feature_count = %s
                        WHERE id = %s
                    """, (json.dumps(features_json), len(features), mol_id))
                else:
                    cursor.execute("UPDATE gpr75_molecules SET pharmacophore_feature_count = 0 WHERE id = %s", (mol_id,))
            conn.commit()

def calculate_pharmacophore_similarity(features1, features2, distance_threshold=2.0):
    if not features1 or not features2:
        return 0.0
    
    def group_features(features):
        grouped = {}
        for f in features:
            ftype = f['type']
            if ftype not in grouped:
                grouped[ftype] = []
            grouped[ftype].append(np.array([f['x'], f['y'], f['z']]))
        return grouped
    
    def type_similarity(coords1, coords2):
        if not coords1 or not coords2:
            return 0.0
        distances = cdist(coords1, coords2)
        min_distances = np.min(distances, axis=1)
        similarities = np.exp(-min_distances**2 / (2 * distance_threshold**2))
        weight = min(len(coords1), len(coords2)) / max(len(coords1), len(coords2))
        return weight * np.mean(similarities)
    
    grouped1, grouped2 = group_features(features1), group_features(features2)
    all_types = set(grouped1) | set(grouped2)
    
    similarities = []
    for ftype in all_types:
        sim = type_similarity(grouped1.get(ftype, []), grouped2.get(ftype, []))
        weight = FEATURE_WEIGHTS.get(ftype, 0.5)
        similarities.append(sim * weight)
    
    total_weight = sum(FEATURE_WEIGHTS.get(ftype, 0.5) for ftype in all_types)
    return sum(similarities) / total_weight if total_weight > 0 else 0.0

def calculate_similarity_chunk(chunk_data):
    chunk, features_dict, id_to_idx = chunk_data
    return [(id_to_idx[i], id_to_idx[j], 
             calculate_pharmacophore_similarity(features_dict[i], features_dict[j]))
            for i, j in chunk]

def calculate_pharmacophore_similarities(n_cpus=None):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, pharmacophore_features FROM gpr75_molecules 
            WHERE validity = true AND pharmacophore_features IS NOT NULL
            AND pharmacophore_feature_count >= 3
        """)
        
        features_dict = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        if not features_dict:
            print("No valid molecules with sufficient pharmacophore features found")
            return
        
        molecule_ids = list(features_dict.keys())
        id_to_idx = {mol_id: idx for idx, mol_id in enumerate(molecule_ids)}
        pairs = list(combinations(molecule_ids, 2))
        
        n_cpus = n_cpus or mp.cpu_count()
        chunk_size = max(1, len(pairs) // (n_cpus * 4))
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        
        similarity_matrix = np.zeros((len(molecule_ids), len(molecule_ids)))
        with mp.Pool(n_cpus) as pool:
            for chunk_results in tqdm(pool.imap(calculate_similarity_chunk, 
                                              [(c, features_dict, id_to_idx) for c in chunks]),
                                    total=len(chunks), desc="Calculating similarities"):
                for i, j, sim in chunk_results:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim
        
        with open('pharmacophore_similarity_matrix.pkl', 'wb') as f:
            pickle.dump({'similarity_matrix': similarity_matrix,
                        'molecule_ids': molecule_ids,
                        'id_to_idx': id_to_idx}, f)

def load_similarities():
    with open('pharmacophore_similarity_matrix.pkl', 'rb') as f:
        data = pickle.load(f)
        return data['similarity_matrix'], data['molecule_ids'], data['id_to_idx']

def plot_similarities():
    similarity_matrix, _, _ = load_similarities()
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False,
                yticklabels=False, vmin=0, vmax=1)
    plt.title('Pharmacophore Similarity Matrix')
    plt.savefig('pharmacophore_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def check_sulfar_molecules():
    sulfur_mols = {}
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, sdf FROM gpr75_molecules WHERE validity = true")
        for mol_id, sdf_bytes in cursor.fetchall():
            sdf_content = sdf_bytes.tobytes().decode('utf-8')
            # Check for standalone 'S' atom in the atom block of SDF
            # Atom lines in SDF format have the atom symbol starting at position 31
            has_sulfur = any(line.strip().startswith('S ') or line.strip().startswith('S\t') 
                           for line in sdf_content.split('\n'))
            sulfur_mols[mol_id] = has_sulfur
    return sulfur_mols

def cluster_pharmacophore_similarities(n_clusters=10, filter=None):
    with db_connection() as conn:
        cursor = conn.cursor()
        similarity_matrix, molecule_ids, _ = load_similarities()
        
        if filter is None:
            # If no filter provided, include all molecules
            filter = {mol_id: True for mol_id in molecule_ids}
        
        # Get indices of molecules to include in clustering
        include_indices = [i for i, mol_id in enumerate(molecule_ids) if filter.get(mol_id, False)]
        
        if not include_indices:
            print("No molecules found for clustering after filtering")
            return
        
        # Create submatrix for included molecules
        include_matrix = similarity_matrix[np.ix_(include_indices, include_indices)]
        
        # Perform clustering on included molecules
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        ).fit(1 - include_matrix)
        
        # Create full labels array with -1 for excluded molecules
        labels = np.full(len(molecule_ids), -1)
        for i, idx in enumerate(include_indices):
            labels[idx] = clustering.labels_[i]
        
        # Plot dendrogram for included molecules
        plt.figure(figsize=(15, 10))
        dendrogram(linkage(1 - include_matrix, method='average'), truncate_mode='level', p=5)
        plt.title('Hierarchical Clustering Dendrogram (Filtered Molecules)')
        plt.xlabel('Molecule Index')
        plt.ylabel('Distance')
        plt.savefig('pharmacophore_dendrogram_filtered.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Update cluster IDs in database
        for i, mol_id in enumerate(molecule_ids):
            cursor.execute("UPDATE gpr75_molecules SET cluster_id = %s WHERE id = %s",
                         (int(labels[i]), mol_id))
        conn.commit()
        
        # Print statistics
        unique_labels = set(labels)
        print("\nCluster Statistics:")
        for label in sorted(unique_labels):
            count = sum(labels == label)
            if label == -1:
                print(f"Excluded molecules: {count}")
            else:
                print(f"Cluster {label}: {count} molecules")
        
        return labels

def plot_centroid_pharmacophore():
    with db_connection() as conn:
        cursor = conn.cursor()
        # Get the largest cluster ID
        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size 
            FROM gpr75_molecules 
            WHERE cluster_id IS NOT NULL AND cluster_id >= 0 
            GROUP BY cluster_id 
            ORDER BY size DESC 
            LIMIT 1
        """)
        cluster_id, size = cursor.fetchone() or (None, 0)
        if not size:
            print("No valid clusters found")
            return
        
        print(f"Analyzing cluster {cluster_id} with {size} molecules")
        
        # Get all molecules in this cluster
        cursor.execute("""
            SELECT id, sdf 
            FROM gpr75_molecules 
            WHERE cluster_id = %s
        """, (cluster_id,))
        
        # Just take the first molecule as representative
        mol_id, sdf_bytes = cursor.fetchone()
        mol = Chem.MolFromMolBlock(sdf_bytes.tobytes().decode('utf-8'))
        if not mol:
            print("Could not create molecule from SDF")
            return
        
        # Create a new molecule without 3D coordinates
        mol_2d = Chem.Mol(mol)
        for atom in mol_2d.GetAtoms():
            atom.SetNoImplicit(False)
        AllChem.Compute2DCoords(mol_2d)
        
        # Draw molecule as SVG
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(800, 800)
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        
        # Save SVG image
        svg_content = drawer.GetDrawingText()
        with open(f'centroid_molecule_cluster_{cluster_id}.svg', 'w') as f:
            f.write(svg_content)
        print(f"Saved centroid molecule visualization to centroid_molecule_cluster_{cluster_id}.svg")
        
        # Display molecule
        img = Draw.MolToImage(mol_2d)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        return mol_2d

def plot_clustering_results():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size
            FROM gpr75_molecules WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id ORDER BY cluster_id
        """)
        cluster_stats = cursor.fetchall()
        if not cluster_stats:
            print("No clustering results found in database")
            return
        
        similarity_matrix, molecule_ids, id_to_idx = load_similarities()
        cursor.execute("""
            SELECT id, cluster_id FROM gpr75_molecules
            WHERE cluster_id IS NOT NULL ORDER BY id
        """)
        
        labels = [None] * len(molecule_ids)
        for mol_id, cluster_id in cursor.fetchall():
            if mol_id in id_to_idx:
                labels[id_to_idx[mol_id]] = cluster_id
        
        order = np.argsort(labels)
        reordered_matrix = similarity_matrix[order][:, order]
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(reordered_matrix, cmap='viridis', xticklabels=False,
                   yticklabels=False, cbar_kws={'label': 'Similarity'})
        
        current_pos = 0
        total_molecules = 0
        for cluster_id, size in cluster_stats:
            if size > 0:
                center_pos = current_pos + size/2
                plt.text(-0.05, center_pos, f'C{cluster_id}\n({size})',
                        horizontalalignment='right', verticalalignment='center')
                
                if current_pos > 0:
                    plt.axhline(y=current_pos, color='white', linewidth=0.5)
                    plt.axvline(x=current_pos, color='white', linewidth=0.5)
                
                current_pos += size
                total_molecules += size
        
        plt.title('Pharmacophore Similarity Matrix\n(Sorted by Clusters)')
        plt.tight_layout()
        plt.savefig('GPR75_clustering_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nCluster Statistics:")
        for cluster_id, size in cluster_stats:
            if size > 0:
                print(f"Cluster {cluster_id}: {size} molecules ({size/total_molecules*100:.1f}%)")



if __name__ == '__main__':
    # add_pharmacophore_columns()
    # calculate_molecule_pharmacophores()
    # calculate_pharmacophore_similarities(n_cpus=12)
    # plot_similarities()
    # is_sulfar = check_sulfar_molecules()
    # cluster_pharmacophore_similarities(n_clusters=10, filter={mol_id: not is_sulfar for mol_id, is_sulfar in is_sulfar.items()})
    # plot_centroid_pharmacophore()
    # plot_clustering_results()
