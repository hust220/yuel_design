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

def select_molecules():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sdf FROM gpr75_molecules 
            WHERE validity = true
        """)
        mol_ids = []
        sdfs = []
        for mol_id, sdf_bytes in cursor.fetchall():
            mol_ids.append(mol_id)
            sdfs.append(sdf_bytes.tobytes().decode('utf-8'))
        return mol_ids, sdfs

def filter_molecules(mol_ids, sdfs):
    filtered_mol_ids = []
    filtered_sdfs = []
    
    for mol_id, sdf in tqdm(zip(mol_ids, sdfs), desc="Filtering molecules", total=len(mol_ids)):
        # Check for standalone 'S' atom in the atom block of SDF
        has_sulfur = any(' S ' in line for line in sdf.split('\n'))
        if not has_sulfur:
            filtered_mol_ids.append(mol_id)
            filtered_sdfs.append(sdf)
    
    print(f"Filtered out {len(mol_ids) - len(filtered_mol_ids)} molecules containing sulfur")
    print(f"Remaining molecules: {len(filtered_mol_ids)}")
    
    return filtered_mol_ids, filtered_sdfs

def calculate_fingerprints(mol_ids, sdfs):
    fingerprints = []
    for sdf in tqdm(sdfs, desc="Calculating fingerprints"):
        mol = Chem.MolFromMolBlock(sdf, sanitize=False)
        if mol:
            # Using Morgan fingerprints (ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            # Convert to numpy array for KMeans clustering
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(None)
    return np.array([fp for fp in fingerprints if fp is not None])

def calculate_similarities(fingerprints):
    # Calculate pairwise Tanimoto similarities using numpy operations
    dot_product = np.dot(fingerprints, fingerprints.T)
    norm = np.sum(fingerprints, axis=1)
    similarity_matrix = dot_product / (norm[:, None] + norm[None, :] - dot_product)
    return similarity_matrix

def cluster_fingerprints(fingerprints, n_clusters=10):
    # Perform KMeans clustering directly on fingerprints
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(fingerprints)
    
    # Print cluster sizes
    unique_clusters, counts = np.unique(labels, return_counts=True)
    print("\nCluster Statistics:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster}: {count} molecules ({count/len(labels)*100:.1f}%)")
    
    # Return both labels and cluster centers for finding the most representative molecules
    return labels, kmeans.cluster_centers_

def extract_pharmacophore_features_simple(mol):
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    features = feature_factory.GetFeaturesForMol(mol)
    return [(RDKIT_TO_FEATURE_MAP[f.GetFamily()], f.GetPos()) 
            for f in features if f.GetFamily() in RDKIT_TO_FEATURE_MAP]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def analyze_pharmacophore(mol):
    """Analyze pharmacophore features of a molecule."""
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    features = feature_factory.GetFeaturesForMol(mol)
    
    # Initialize dictionary to store atoms for each feature type
    feature_atoms = {
        'Donor': [],
        'Acceptor': [],
        'Aromatic': [],
        'Hydrophobe': [],
        'PosIonizable': [],
        'NegIonizable': []
    }
    
    # Collect atoms for each feature
    for feature in features:
        feature_type = feature.GetFamily()
        if feature_type in feature_atoms:
            feature_atoms[feature_type].extend(feature.GetAtomIds())
            
    return feature_atoms

def draw_svg(mol, filename, qed_score=None):
    # Draw molecule
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(350, 250)  # Size matches our 3.5,2.5 ratio
    
    # Customize drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False  # Hide R/S stereochemistry annotations
    opts.addAtomIndices = False
    opts.baseFontSize = 1
    opts.bondLineWidth = 3
    opts.noAtomLabels = True  # Hide all atom labels
    
    # Set custom atom colors
    opts.updateAtomPalette({
        7: (0, 0, 1),  # N: blue
        8: (1, 0, 0),  # O: red
        6: (0.4, 0.4, 0.4)  # C: grey
    })
    
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # Add QED score
    svg_additions = []
    if qed_score is not None:
        svg_additions.append(f'<text x="10" y="20" font-family="sans-serif" font-size="12px">QED: {qed_score:.3f}</text>')
    
    # Insert all additions before closing tag
    if svg_additions:
        svg = svg.replace('</svg>', '\n'.join(svg_additions) + '</svg>')
    
    # Save SVG
    print(f"Saving structure to {filename}")
    with open(filename, 'w') as f:
        f.write(svg)
    
    # Display SVG
    display(SVG(svg))

def process_single_molecule(mol, sanitize=True):
    """Process a single molecule and return its 2D coordinates and QED score"""
    if not mol:
        return None, None
    
    mol_2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_2d)
    
    if sanitize:
        try:
            Chem.SanitizeMol(mol_2d)
            qed_score = QED.default(mol_2d)
            return mol_2d, qed_score
        except:
            return mol_2d, None
    return mol_2d, None

def find_representative_molecule(cluster_fingerprints, cluster_center, cluster_indices, sdfs):
    """Find the molecule closest to cluster center"""
    distances = np.linalg.norm(cluster_fingerprints - cluster_center, axis=1)
    closest_idx = cluster_indices[np.argmin(distances)]
    mol = Chem.MolFromMolBlock(sdfs[closest_idx], sanitize=False)
    return process_single_molecule(mol)

def perform_secondary_clustering(cluster_mols, cluster_qeds, n_subclusters):
    """Perform secondary clustering on high-QED molecules"""
    sub_fingerprints = []
    for mol in cluster_mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        sub_fingerprints.append(arr)
    sub_fingerprints = np.array(sub_fingerprints)
    
    sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init='auto')
    sub_labels = sub_kmeans.fit_predict(sub_fingerprints)
    
    return sub_kmeans, sub_labels, sub_fingerprints

def process_subcluster(sub_idx, sub_kmeans, sub_fingerprints, sub_labels, cluster_mols, cluster_qeds, subcluster_dir):
    """Process and visualize a single subcluster"""
    sub_indices = np.where(sub_labels == sub_idx)[0]
    if len(sub_indices) == 0:
        return
    
    # Find molecule closest to subcluster center
    sub_center = sub_kmeans.cluster_centers_[sub_idx]
    sub_distances = np.linalg.norm(sub_fingerprints[sub_indices] - sub_center, axis=1)
    sub_rep_idx = sub_indices[np.argmin(sub_distances)]
    
    # Get representative molecule and its QED
    rep_mol = cluster_mols[sub_rep_idx]
    rep_qed = cluster_qeds[sub_rep_idx]
    
    print(f"\n    Subcluster {sub_idx}:")
    print(f"      Size: {len(sub_indices)} molecules")
    print(f"      Representative QED: {rep_qed:.3f}")
    
    # Draw representative molecule with pharmacophore features
    output_filename = f'subcluster_{sub_idx}_QED_{rep_qed:.3f}_pharmacophore.svg'
    draw_molecule_with_pharmacophore(
        rep_mol, 
        os.path.join(subcluster_dir, output_filename)
    )

def plot_centroid(cluster_ids, cluster_centers, sdfs, fingerprints):
    """Analyze and visualize cluster centroids with improved organization and error handling"""
    # Create output directory
    output_dir = "GPR75_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    n_clusters = len(cluster_centers)
    total_molecules = len(cluster_ids)
    
    # Lists to store high QED molecules and their fingerprints
    mols_high_qed = []
    fingerprints_high_qed = []
    
    print("\nCluster Centroids Analysis:")
    print("-" * 30)
    
    for cluster_idx in range(n_clusters):
        # Get molecules in this cluster
        cluster_indices = np.where(cluster_ids == cluster_idx)[0]
        if len(cluster_indices) == 0:
            continue
        
        # Find representative molecule
        cluster_center = cluster_centers[cluster_idx]
        cluster_fingerprints = fingerprints[cluster_indices]
        mol_2d, qed_score = find_representative_molecule(
            cluster_fingerprints, cluster_center, cluster_indices, sdfs
        )
        
        if mol_2d is None:
            print(f"\nWarning: Could not process cluster {cluster_idx}")
            continue
        
        # Print cluster statistics
        cluster_size = len(cluster_indices)
        print(f"\nCluster {cluster_idx}:")
        print(f"  Size: {cluster_size} molecules ({cluster_size/total_molecules*100:.1f}%)")
        if qed_score is not None:
            print(f"  QED Score: {qed_score:.3f}")
        
        # Draw basic structure with QED
        draw_svg(mol_2d, f'{output_dir}/centroid_molecule_cluster_{cluster_idx}.svg', qed_score)
        
        # Process high-QED clusters
        if qed_score is not None and qed_score > 0.7:
            print(f"\n  High QED score ({qed_score:.3f}) - Processing cluster:")
            
            # Collect and calculate QED for all molecules in this cluster
            cluster_mols = []
            cluster_qeds = []
            cluster_sdfs = []  # Store original SDF strings
            cluster_fps = []   # Store fingerprints
            
            print("  Calculating QED scores for all molecules in cluster...")
            for idx in cluster_indices:
                mol = Chem.MolFromMolBlock(sdfs[idx], sanitize=False)
                mol_2d, qed = process_single_molecule(mol)
                if mol_2d is not None and qed is not None:
                    cluster_mols.append(mol_2d)
                    cluster_qeds.append(qed)
                    cluster_sdfs.append(sdfs[idx])
                    cluster_fps.append(fingerprints[idx])
            
            # Filter molecules with QED > 0.7
            high_qed_indices = [i for i, qed in enumerate(cluster_qeds) if qed > 0.7]
            high_qed_mols = [cluster_mols[i] for i in high_qed_indices]
            high_qed_scores = [cluster_qeds[i] for i in high_qed_indices]
            high_qed_sdfs = [cluster_sdfs[i] for i in high_qed_indices]
            high_qed_fps = [cluster_fps[i] for i in high_qed_indices]
            
            # Add high QED molecules and their fingerprints to our collection
            mols_high_qed.extend(high_qed_mols)
            fingerprints_high_qed.extend(high_qed_fps)
            
            print(f"  Found {len(high_qed_mols)} molecules with QED > 0.7")
            
            # Create directory for all high QED molecules
            sdf_dir = os.path.join(output_dir, f'cluster_{cluster_idx}_high_qed_molecules')
            os.makedirs(sdf_dir, exist_ok=True)
            
            # Save all high QED molecules as individual SDF files
            print("  Saving high QED molecules as SDF files...")
            for i, (mol, qed, sdf) in enumerate(zip(high_qed_mols, high_qed_scores, high_qed_sdfs)):
                sdf_filename = os.path.join(sdf_dir, f'molecule_{i}_QED_{qed:.3f}.sdf')
                with open(sdf_filename, 'w') as f:
                    f.write(sdf)
            
            if len(high_qed_mols) > 1:
                # Determine number of subclusters
                n_subclusters = min(5, max(2, len(high_qed_mols) // 10))
                
                # Perform secondary clustering
                sub_kmeans, sub_labels, sub_fingerprints = perform_secondary_clustering(
                    high_qed_mols, high_qed_scores, n_subclusters
                )
                
                print(f"\n  Secondary clustering into {n_subclusters} subclusters:")
                
                # Create directory for subclusters
                subcluster_dir = os.path.join(output_dir, f'cluster_{cluster_idx}_high_qed_subclusters')
                os.makedirs(subcluster_dir, exist_ok=True)
                
                # Process each subcluster
                for sub_idx in range(n_subclusters):
                    process_subcluster(
                        sub_idx, sub_kmeans, sub_fingerprints, sub_labels,
                        high_qed_mols, high_qed_scores, subcluster_dir
                    )
            else:
                print("  Not enough high-QED molecules for clustering")
    
    print(f"\nAll visualizations saved to {output_dir}")
    print(f"Total clusters processed: {n_clusters}")
    print(f"Total high QED molecules collected: {len(mols_high_qed)}")
    
    # Convert fingerprints list to numpy array
    fingerprints_high_qed = np.array(fingerprints_high_qed)
    
    return mols_high_qed, fingerprints_high_qed

def draw_molecule_with_pharmacophore(mol, filename):
    """Draw molecule with highlighted pharmacophore features."""
    # Get pharmacophore features
    feature_atoms = analyze_pharmacophore(mol)
    
    # Define highlight colors for different features using specified hex colors
    colors = {
        'Donor': hex_to_rgb('#A9CA70'),      # Specified green
        'Acceptor': hex_to_rgb('#F18C54'),   # Specified orange/red
        'Aromatic': hex_to_rgb('#C5D6F0'),   # Specified blue
        'Hydrophobe': (0.7, 0.7, 0.7),       # Gray
        'PosIonizable': (1, 0.7, 0.7),       # Light red
        'NegIonizable': (0.7, 0.7, 1)        # Light blue
    }
    
    # Create atom highlights dictionary
    highlight_atoms = set()  # Use set to avoid duplicates
    highlight_colors = {}
    
    # Get total number of atoms in molecule
    num_atoms = mol.GetNumAtoms()
    
    # Collect all atoms to highlight and their colors
    for feature_type, atoms in feature_atoms.items():
        for atom_idx in atoms:
            # Check if atom index is valid
            if 0 <= atom_idx < num_atoms:
                highlight_atoms.add(atom_idx)
                highlight_colors[atom_idx] = colors[feature_type]
    
    # Convert set to list for rdkit
    highlight_atoms = list(highlight_atoms)
    
    # Generate 2D coordinates for better visualization
    rdDepictor.Compute2DCoords(mol)
    
    try:
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
        drawer.SetFontSize(0.8)
        
        # Draw molecule with highlights
        if highlight_atoms:
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, 
                                              highlightAtoms=highlight_atoms,
                                              highlightAtomColors=highlight_colors)
        else:
            # If no highlights, just draw the molecule
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        
        # Save SVG
        drawer.FinishDrawing()
        with open(filename, 'w') as f:
            f.write(drawer.GetDrawingText())

        # Print feature summary
        print(f"\nPharmacophore features for {filename}:")
        for feature_type, atoms in feature_atoms.items():
            valid_atoms = [a for a in atoms if 0 <= a < num_atoms]
            if valid_atoms:
                print(f"{feature_type}: {len(valid_atoms)} atoms")
    
    except Exception as e:
        print(f"Error drawing molecule {filename}: {str(e)}")
        # Try to draw molecule without highlights as fallback
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
            drawer.SetFontSize(0.8)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            with open(filename, 'w') as f:
                f.write(drawer.GetDrawingText())
            print(f"Generated basic structure for {filename} without highlights")
        except Exception as e2:
            print(f"Failed to generate even basic structure for {filename}: {str(e2)}")

def calculate_hmdb_fingerprints(hmdb_file, output_file):
    """
    Calculate Morgan fingerprints for HMDB molecules and save them to a pickle file.
    
    Args:
        hmdb_file (str): Path to the HMDB SDF file
        output_file (str): Path to save the fingerprints pickle file
    """
    print(f"Reading molecules from {hmdb_file}...")
    supplier = Chem.SDMolSupplier(hmdb_file)
    
    fingerprints = []
    smiles = []
    
    for i, mol in enumerate(tqdm(supplier, desc="Calculating fingerprints")):
        if mol is not None:
            try:
                # Generate SMILES
                smi = Chem.MolToSmiles(mol)
                # Calculate Morgan fingerprint (ECFP4)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # Convert to numpy array
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                
                fingerprints.append(arr)
                smiles.append(smi)
            except:
                print(f"Failed to process molecule {i}")
                continue
    
    fingerprints = np.array(fingerprints)
    
    print(f"Saving {len(fingerprints)} fingerprints to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump({'smiles': smiles, 'fingerprints': fingerprints}, f)

def load_fingerprints(pickle_file):
    """Load fingerprints from a pickle file."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data['smiles'], data['fingerprints']

def calculate_hmdb_similarities(hmdb_fingerprints, fingerprints_high_qed):
    def array_to_bitvect(arr):
        bv = DataStructs.ExplicitBitVect(len(arr))
        for i in range(len(arr)):
            if arr[i]:
                bv.SetBit(i)
        return bv
    
    print("Converting fingerprints to RDKit format...")
    rdkit_fps_high_qed = [array_to_bitvect(fp) for fp in tqdm(fingerprints_high_qed, desc="Converting high QED fingerprints")]
    rdkit_fps_hmdb = [array_to_bitvect(fp) for fp in tqdm(hmdb_fingerprints, desc="Converting HMDB fingerprints")]
    
    similarities = []
    for fp1 in tqdm(rdkit_fps_hmdb, desc="Calculating similarities"):
        max_sim = max(DataStructs.TanimotoSimilarity(fp1, fp2) for fp2 in rdkit_fps_high_qed)
        similarities.append(max_sim)
    
    return np.array(similarities)

def find_metabolites_with_high_similarity(hmdb_similarities, top_n=10):
    sorted_indices = np.argsort(hmdb_similarities)[::-1]
    top_indices = sorted_indices[:top_n]
    mol_indices = np.arange(top_n)
    
    print("\nTop similar metabolites:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Metabolite {idx}: similarity = {hmdb_similarities[idx]:.3f}")
    
    return top_indices, mol_indices

def plot_metabolites(metabolite_indices, mol_indices, hmdb_smiles, high_qed_mols, similarities=None):    
    output_dir = "similarity_pairs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (met_idx, mol_idx) in enumerate(zip(metabolite_indices, mol_indices)):
        sim_str = f"{similarities[met_idx]:.3f}" if similarities is not None else "unknown"
        pair_dir = os.path.join(output_dir, f"pair_{i+1}_sim_{sim_str}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Process metabolite
        smiles = hmdb_smiles[met_idx]
        mol_metabolite = Chem.MolFromSmiles(smiles)
        mol_pred = high_qed_mols[mol_idx]
        
        if mol_metabolite is None:
            print(f"Could not create molecule from SMILES for metabolite {met_idx}")
            continue
            
        try:
            # Save metabolite files
            mol_2d = Chem.Mol(mol_metabolite)
            AllChem.Compute2DCoords(mol_2d)
            draw_svg(mol_2d, os.path.join(pair_dir, "metabolite.svg"))
            
            # Save prediction files
            mol_2d_pred = Chem.Mol(mol_pred)
            AllChem.Compute2DCoords(mol_2d_pred)
            draw_svg(mol_2d_pred, os.path.join(pair_dir, "prediction.svg"))
            
            # Save SDF files
            writer = Chem.SDWriter(os.path.join(pair_dir, "metabolite.sdf"))
            writer.write(mol_2d)
            writer.close()
            
            writer = Chem.SDWriter(os.path.join(pair_dir, "prediction.sdf"))
            writer.write(mol_2d_pred)
            writer.close()
            
            # Print information
            print(f"\nPair {i+1} (similarity: {sim_str}):")
            print(f"  HMDB Index: {met_idx}")
            print(f"  SMILES: {smiles}")
            print(f"  Files saved in: {pair_dir}")
            
        except Exception as e:
            print(f"Error processing pair {i+1}: {str(e)}")
            continue

if __name__ == '__main__':
    mol_ids, sdfs = select_molecules()
    mol_ids, sdfs = filter_molecules(mol_ids, sdfs)
    # print(len(mol_ids))
    fingerprints = calculate_fingerprints(mol_ids, sdfs)
    # similarities = calculate_similarities(fingerprints)
    cluster_ids, cluster_centers = cluster_fingerprints(fingerprints)
    mols_high_qed, fingerprints_high_qed = plot_centroid(cluster_ids, cluster_centers, sdfs, fingerprints)

    # calculate_hmdb_fingerprints('hmdb.sdf', 'hmdb_fingerprints.pkl')
    # hmdb_smiles, hmdb_fingerprints = load_fingerprints('hmdb_fingerprints.pkl')
    # hmdb_similarities = calculate_hmdb_similarities(hmdb_fingerprints, fingerprints_high_qed)
    # metabolite_indices, mol_indices = find_metabolites_with_high_similarity(hmdb_similarities)

    # print(len(metabolite_indices))
    # print(len(hmdb_smiles))
    # plot_metabolites(metabolite_indices, mol_indices, hmdb_smiles, mols_high_qed, hmdb_similarities)



# %%
