import os
import numpy as np
import pickle
import torch

from torch.utils.data import Dataset
from rdkit import Chem
from src import const
from src.pdb_utils import Structure
import src.gnn as gnn
from src.cache import FileCache
from src.db_utils import db_connection
from src.distance_discretization import discretize_distance_numpy
from src.const import PROTEIN_ATOM_TYPES
from src.disc2.dataset import detect_ligand_rings

# Define unified atom types for coarse-grained representation
# Protein atoms: CA + all non-C PDB atoms (from src.const.PROTEIN_ATOM_TYPES)

# Protein non-C atoms: extract all non-C atoms from PDB_ATOM_TYPES
# This includes: N, O, S, P, ND1, OD1, SG, etc.
protein_non_c_atoms = []
for element, atom_names in PROTEIN_ATOM_TYPES.items():
    if element != 'C':
        protein_non_c_atoms.extend(atom_names)

# Ring center virtual atoms (by ring size) - shared by both protein and ligand
RING_CENTERS = ['RING_3', 'RING_4', 'RING_5', 'RING_6', 'RING_X']  # 3, 4, 5, 6, and larger rings

# Protein atom types: CA + all non-C atoms + ring center virtual atoms
protein_atoms = ['CA'] + protein_non_c_atoms + RING_CENTERS

# Ligand elements: common elements (prefixed with _ to avoid conflict with protein atoms)
ligand_elements = ['_C', '_O', '_N', '_F', '_S', '_P', '_Cl', '_Br', '_I',
                   '_ZN', '_MG', '_FE', '_CU', '_MN', '_CO', '_NI', '_MO', '_W', '_SE']

# Combined atom types: X (unknown) + protein atoms + ligand elements
# Note: RING_CENTERS are already in protein_atoms, so we only add ligand_elements (not LIGAND_ATOM_TYPES)
# to avoid duplicating ring centers
ALLOWED_ATOM_TYPES = ['X'] + protein_atoms + ligand_elements

# Create mappings
ATOM2IDX = {atom: idx for idx, atom in enumerate(ALLOWED_ATOM_TYPES)}

def get_one_hot(atom, atoms_dict):
    """Get one-hot encoding for atom."""
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def get_atom_one_hot(atom_name):
    """Get one-hot encoding for atom name.
    Supports protein atoms (CA, ND1, OD1, SG, etc.), ligand elements, and ring centers.
    For unknown atoms, uses 'X' as fallback.
    """
    if atom_name in ATOM2IDX:
        return get_one_hot(atom_name, ATOM2IDX)
    else:
        # Fallback to 'X' for unknown atoms
        return get_one_hot('X', ATOM2IDX)

def detect_protein_ring(res_name, residue_atoms, coords):
    """Detect rings in protein residue side chains.
    Returns list of (ring_size, ring_center_coord), where ring_size is used to determine ring type.
    """
    ring_centers = []
    
    # Known aromatic/cyclic amino acids with rings (mapping to ring sizes)
    aromatic_residues = {
        'PHE': (['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 6),  # 6-membered ring
        'TYR': (['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 6),  # 6-membered ring
        'TRP': (['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], 9),  # 9-membered fused ring system
        'HIS': (['CG', 'ND1', 'CD2', 'CE1', 'NE2'], 5),  # 5-membered ring
        'PRO': (['CG', 'CD', 'N'], 5),  # 5-membered ring (includes backbone N)
    }
    
    if res_name in aromatic_residues:
        ring_atom_names, ring_size = aromatic_residues[res_name]
        ring_coords = []
        
        for name, coord in zip(residue_atoms, coords):
            if name in ring_atom_names:
                ring_coords.append(coord)
        
        # Only add ring center if we found at least 3 ring atoms
        if len(ring_coords) >= 3:
            ring_center = np.mean(ring_coords, axis=0)
            # Determine ring type based on size
            if ring_size == 3:
                ring_type = 'RING_3'
            elif ring_size == 4:
                ring_type = 'RING_4'
            elif ring_size == 5:
                ring_type = 'RING_5'
            elif ring_size == 6:
                ring_type = 'RING_6'
            else:
                ring_type = 'RING_X'  # Larger rings
            ring_centers.append((ring_type, ring_center))
    
    return ring_centers

def parse_pocket(structure):
    """Parse protein structure: keep all atoms (excluding H), plus ring centers as virtual atoms.
    Only supports protein residues, does not handle RNA/DNA.
    
    Returns:
        dict with keys:
            - reduced_coords: Coordinates of reduced atoms (CA + all non-C atoms + ring centers)
            - all_coords: Coordinates of all atoms (CA, non-C, C atoms, and ring centers)
            - residues: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
              where reduced_atoms are CA + all non-C atoms + ring centers,
              and full_atoms are all atoms (CA, non-C, C atoms, and ring centers)
    """
    reduced_coords_flat = []
    full_coords_flat = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                reduced_atoms, reduced_coords = [], []
                full_atoms, full_coords = [], []

                res_name = residue.res_name
                ca_coord = None
                
                # Collect all atoms (excluding H)
                for atom in residue:
                    if atom.atom_name[0] == 'H':
                        continue
                    atom_name = atom.atom_name
                    atom_coord = atom.get_coord()
                    full_atoms.append(atom_name)
                    full_coords.append(atom_coord)
                    if atom_name == 'CA' or atom_name.startswith('RING_') or atom_name[0] != 'C':
                        reduced_atoms.append(atom_name)
                        reduced_coords.append(atom_coord)
                        if atom_name == 'CA':
                            ca_coord = atom_coord
                
                if ca_coord is not None:
                    # Detect ring centers
                    ring_centers = detect_protein_ring(res_name, full_atoms, full_coords)
                    for ring_type, ring_center in ring_centers:
                        reduced_atoms.append(ring_type)
                        reduced_coords.append(ring_center)
                        full_atoms.append(ring_type)
                        full_coords.append(ring_center)

                    reduced_coords_flat.extend(reduced_coords)
                    full_coords_flat.extend(full_coords)
                    residues.append((reduced_atoms, full_atoms))
    
    return {
        'reduced_coords': np.array(reduced_coords_flat),
        'full_coords': np.array(full_coords_flat),
        'residues': residues,
    }

def create_dist_to_coords_features(pocket_pdb, ligand_mol):
    # Parse ligand molecule
    mol = Chem.MolFromMolBlock(ligand_mol)
    if mol is None:
        # Return None to skip this sample
        return None

    structure = Structure()
    from io import StringIO
    structure.read(StringIO(pocket_pdb))

    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        return None

    # Parse ligand molecule
    ligand_reduced_atoms, ligand_reduced_coords, ligand_full_atoms, ligand_full_coords = get_ligand_atoms_and_coords(mol)
    if len(ligand_reduced_coords) == 0 or len(ligand_full_coords) == 0:
        return None
    ligand_size = len(ligand_full_coords)

    # Get reduced coords and full coords from pocket_info
    receptor_residues = pocket_info['residues']
    receptor_reduced_coords = pocket_info['reduced_coords']
    receptor_full_coords = pocket_info['full_coords']
    
    # Create distance matrix for reduced atoms (CA + non-C + ring centers) and ligand reduced atoms (non-C + ring centers)
    dist_matrix = create_dist_matrix(receptor_reduced_coords, ligand_reduced_coords, discretization_config='b12')
    
    features = init_dist_to_coords_features(
        dist_matrix=dist_matrix,
        receptor_residues=receptor_residues,
        ligand_fixed_atoms=ligand_reduced_atoms,
        ligand_size=ligand_size
    )
    features['x'] = np.concatenate([receptor_full_coords, ligand_full_coords], axis=0)

    return features

def init_dist_to_coords_features(dist_matrix, receptor_residues, ligand_fixed_atoms, ligand_size):
    """Initialize features from distance matrix and receptor/ligand information.
    
    Args:
        dist_matrix: Reduced distance matrix with shape (n_reduced_receptor + n_ligand, n_reduced_receptor + n_ligand)
        receptor_residues: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
        ligand_fixed_atoms: List of ligand atom types for ligand atoms in dist_matrix
        ligand_size: Number of ligand atoms
    
    Returns:
        dict with features including 'h', 'edge_index', 'edge_dist', 'edge_attr'
    """
    # Calculate total number of reduced receptor atoms
    n_reduced_receptor = sum(len(reduced_atoms) for reduced_atoms, _ in receptor_residues)
    assert n_reduced_receptor + len(ligand_fixed_atoms) == dist_matrix.shape[0], \
        f"n_reduced_receptor ({n_reduced_receptor}) + ligand_fixed_atoms ({len(ligand_fixed_atoms)}) must equal dist_matrix.shape[0] ({dist_matrix.shape[0]})"
    
    # Expand distance matrix to include all receptor atoms
    full_dist_matrix, full_receptor_atoms = expand_dist_matrix(
        dist_matrix, receptor_residues, ligand_size
    )
    node_attr = build_nodes(full_receptor_atoms, ligand_fixed_atoms, ligand_size)
    edge_index, edge_dist, edge_attr = build_edges(
        dist_matrix=full_dist_matrix,
        receptor_residues=receptor_residues,
        ligand_size=ligand_size
    )
    
    features = {
        'h': node_attr,
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_attr': edge_attr,
    }
    
    return features

def expand_dist_matrix(dist_matrix, receptor_residues, ligand_size):
    """Expand reduced distance matrix (CA, ring centers, ligand only) to full node set.
    
    dist_matrix has shape (n_reduced_receptor + n_reduced_ligand, n_reduced_receptor + n_reduced_ligand)
    where n_reduced_receptor is the number of CA/ring centers, and n_reduced_ligand is the number of
    ligand atoms in dist_matrix (typically fixed ligand atoms, which may be less than ligand_size).
    Returns full matrix with shape (n_receptor + n_ligand, n_receptor + n_ligand)
    where n_receptor is the total number of receptor atoms (CA + non-C atoms + C atoms + ring centers).
    
    Args:
        dist_matrix: Reduced distance matrix with shape (n_reduced_receptor + n_reduced_ligand, ...)
            where n_reduced_ligand <= ligand_size
        receptor_residues: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
        ligand_size: Total number of ligand atoms (including atoms not in dist_matrix)
    
    Returns:
        tuple: (full_dist_matrix, full_receptor_atoms, full_receptor_residues)
            full_dist_matrix: Full distance matrix with shape (n_receptor + n_ligand, n_receptor + n_ligand)
            full_receptor_atoms: List of all receptor atom names
    """
    # Calculate total number of reduced receptor atoms
    n_reduced_receptor = sum(len(reduced_atoms) for reduced_atoms, _ in receptor_residues)
    n_reduced_total = dist_matrix.shape[0]
    n_reduced_ligand = n_reduced_total - n_reduced_receptor
    assert n_reduced_ligand <= ligand_size, \
        f"n_reduced_ligand ({n_reduced_ligand}) in dist_matrix must be <= ligand_size ({ligand_size})"
    
    # Build full receptor atom list and mapping from reduced to full indices
    full_receptor_atoms = []
    reduced_to_full_map = {}  # Map from reduced receptor index to full receptor index
    
    reduced_idx = 0
    full_idx_offset = 0
    
    # Process each residue in receptor_residues
    for reduced_atoms, full_atoms in receptor_residues:
        # Build mapping from reduced atoms to full atoms
        for atom_name in reduced_atoms:
            if atom_name in full_atoms:
                full_atom_idx = full_atoms.index(atom_name)
                reduced_to_full_map[reduced_idx] = full_idx_offset + full_atom_idx
            reduced_idx += 1
        
        # Add all full atoms to the full receptor atom list
        for atom_name in full_atoms:
            full_receptor_atoms.append(atom_name)
            full_idx_offset += 1
    
    n_receptor = len(full_receptor_atoms)
    n = n_receptor + ligand_size
    full_dist_matrix = np.zeros((n, n), dtype=np.int64)
    
    # Build complete index mapping from reduced to full indices
    full_indices = np.zeros(n_reduced_total, dtype=np.int64)
    
    # Map reduced receptor indices to full receptor indices
    for i_reduced, i_full in reduced_to_full_map.items():
        full_indices[i_reduced] = i_full
    
    # Map reduced ligand indices to full ligand indices
    ligand_start = n_reduced_receptor
    for i in range(n_reduced_ligand):
        full_indices[ligand_start + i] = n_receptor + i
    
    # Use advanced indexing to copy all distances at once
    i_reduced = np.arange(n_reduced_total)[:, None]
    j_reduced = np.arange(n_reduced_total)[None, :]
    i_full = full_indices[:, None]
    j_full = full_indices[None, :]
    full_dist_matrix[i_full, j_full] = dist_matrix[i_reduced, j_reduced] + 1
    
    return full_dist_matrix, full_receptor_atoms

def build_edges(dist_matrix, receptor_residues, ligand_size):
    """Build graph edges from distance matrix.
    Creates a fully connected graph with residue relationship features.
    
    Args:
        dist_matrix: Distance matrix with shape (n_receptor + n_ligand, n_receptor + n_ligand)
        receptor_residues: [([reduced_atoms...], [full_atoms...]), ...]
        ligand_size: Number of ligand atoms
    
    Returns:
        tuple: (edge_index, edge_dist, edge_attr)
    """
    # Build residue indices for each receptor atom
    res_indices = []
    for res_idx, (reduced_atoms, full_atoms) in enumerate(receptor_residues):
        res_indices.extend([res_idx] * len(full_atoms))
    
    n_receptor = len(res_indices)
    n = n_receptor + ligand_size
    assert dist_matrix.shape[0] == n, f"dist_matrix shape {dist_matrix.shape[0]} must equal n_receptor + ligand_size = {n}"
    
    edge_list, edge_dist, edge_attr = [], [], []
    
    # Extend residue indices for ligand (use -1 for ligand atoms)
    res_indices = res_indices + [-1] * ligand_size
    
    for i in range(n):
        for j in range(i+1, n):
            # Bidirectional edges
            edge_list.append([i, j])
            edge_list.append([j, i])
            
            # Store discrete distance from dist_matrix
            discrete_dist = int(dist_matrix[i, j])
            edge_dist.append(discrete_dist)
            edge_dist.append(discrete_dist)
            
            res_i, res_j = res_indices[i], res_indices[j]
            is_same_residue = 1 if res_i == res_j and res_i != -1 else 0
            edge_attr.append([is_same_residue])
            edge_attr.append([is_same_residue])
    
    return np.array(edge_list).T, np.array(edge_dist), np.array(edge_attr)

def build_nodes(receptor_atoms, ligand_fixed_atoms, ligand_size):
    """Build node attributes from receptor atoms and ligand fixed atoms.
    
    Args:
        receptor_atoms: List of receptor atom names (e.g., 'CA', 'RING_3', etc.)
        ligand_fixed_atoms: List of ligand atom types for fixed atoms (e.g., '_C', '_O', etc.)
        ligand_size: Total number of ligand atoms (including fixed and to-be-generated atoms)
    
    Returns:
        numpy array of node attributes with shape (n_receptor + ligand_size, node_feat_dim)
    """
    node_attrs = []
    
    # Build receptor node attributes
    for atom_name in receptor_atoms:
        # Determine mol_type based on atom name
        if atom_name == 'CA':
            mol_type = [1, 0, 0]  # backbone
        else:
            mol_type = [0, 1, 0]  # side chain or ring center
        
        atom_one_hot = get_atom_one_hot(atom_name)
        node_attrs.append(np.concatenate([mol_type, atom_one_hot]))
    
    # Build ligand node attributes
    # First add fixed ligand atoms
    for atom_symbol in ligand_fixed_atoms:
        mol_type = [0, 0, 1]  # ligand
        atom_one_hot = get_atom_one_hot(atom_symbol)
        node_attrs.append(np.concatenate([mol_type, atom_one_hot]))
    
    # Then add atoms to be generated (use 'X' for unknown atom type)
    n_generated = ligand_size - len(ligand_fixed_atoms)
    for _ in range(n_generated):
        mol_type = [0, 0, 1]  # ligand
        atom_one_hot = get_atom_one_hot('X')  # unknown atom type
        node_attrs.append(np.concatenate([mol_type, atom_one_hot]))
    
    return np.array(node_attrs)

def get_ligand_atoms_and_coords(mol):
    """Get ligand atoms and coordinates from molecule.
    
    Returns:
        tuple: (reduced_atoms, reduced_coords, full_atoms, full_coords)
            reduced_atoms: List of reduced atom types (non-C atoms + ring centers)
            reduced_coords: Coordinates of reduced atoms
            full_atoms: List of all atom types (all atoms including C atoms + ring centers)
            full_coords: Coordinates of all atoms
    """
    conf = mol.GetConformer()
    
    # Collect all atoms (excluding H)
    full_atoms = []
    full_coords = []
    reduced_atoms = []
    reduced_coords = []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx = atom.GetIdx()
        
        # Convert atom symbol to ligand element format with _ prefix
        atom_symbol = '_' + atom.GetSymbol()
        coord = conf.GetAtomPosition(atom_idx)
        coord = np.array([coord.x, coord.y, coord.z])
        
        # Add to full atoms
        full_atoms.append(atom_symbol)
        full_coords.append(coord)
        
        # Add to reduced atoms if non-C
        if atom.GetSymbol() != 'C':
            reduced_atoms.append(atom_symbol)
            reduced_coords.append(coord)
    
    # Detect and add ring centers
    ring_centers = detect_ligand_rings(mol)
    for ring_type, ring_center in ring_centers:
        reduced_atoms.append(ring_type)
        reduced_coords.append(ring_center)
        full_atoms.append(ring_type)
        full_coords.append(ring_center)
    
    return reduced_atoms, np.array(reduced_coords), full_atoms, np.array(full_coords)

def create_dist_matrix(receptor_coords, ligand_coords, discretization_config='b12'):
    """Helper function to create distance matrix from coordinates.
    
    Args:
        receptor_coords: Array of receptor atom coordinates (CA and ring centers)
        ligand_coords: Array of ligand atom coordinates
        discretization_config: Distance discretization configuration
        
    Returns:
        Distance matrix with discretized distances between all receptor and ligand atoms
    """
    coords = np.concatenate([receptor_coords, ligand_coords], axis=0)

    # Create distance matrix from coordinates
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(coords[i] - coords[j])
                # Discretize distance using b12 configuration
                discrete_dist = discretize_distance_numpy(distance, discretization_config)
                dist_matrix[i, j] = discrete_dist
            else:
                dist_matrix[i, j] = 0  # self-distance
    
    return dist_matrix

class CoordsDataset(Dataset):
    """Dataset for predicting continuous features from distance matrix."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='coords2')
        self.split = split
        
        with db_connection() as conn:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT id FROM moad_pockets 
                    WHERE split = %s
                    ORDER BY id
                    """,
                    (split,),
                )
                self.ids = [row[0] for row in c.fetchall()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        item_id = self.ids[item]
        cached_data = self.cache.get(item_id)
        
        if cached_data is None:
            with db_connection() as conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT ml.name, mp.pdb, ml.mol 
                        FROM moad_pockets mp
                        JOIN moad_ligands ml ON mp.ligand_name = ml.name
                        WHERE mp.id = %s
                    """, (item_id,))
                    row = c.fetchone()
                    
                    if row is None:
                        raise IndexError(f"Item with id {item_id} not found in database")
                    
                    ligand_name, pocket_pdb, ligand_mol = row
                    features = create_dist_to_coords_features(pocket_pdb, ligand_mol)
                    if features is None:
                        # Skip this sample and try a random one
                        import random
                        random_item = random.randint(0, len(self.ids) - 1)
                        return self.__getitem__(random_item)
                    self.cache.set(item_id, (features, ligand_name))
        else:
            features, ligand_name = cached_data
        
        return self._to_torch(features)

    def _to_torch(self, features):
        # Convert to tensors
        x = torch.tensor(features['x'], dtype=const.TORCH_FLOAT)
        edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
        
        # Add noise to edge distances here (after cache retrieval, before tensor conversion)
        edge_dist = features['edge_dist']
        # for i in range(len(edge_dist)):
        #     # Add random perturbation to discrete distance
        #     perturbation = np.random.choice([-1, 1])
        #     perturbed_dist = edge_dist[i] + perturbation
        #     # Ensure within valid range [0, 11]
        #     edge_dist[i] = max(0, min(11, perturbed_dist))
        
        # Create custom graph
        g = gnn.graph(edge_index, num_nodes=x.shape[0])
        
        # Add node data to ndata
        g.ndata['x'] = x
        g.ndata['h'] = torch.tensor(features['h'], dtype=const.TORCH_FLOAT)
        
        g.edata['edge_dist'] = torch.tensor(edge_dist, dtype=torch.long)
        g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
        
        return g
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for CoordsDataset using custom batch functionality"""
        return gnn.batch(batch_data)



