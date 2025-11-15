import numpy as np
import torch

from torch.utils.data import Dataset
from src import const
from src.pdb_utils import Structure
import src.gnn as gnn
from src.cache import FileCache
from src.db_utils import db_connection
from src.distance_discretization import discretize_distance_numpy
from src.const import PROTEIN_ATOM_TYPES

# Define unified atom types for coarse-grained representation
# Protein atoms: CA + all non-C PDB atoms (from src.const.PROTEIN_ATOM_TYPES)

# Protein non-C atoms: extract all non-C atoms from PDB_ATOM_TYPES
# This includes: N, O, S, P, ND1, OD1, SG, etc.
protein_non_c_atoms = []
for element, atom_names in PROTEIN_ATOM_TYPES.items():
    if element != 'C':
        protein_non_c_atoms.extend(atom_names)

# Ring center virtual atoms (by ring size)
RING_CENTERS = ['RING_3', 'RING_4', 'RING_5', 'RING_6', 'RING_X']  # 3, 4, 5, 6, and larger rings

# Protein atom types: CA + all non-C atoms + ring center virtual atoms
protein_atoms = ['CA'] + protein_non_c_atoms + RING_CENTERS

# Combined atom types: X (unknown) + protein atoms
ALLOWED_ATOM_TYPES = ['X'] + protein_atoms

# Create mappings
ATOM2IDX = {atom: idx for idx, atom in enumerate(ALLOWED_ATOM_TYPES)}

def get_one_hot(atom, atoms_dict):
    """Get one-hot encoding for atom."""
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def get_atom_one_hot(atom_name):
    """Get one-hot encoding for atom name.
    Supports protein atoms (CA, ND1, OD1, SG, etc.) and ring centers.
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
            - full_coords: Coordinates of all atoms (CA, non-C, C atoms, and ring centers)
            - atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
              where reduced_atoms are CA + all non-C atoms + ring centers,
              and full_atoms are all atoms (CA, non-C, C atoms, and ring centers)
            - residues: List of tuples, each tuple is (residue_name, residue_id, chain_id)
    """
    reduced_coords_flat = []
    full_coords_flat = []
    atoms = []
    residues = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.chain_id
            for residue in chain:
                reduced_atoms, reduced_coords = [], []
                full_atoms, full_coords = [], []

                res_name = residue.res_name
                res_id = residue.res_id  # residue sequence number
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
                    atoms.append((reduced_atoms, full_atoms))
                    residues.append((res_name, res_id, chain_id))
    
    return {
        'reduced_coords': np.array(reduced_coords_flat),
        'full_coords': np.array(full_coords_flat),
        'atoms': atoms,
        'residues': residues,
    }

def create_receptor_features(pocket_pdb):
    """Create features from receptor (protein) structure only.
    
    Args:
        pocket_pdb: PDB string of the protein structure
    
    Returns:
        dict with keys 'x', 'h', 'edge_index', 'edge_dist', 'edge_attr'
    """
    structure = Structure()
    from io import StringIO
    structure.read(StringIO(pocket_pdb))

    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        return None

    # Get reduced coords and full coords from pocket_info
    receptor_atoms = pocket_info['atoms']
    receptor_reduced_coords = pocket_info['reduced_coords']
    receptor_full_coords = pocket_info['full_coords']
    
    # Create distance matrix for reduced atoms (CA + non-C + ring centers)
    dist_matrix = create_dist_matrix(receptor_reduced_coords, discretization_config='b12')
    
    features = init_receptor_features(
        dist_matrix=dist_matrix,
        receptor_atoms=receptor_atoms
    )
    features['x'] = receptor_full_coords

    return features

def init_receptor_features(dist_matrix, receptor_atoms):
    """Initialize features from distance matrix and receptor information.
    
    Args:
        dist_matrix: Reduced distance matrix with shape (n_reduced_receptor, n_reduced_receptor)
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
    
    Returns:
        dict with features including 'h', 'edge_index', 'edge_dist', 'edge_attr', 'mask'
    """
    # Calculate total number of reduced receptor atoms
    n_reduced_receptor = sum(len(reduced_atoms) for reduced_atoms, _ in receptor_atoms)
    assert n_reduced_receptor == dist_matrix.shape[0], \
        f"n_reduced_receptor ({n_reduced_receptor}) must equal dist_matrix.shape[0] ({dist_matrix.shape[0]})"
    
    # Expand distance matrix to include all receptor atoms
    full_dist_matrix, full_receptor_atoms = expand_dist_matrix(
        dist_matrix, receptor_atoms
    )
    node_attr = build_nodes(full_receptor_atoms)
    edge_index, edge_dist, edge_attr = build_edges(
        dist_matrix=full_dist_matrix,
        receptor_atoms=receptor_atoms
    )
    
    # Create mask for CA atoms
    mask = np.array([1 if atom_name == 'CA' else 0 for atom_name in full_receptor_atoms], dtype=np.int64)
    
    features = {
        'h': node_attr,
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_attr': edge_attr,
        'mask': mask,
    }
    
    return features

def expand_dist_matrix(dist_matrix, receptor_atoms):
    """Expand reduced distance matrix (CA, ring centers, non-C atoms) to full node set.
    
    dist_matrix has shape (n_reduced_receptor, n_reduced_receptor)
    where n_reduced_receptor is the number of CA + non-C atoms + ring centers.
    Returns full matrix with shape (n_receptor, n_receptor)
    where n_receptor is the total number of receptor atoms (CA + non-C atoms + C atoms + ring centers).
    
    Args:
        dist_matrix: Reduced distance matrix with shape (n_reduced_receptor, n_reduced_receptor)
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
    
    Returns:
        tuple: (full_dist_matrix, full_receptor_atoms)
            full_dist_matrix: Full distance matrix with shape (n_receptor, n_receptor)
            full_receptor_atoms: List of all receptor atom names
    """
    # Calculate total number of reduced receptor atoms
    n_reduced_receptor = sum(len(reduced_atoms) for reduced_atoms, _ in receptor_atoms)
    assert n_reduced_receptor == dist_matrix.shape[0], \
        f"n_reduced_receptor ({n_reduced_receptor}) must equal dist_matrix.shape[0] ({dist_matrix.shape[0]})"
    
    # Build full receptor atom list and mapping from reduced to full indices
    full_receptor_atoms = []
    reduced_to_full_map = {}  # Map from reduced receptor index to full receptor index
    
    reduced_idx = 0
    full_idx_offset = 0
    
    # Process each residue in receptor_atoms
    for reduced_atoms, full_atoms in receptor_atoms:
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
    full_dist_matrix = np.zeros((n_receptor, n_receptor), dtype=np.int64)
    
    # Build complete index mapping from reduced to full indices
    full_indices = np.zeros(n_reduced_receptor, dtype=np.int64)
    
    # Map reduced receptor indices to full receptor indices
    for i_reduced, i_full in reduced_to_full_map.items():
        full_indices[i_reduced] = i_full
    
    # Use advanced indexing to copy all distances at once
    i_reduced = np.arange(n_reduced_receptor)[:, None]
    j_reduced = np.arange(n_reduced_receptor)[None, :]
    i_full = full_indices[:, None]
    j_full = full_indices[None, :]
    full_dist_matrix[i_full, j_full] = dist_matrix[i_reduced, j_reduced] + 1
    
    return full_dist_matrix, full_receptor_atoms

def build_edges(dist_matrix, receptor_atoms):
    """Build graph edges from distance matrix.
    Creates a fully connected graph with residue relationship features.
    
    Args:
        dist_matrix: Distance matrix with shape (n_receptor, n_receptor)
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
    
    Returns:
        tuple: (edge_index, edge_dist, edge_attr)
    """
    # Build residue indices for each receptor atom
    res_indices = []
    for res_idx, (reduced_atoms, full_atoms) in enumerate(receptor_atoms):
        res_indices.extend([res_idx] * len(full_atoms))
    
    n_receptor = len(res_indices)
    assert dist_matrix.shape[0] == n_receptor, f"dist_matrix shape {dist_matrix.shape[0]} must equal n_receptor = {n_receptor}"
    
    edge_list, edge_dist, edge_attr = [], [], []
    
    for i in range(n_receptor):
        for j in range(i+1, n_receptor):
            # Bidirectional edges
            edge_list.append([i, j])
            edge_list.append([j, i])
            
            # Store discrete distance from dist_matrix
            discrete_dist = int(dist_matrix[i, j])
            edge_dist.append(discrete_dist)
            edge_dist.append(discrete_dist)
            
            res_i, res_j = res_indices[i], res_indices[j]
            is_same_residue = 1 if res_i == res_j else 0
            edge_attr.append([is_same_residue])
            edge_attr.append([is_same_residue])
    
    return np.array(edge_list).T, np.array(edge_dist), np.array(edge_attr)

def build_nodes(receptor_atoms):
    """Build node attributes from receptor atoms.
    
    Args:
        receptor_atoms: List of receptor atom names (e.g., 'CA', 'RING_3', etc.)
    
    Returns:
        numpy array of node attributes with shape (n_receptor, node_feat_dim)
    """
    node_attrs = []
    
    # Build receptor node attributes
    for atom_name in receptor_atoms:
        # Determine mol_type based on atom name
        if atom_name == 'CA':
            mol_type = [1, 0]  # backbone
        else:
            mol_type = [0, 1]  # side chain or ring center
        
        atom_one_hot = get_atom_one_hot(atom_name)
        node_attrs.append(np.concatenate([mol_type, atom_one_hot]))
    
    return np.array(node_attrs)

def create_dist_matrix(receptor_coords, discretization_config='b12'):
    """Helper function to create distance matrix from receptor coordinates.
    
    Args:
        receptor_coords: Array of receptor atom coordinates (CA, non-C atoms, and ring centers)
        discretization_config: Distance discretization configuration
        
    Returns:
        Distance matrix with discretized distances between all receptor atoms
    """
    # Create distance matrix from coordinates
    n = len(receptor_coords)
    dist_matrix = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(receptor_coords[i] - receptor_coords[j])
                # Discretize distance using b12 configuration
                discrete_dist = discretize_distance_numpy(distance, discretization_config)
                dist_matrix[i, j] = discrete_dist
            else:
                dist_matrix[i, j] = 0  # self-distance
    
    return dist_matrix

class SidechainDataset(Dataset):
    """Dataset for predicting continuous features from distance matrix."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='sidechain')
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
                        SELECT pdb 
                        FROM moad_pockets 
                        WHERE id = %s
                    """, (item_id,))
                    row = c.fetchone()
                    
                    if row is None:
                        raise IndexError(f"Item with id {item_id} not found in database")
                    
                    pocket_pdb = row[0]
                    features = create_receptor_features(pocket_pdb)
                    if features is None:
                        # Skip this sample and try a random one
                        import random
                        random_item = random.randint(0, len(self.ids) - 1)
                        return self.__getitem__(random_item)
                    self.cache.set(item_id, features)
        else:
            features = cached_data
        
        return self._to_torch(features)

    @staticmethod
    def features_to_graph(features):
        """Convert features dict to graph.
        
        Args:
            features: dict with keys 'x', 'edge_index', 'h', 'edge_dist', 'edge_attr', 'mask'
        
        Returns:
            Graph object (from src.gnn) with node and edge data
        """
        x = torch.tensor(features['x'], dtype=const.TORCH_FLOAT)
        edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
        
        edge_dist = features['edge_dist']
        
        g = gnn.graph(edge_index, num_nodes=x.shape[0])
        
        g.ndata['x'] = x
        g.ndata['h'] = torch.tensor(features['h'], dtype=const.TORCH_FLOAT)
        g.ndata['mask'] = torch.tensor(features['mask'], dtype=torch.long)
        
        g.edata['edge_dist'] = torch.tensor(edge_dist, dtype=torch.long)
        g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
        
        return g
    
    def _to_torch(self, features):
        return self.features_to_graph(features)
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for SidechainDataset using custom batch functionality"""
        return gnn.batch(batch_data)



