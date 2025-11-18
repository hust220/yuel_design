import os
import numpy as np
import pickle
import torch
import warnings

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const
from src.pdb_utils import Structure
import src.gnn as gnn

# DGL warnings no longer needed since we use custom graph implementation

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from .cache import FileCache
from .db_utils import db_connection
from .distance_discretization import discretize_distance_numpy
from .const import ALLOWED_PDB_ATOM_TYPES, ALLOWED_ELEMENT_TYPES, PDB_ATOM2IDX, ELEMENT2IDX

# Dataset configurations for different training modes
DATASET_CONFIGS = {
    'dist': {
        'dataset_class': 'DistDataset',
        'no_dist_bins': 12,
        'seq_input_dim': 3 + len(PDB_ATOM2IDX) + len(ELEMENT2IDX),  # mol_types(3) + PDB_ATOM_TYPES + ELEMENT_TYPES
        'z_input_dim': 4,     # z matrix has 4 dimensions
    },
    'coords': {
        'dataset_class': 'CoordsDataset',
        'no_dist_bins': 12,
        'num_node_features': len(ELEMENT2IDX),  # element type dimension
        'num_node_attr_features': 3 + len(ALLOWED_PDB_ATOM_TYPES) + len(ELEMENT2IDX),  # mol_types + PDB atom types + element types
        'num_edge_features': 12 + 3,  # distance + residue
        'n_dims': 3,
    },
}

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def get_pdb_atom_one_hot(atom_name):
    if atom_name in ALLOWED_PDB_ATOM_TYPES:
        return get_one_hot(atom_name, PDB_ATOM2IDX)
    else:
        return get_one_hot('X', PDB_ATOM2IDX)

def get_element_one_hot(element):
    if element in ELEMENT2IDX:
        return get_one_hot(element, ELEMENT2IDX)
    else:
        return get_one_hot('X', ELEMENT2IDX)

def parse_pdb_atom_name(atom_name: str):
    return [get_pdb_atom_one_hot(atom_name), get_element_one_hot(atom_name[:1])]

def parse_mol_atom_name(atom_name: str):
    return [get_pdb_atom_one_hot('X'), get_element_one_hot(atom_name)]

# def parse_structure_and_ligand(pocket_pdb, ligand_size, mask_ligand=True):
def parse_pocket(structure):
    """Parse protein structure and ligand molecule - common function."""
    
    protein_backbone_atoms = ['CA', 'N', 'C', 'O']
    NA_backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    
    bb_coords, coords, mol_types, codes, atom_names, res_ids, res_names = [], [], [], [], [], [], []
    ires = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                # collect backbone atoms for this residue
                protein_backbone_coords, NA_backbone_coords = [], []
                natoms = 0
                for atom in residue:
                    if atom.atom_name[0] == 'H':
                        continue
                    coords.append(atom.get_coord())
                    codes.append(parse_pdb_atom_name(atom.atom_name))  # Store atom name as string
                    atom_names.append(atom.atom_name)  # Store original atom name
                    res_ids.append(ires)
                    res_names.append(residue.res_name)

                    if atom.atom_name in protein_backbone_atoms:
                        protein_backbone_coords.append(atom.get_coord())
                    if atom.atom_name in NA_backbone_atoms:
                        NA_backbone_coords.append(atom.get_coord())

                    natoms += 1

                if len(protein_backbone_coords) >= 4: # this is a protein residue
                    mol_types.extend([[1, 0, 0] for _ in range(natoms)])
                    bb_coords.append(protein_backbone_coords)
                elif len(NA_backbone_coords) >= 5: # this is a NA residue, sometimes P is missing
                    mol_types.extend([[0, 1, 0] for _ in range(natoms)])
                    bb_coords.append(NA_backbone_coords)
                else:
                    mol_types.extend([[0, 0, 1] for _ in range(natoms)])
                    bb_coords.append([])

                ires += 1

    return {
        'bb_coords': bb_coords,
        'coords': coords,
        'mol_types': mol_types,
        'codes': codes,
        'atom_names': atom_names,
        'res_ids': res_ids,
        'residue_names': res_names,
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
    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol)
    ligand_coords = np.array(ligand_coords)
    ligand_size = len(ligand_coords)

    # Create distance matrix
    dist_matrix = create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12')

    features = init_dist_to_coords_features(pocket_info, ligand_size, dist_matrix)
    features['positions'] = np.concatenate([pocket_info['coords'], ligand_coords], axis=0)
    features['one_hot'] = np.array([code[1] for code in pocket_info['codes']] + [get_element_one_hot(atom) for atom in ligand_atoms])

    return features

def init_dist_to_coords_features(pocket_info, ligand_size, dist_matrix):
    bb_coords = pocket_info['bb_coords']
    receptor_coords = np.array(pocket_info['coords'])
    mol_types = pocket_info['mol_types']
    codes = pocket_info['codes']
    receptor_atoms = pocket_info['atom_names']
    receptor_residues = pocket_info['res_ids']
    protein_size = len(receptor_coords)

    n = protein_size + ligand_size
            
    positions = np.concatenate([receptor_coords, np.zeros((ligand_size, 3))], axis=0)
            
    # Create one_hot encoding for atoms (element type only)
    one_hot = [codes[i][1] for i in range(protein_size)] + [get_element_one_hot('X')]*ligand_size
    one_hot = np.array(one_hot)
    
    # Create edge_index, edge_dist, edge_residue, edge_mask
    edge_index, edge_dist, edge_residue, edge_mask = create_graph_from_coords(receptor_coords, ligand_size, dist_matrix, receptor_residues, bb_coords, receptor_atoms)
    
    # node_attr
    # mol_types = np.concatenate([mol_types, np.array([[0, 0, 1]]*ligand_size)])
    # codes = np.concatenate([codes, np.array([parse_mol_atom_name('X')]*ligand_size)])
    node_attr = create_node_attributes(mol_types, codes, ligand_size)

    # node_mask and anchor_mask
    node_mask = np.ones(n)  # All nodes are valid
    anchor_mask = np.zeros(n)  # No anchors (all atoms can be generated)
    
    features = {
        'name': f'PL_{n}',
        'positions': positions,
        'one_hot': one_hot,
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_residue': edge_residue,
        'node_attr': node_attr,
        'node_mask': node_mask,
        'anchor_mask': anchor_mask,
        'edge_mask': edge_mask,
    }
    
    return features

def create_z_matrix(receptor_residues, bb_coords, receptor_coords, receptor_atoms, ligand_size, add_backbone_distance=False):
    """Create z matrix for residue relationships - shared function."""
    n = len(receptor_residues) + ligand_size
    res_ids = receptor_residues + [receptor_residues[-1]+1]*ligand_size
    bb_coords.append([])
    atom_names = receptor_atoms + ['X']*ligand_size
    z_matrix = np.zeros((n, n, 4 if add_backbone_distance else 3), dtype=np.float32)  # Add distance dimension
    for i in range(n):
        for j in range(n):
            if i != j:
                res_i, res_j = res_ids[i], res_ids[j]
                if res_i == res_j:
                    z_matrix[i, j, 1] = 1 # same_residue
                else:
                    bb_1, bb_2 = bb_coords[res_i], bb_coords[res_j]
                    d = min_dist(bb_1, bb_2)
                    if len(bb_1) > 0 and len(bb_2) > 0 and d < 2.0:
                        z_matrix[i, j, 2] = 1 # neighbor_residue
                    else:
                        z_matrix[i, j, 0] = 1 # distant_residue
                
                # Add distance for CA or C4' atoms (any combination)
                if add_backbone_distance and (atom_names[i] == 'CA' or atom_names[i] == "C4'") and \
                   (atom_names[j] == 'CA' or atom_names[j] == "C4'"):
                    # Calculate distance between CA or C4' atoms
                    dist = np.linalg.norm(receptor_coords[i] - receptor_coords[j])
                    z_matrix[i, j, 3] = dist  # No noise here, will be added in _to_torch
    return z_matrix

def create_graph_from_coords(receptor_coords, ligand_size, dist_matrix, receptor_residues, bb_coords, receptor_atoms):
    """Create graph structure from coordinates and distance matrix."""
    n = len(receptor_coords) + ligand_size
    edge_list, edge_dists, edge_residues, edge_masks = [], [], [], []

    # Use shared function to create z matrix
    z_matrix = create_z_matrix(receptor_residues, bb_coords, receptor_coords, receptor_atoms, ligand_size, add_backbone_distance=False)
    
    for i in range(n):
        for j in range(i+1, n):
            # Use distance matrix for edge creation
            edge_list.append([i, j])
            edge_list.append([j, i])  # Bidirectional
            
            # Store discrete distance as integer (no one-hot, no noise yet)
            discrete_dist = int(dist_matrix[i, j])
            
            # Residue relationship features - convert to categorical
            residue_features = z_matrix[i, j]  # [distant_residue, same_residue, neighbor_residue]
            # Convert one-hot to categorical: [0, 1, 0] -> 1, [1, 0, 0] -> 0, [0, 0, 1] -> 2
            residue_category = np.argmax(residue_features)
            
            # Store distance and residue separately
            edge_dists.append(discrete_dist)
            edge_dists.append(discrete_dist)
            edge_residues.append(residue_category)
            edge_residues.append(residue_category)
            
            # Edge mask: 1 if edge is valid
            edge_masks.append(1.0)
            edge_masks.append(1.0)
        
    return np.array(edge_list).T, np.array(edge_dists), np.array(edge_residues), np.array(edge_masks)

def create_node_attributes(mol_types, codes, ligand_size):
    """Create node attributes for EDM2."""
    # Combine molecular type, PDB atom type, and element type as node attributes
    node_attrs = []
    for i in range(len(mol_types)):
        node_attrs.append(np.concatenate([mol_types[i], codes[i][0], codes[i][1]]))

    for i in range(ligand_size):
        node_attrs.append(np.concatenate([[0, 0, 1], get_pdb_atom_one_hot('X'), get_element_one_hot('X')]))
    
    return np.array(node_attrs)

def min_dist(coords1, coords2):
    d = float('inf')
    for coord1 in coords1:
        for coord2 in coords2:
            dist = np.linalg.norm(coord1 - coord2)
            d = min(d, dist)
    return d

def get_ligand_atoms_and_coords(mol):
    """Get ligand atoms and coordinates from molecule."""
    ligand_atoms = []
    ligand_coords = []
    
    # Get the conformer from the molecule
    conf = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        ligand_atoms.append(atom.GetSymbol())
        ligand_coords.append(conf.GetAtomPosition(atom.GetIdx()))
    
    return ligand_atoms, ligand_coords

def create_dist_features(pocket_pdb, ligand_mol):
    """Create distance-based features from protein pocket and ligand."""
    # seq: [is_protein, is_NA, is_other, element_one_hot, atom_one_hot] ligand element and atom are always zero
    # z: [distant_residue, same_residue, neighbor_residue, backbone_distance]
    # dist: [distance_discretization]

    # Parse ligand molecule
    mol = Chem.MolFromMolBlock(ligand_mol)
    if mol is None:
        # Return None to skip this sample
        return None

    # Parse protein pocket structure
    structure = Structure()
    from io import StringIO
    structure.read(StringIO(pocket_pdb))

    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol)
    ligand_coords = np.array(ligand_coords)
    ligand_size = len(ligand_atoms)
    
    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        return None

    dist_matrix = create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12')

    features = init_dist_features(pocket_info, ligand_size)

    features['dist'] = dist_matrix

    return features

def init_dist_features(pocket_info, ligand_size):
    bb_coords = pocket_info['bb_coords']
    receptor_coords = np.array(pocket_info['coords'])
    mol_types = pocket_info['mol_types']
    codes = pocket_info['codes']
    receptor_atoms = pocket_info['atom_names']
    receptor_residues = pocket_info['res_ids']
    protein_size = len(receptor_coords)

    mol_types.extend([[0, 0, 1]]*ligand_size)
    codes.extend([parse_mol_atom_name('X')]*ligand_size)
    seq = np.concatenate([mol_types, np.array([code[0] for code in codes]), np.array([code[1] for code in codes])], axis=1)
            
    # Create z matrix: [distant_residue, same_residue, neighbor_residue, distance]
    z_matrix = create_z_matrix(receptor_residues, bb_coords, receptor_coords, receptor_atoms, ligand_size, add_backbone_distance=True)
                
    # Create masks
    n = protein_size + ligand_size
    seq_mask = np.ones(n)
    pair_mask = np.ones((n, n))

    features = {
        'seq': seq,
        'z': z_matrix,
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
    }
    
    return features

def create_dist_matrix(receptor_coords, ligand_coords, discretization_config='b12'):
    """Helper function to create distance matrix from coordinates"""

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

class DistDataset(Dataset):
    """Dataset for distance-based protein-ligand interaction prediction."""
    
    def __init__(self, split='train', cache_mode='memory', cache_dir='cache', no_dist_bins=12):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='dist')
        self.split = split
        self.no_dist_bins = no_dist_bins
        
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
                    features = create_dist_features(pocket_pdb, ligand_mol)
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
        # Convert to tensor first
        z = torch.tensor(features['z'], dtype=const.TORCH_FLOAT)
        
        if self.split == 'train':  # Only add noise during training
            # Add random noise 0-2 to backbone distances (4th dimension)
            # Create noise tensor with same shape as z
            noise = torch.rand_like(z) * 2  # Random noise 0-2
            # Only add noise to backbone distances (4th dimension) where z > 0
            backbone_mask = (z[:, :, 3] > 0).unsqueeze(-1)  # [N, N, 1]
            z = z + noise * backbone_mask
        
        return {
            'seq': torch.tensor(features['seq'], dtype=const.TORCH_FLOAT),
            'z': z,
            'dist': torch.tensor(features['dist'], dtype=const.TORCH_INT),
            'seq_mask': torch.tensor(features['seq_mask'], dtype=const.TORCH_FLOAT),
            'pair_mask': torch.tensor(features['pair_mask'], dtype=const.TORCH_FLOAT),
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function for DistDataset"""
        # batch: list of dicts with variable N
        max_n = max(sample['seq'].size(0) for sample in batch)
        
        # Determine feature sizes
        bsz = len(batch)
        feat_seq = batch[0]['seq'].size(-1)
        feat_z = batch[0]['z'].size(-1)
        
        seq = torch.zeros(bsz, max_n, feat_seq, dtype=torch.float32)
        z = torch.zeros(bsz, max_n, max_n, feat_z, dtype=torch.float32)
        dist = torch.zeros(bsz, max_n, max_n, dtype=torch.int64)
        seq_mask = torch.zeros(bsz, max_n, dtype=torch.float32)
        pair_mask = torch.zeros(bsz, max_n, max_n, dtype=torch.float32)
        
        for i, sample in enumerate(batch):
            n = sample['seq'].size(0)
            seq[i, :n] = sample['seq']
            z[i, :n, :n] = sample['z']
            dist[i, :n, :n] = sample['dist']
            seq_mask[i, :n] = sample['seq_mask']
            pair_mask[i, :n, :n] = sample['pair_mask']
        
        return {
            'seq': seq,
            'z': z,
            'dist': dist,
            'seq_mask': seq_mask,
            'pair_mask': pair_mask,
        }

class CoordsDataset(Dataset):
    """Dataset for predicting coordinates from distance matrix."""
    
    def __init__(self, split='train', cache_mode='memory', cache_dir='cache', no_dist_bins=12):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='coords')
        self.split = split
        self.no_dist_bins = no_dist_bins
        
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
        positions = torch.tensor(features['positions'], dtype=const.TORCH_FLOAT)
        edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
        
        # Add noise to edge distances here (after cache retrieval, before tensor conversion)
        edge_dist = features['edge_dist'].copy()
        for i in range(len(edge_dist)):
            # Add random perturbation to discrete distance
            perturbation = np.random.choice([-1, 1])
            perturbed_dist = edge_dist[i] + perturbation
            # Ensure within valid range [0, 11]
            edge_dist[i] = max(0, min(11, perturbed_dist))
        
        # Create custom graph
        g = gnn.graph(edge_index, num_nodes=positions.shape[0])
        
        # Add node data to ndata
        g.ndata['positions'] = positions
        g.ndata['one_hot'] = torch.tensor(features['one_hot'], dtype=const.TORCH_FLOAT)
        g.ndata['node_attr'] = torch.tensor(features['node_attr'], dtype=const.TORCH_FLOAT)
        g.ndata['node_mask'] = torch.tensor(features['node_mask'], dtype=const.TORCH_FLOAT)
        g.ndata['anchor_mask'] = torch.tensor(features['anchor_mask'], dtype=const.TORCH_FLOAT)
        
        # Add edge data to edata - separate distance and residue attributes
        g.edata['edge_dist'] = torch.tensor(edge_dist, dtype=torch.long)
        g.edata['edge_residue'] = torch.tensor(features['edge_residue'], dtype=torch.long)
        g.edata['edge_mask'] = torch.tensor(features['edge_mask'], dtype=const.TORCH_FLOAT)
        
        return g
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for CoordsDataset using custom batch functionality"""
        
        # Batch is now a list of custom graphs with attributes
        graphs = batch_data
        
        # Use custom batch functionality
        batched_graph = gnn.batch(graphs)
        
        return batched_graph


def collate(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['protein_mask'].bool() | out['ligand_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False, num_workers=0, device=None):
    return DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )


