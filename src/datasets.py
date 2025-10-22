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

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def get_atom_one_hot(residue, atom_name):
    rna_atom = f'{residue}_{atom_name}'
    residue_one_hot = get_one_hot(residue, const.RESIDUE2IDX)
    atom_one_hot = get_one_hot(rna_atom, const.RNA_ATOM2IDX)
    return np.concatenate([residue_one_hot, atom_one_hot])

def get_positional_encoding(n, d_model=32):
    pos_encoding = np.zeros((n, d_model))
    position = np.arange(n)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

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

def create_distance_matrix(c4_coords):
    n = len(c4_coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(c4_coords[i] - c4_coords[j])
    return distance_matrix


def parse_residues(rs):
    """Parse residues to extract coordinates and types."""
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.res_name
        
        for atom in residue:
            atom_name = atom.atom_name
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                pocket_types.append(residue_name)

    return {
        'coord': pocket_coords,
        'types': pocket_types,
    }

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule

# one hot for atoms
def atom_one_hot(atom):
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    one_hot = np.zeros(n1 + n2)
    one_hot[n1 + const.ATOM2IDX[atom]] = 1
    return one_hot

# one hot for protein atoms
def protein_atom_one_hot(atom_name):
    """Create one-hot encoding for protein atom names."""
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    n3 = const.N_PROTEIN_ATOM_TYPES
    one_hot = np.zeros(n1 + n2 + n3)
    
    if atom_name in const.PROTEIN_ATOM2IDX:
        one_hot[n1 + n2 + const.PROTEIN_ATOM2IDX[atom_name]] = 1
    else:
        # Handle unknown protein atoms by using a default encoding
        one_hot[n1 + n2] = 1  # Use the first position for unknown atoms
    
    return one_hot

# one hot for amino acids
def aa_one_hot(residue):
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    one_hot = np.zeros(n1 + n2)
    one_hot[const.RESIDUE2IDX[residue]] = 1
    return one_hot

def molecule_feat_mask():
    n1 = const.N_RESIDUE_TYPES
    n2 = const.N_ATOM_TYPES
    mask = np.zeros(n1 + n2)
    mask[n1:] = 1
    return mask

def parse_molecule(mol):
    one_hot = []
    for atom in mol.GetAtoms():
        one_hot.append(atom_one_hot(atom.GetSymbol()))
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot)

def parse_pocket(rs):
    """Parse pocket residues to extract coordinates and types."""
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.res_name
        
        for atom in residue:
            atom_name = atom.atom_name
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                pocket_types.append(residue_name)

    pocket_one_hot = []
    for _type in pocket_types:
        pocket_one_hot.append(aa_one_hot(_type))
    pocket_one_hot = np.array(pocket_one_hot)

    return pocket_coords, pocket_one_hot

def get_pocket(mol, pdb_path):
    """Extract protein pocket residues that are in contact with the ligand."""
    structure = Structure(pdb_path)
    residue_ids = []
    atom_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.get_coord())
                    residue_ids.append(len(residue_ids))

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(axis=1) <= 6)[0]])

    # Get contact residues using pdb_utils
    contact_residue_list = []
    residue_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue_idx in contact_residues:
                    contact_residue_list.append(residue)
                residue_idx += 1

    return parse_pocket(contact_residue_list)

import numpy as np

def pad_and_concatenate(tensor1, tensor2):
    N, a = tensor1.shape
    M, b = tensor2.shape
    
    # Pad tensor1 with zeros for the b columns it's missing
    tensor1_padded = np.pad(tensor1, 
                           pad_width=((0, 0), (0, b)),  # Pad b zeros on the right
                           mode='constant',
                           constant_values=0)
    
    # Pad tensor2 with zeros for the a columns it's missing
    tensor2_padded = np.pad(tensor2,
                           pad_width=((0, 0), (a, 0)),  # Pad a zeros on the left
                           mode='constant',
                           constant_values=0)
    
    # Concatenate along the first axis (stack vertically)
    return np.concatenate([tensor1_padded, tensor2_padded], axis=0)

class ProteinLigandDataset(Dataset):
    def __init__(self, data=None, data_path=None, prefix=None, device=None):
        assert (data is not None) or all(x is not None for x in (data_path, prefix, device))
        if data is not None:
            self.data = data
            return

        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        if os.path.exists(dataset_path):
            print(f'Found dataset: {dataset_path}')
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = self.preprocess(data_path, prefix, device)
            print(f'Saving dataset as {dataset_path}')
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data_path = os.path.join(data_path, f'{prefix}.pkl')

        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        generator = tqdm(
            raw_data,
            total=len(raw_data)
        )
        irow = 0
        data = []
        for row in generator:
            molecule_name = row['molecule']

            positions = row['molecule_pos']
            one_hot = row['molecule_one_hot']
            # print(111,one_hot)

            mol_size = len(positions)
            pocket_size = 0

            if 'pocket_pos' in row and 'pocket_one_hot' in row:
                pocket_pos = row['pocket_pos']
                pocket_one_hot = row['pocket_one_hot']
                positions = np.concatenate([pocket_pos, positions], axis=0)
                one_hot = np.concatenate([pocket_one_hot, one_hot], axis=0)
                pocket_size = len(pocket_pos)

            if len(positions) > 150:
                print(f'Skipping molecule {molecule_name} with {len(positions)} atoms')
                continue

            protein_mask = np.zeros(pocket_size + mol_size)
            ligand_mask = np.zeros(pocket_size + mol_size)
            protein_mask[:pocket_size] = 1
            ligand_mask[pocket_size:] = 1

            data.append({
                'name': molecule_name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'protein_mask': torch.tensor(protein_mask, dtype=const.TORCH_FLOAT, device=device),
                'ligand_mask': torch.tensor(ligand_mask, dtype=const.TORCH_FLOAT, device=device),
            })

        return data

class DistDataset(Dataset):
    """Dataset for distance-based protein-ligand interaction prediction."""
    
    def __init__(self, split='train', cache_mode='memory', cache_dir='cache', no_dist_bins=12):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir)
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
        return {
            'seq': torch.tensor(features['seq'], dtype=const.TORCH_FLOAT),
            'z': torch.tensor(features['z'], dtype=const.TORCH_FLOAT),
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
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir)
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
        
        # Create custom graph
        g = gnn.graph(edge_index, num_nodes=positions.shape[0])
        
        # Add node data to ndata
        g.ndata['positions'] = positions
        g.ndata['one_hot'] = torch.tensor(features['one_hot'], dtype=const.TORCH_FLOAT)
        g.ndata['node_attr'] = torch.tensor(features['node_attr'], dtype=const.TORCH_FLOAT)
        g.ndata['node_mask'] = torch.tensor(features['node_mask'], dtype=const.TORCH_FLOAT)
        g.ndata['anchor_mask'] = torch.tensor(features['anchor_mask'], dtype=const.TORCH_FLOAT)
        
        # Add edge data to edata
        g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=const.TORCH_FLOAT)
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

def parse_structure_and_ligand(pocket_pdb, ligand_mol, mask_ligand=True):
    """Parse protein structure and ligand molecule - common function."""
    # Parse protein pocket structure
    structure = Structure()
    
    # pocket_pdb is a string containing PDB file content
    from io import StringIO
    structure.read(StringIO(pocket_pdb))
    
    protein_backbone_atoms = ['CA', 'N', 'C', 'O']
    NA_backbone_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    
    bb_coords, coords, mol_types, codes, atom_names, res_ids = [], [], [], [], [], []
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

    # Parse ligand molecule
    mol = Chem.MolFromMolBlock(ligand_mol)
    if mol is None:
        # Return None to skip this sample
        return None
    
    # Extract ligand coordinates and atom codes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
        mol_types.append([0, 0, 1])
        res_ids.append(ires)

        if mask_ligand:
            codes.append(parse_mol_atom_name('X'))
            atom_names.append('X')
        else:
            codes.append(parse_mol_atom_name(atom.GetSymbol()))
            atom_names.append(atom.GetSymbol())
    bb_coords.append([])

    return bb_coords, coords, mol_types, codes, atom_names, res_ids

def create_dist_to_coords_features(pocket_pdb, ligand_mol):
    """Create features for distance-to-coordinates prediction."""
    result = parse_structure_and_ligand(pocket_pdb, ligand_mol, False)
    if result is None:
        return None
    
    bb_coords, coords, mol_types, codes, atom_names, res_ids = result
    
    res_ids = np.array(res_ids)
    coords = np.array(coords)
    mol_types = np.array(mol_types)
    n = len(coords)
            
    # Create distance matrix
    dist_matrix = create_dist_matrix(coords, discretization_config='b12')
    
    # Determine protein and ligand sizes
    protein_size = 0
    for i in range(n):
        if mol_types[i, 0] == 1:  # protein
            protein_size += 1
    
    # Create protein and ligand masks
    protein_mask = np.zeros(n)
    ligand_mask = np.zeros(n)
    protein_mask[:protein_size] = 1
    ligand_mask[protein_size:] = 1
    
    # Create one_hot encoding for atoms (element type only)
    one_hot = []
    for i in range(n):
        one_hot.append(codes[i][1])
    one_hot = np.array(one_hot)
    
    # Create EDM2 required graph structure
    edge_index, edge_attr, edge_mask = create_graph_from_coords(coords, dist_matrix, res_ids, bb_coords, atom_names, mol_types)
    node_attr = create_node_attributes(mol_types, codes)
    node_mask = np.ones(n)  # All nodes are valid
    anchor_mask = np.zeros(n)  # No anchors (all atoms can be generated)
    
    features = {
        'name': f'protein_ligand_{n}',
        'positions': coords,
        'one_hot': one_hot,
        # EDM2 required fields
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_attr': node_attr,
        'node_mask': node_mask,
        'anchor_mask': anchor_mask,
        'edge_mask': edge_mask,
    }
    
    return features

def create_z_matrix(res_ids, bb_coords, coords, atom_names, add_backbone_distance=False):
    """Create z matrix for residue relationships - shared function."""
    n = len(res_ids)
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
                    dist = np.linalg.norm(coords[i] - coords[j])
                    # Add random noise 0-2
                    noise = np.random.uniform(0, 2)
                    z_matrix[i, j, 3] = dist + noise
    return z_matrix

def get_element_type(atom_name, mol_type):
    """Get element type one-hot encoding, with ligand elements set to zero."""
    if mol_type[0] == 1:  # protein atom
        # Extract element from atom name
        element = atom_name[:1] if atom_name else 'X'
        return get_element_one_hot(element)
    else:  # ligand atom
        # Return zero vector for ligand atoms
        return get_element_one_hot('X')

def create_graph_from_coords(coords, dist_matrix, res_ids, bb_coords, atom_names, mol_types):
    """Create graph structure from coordinates and distance matrix."""
    n = len(coords)
    edge_list = []
    edge_attrs = []
    edge_masks = []
    
    # Use shared function to create z matrix
    z_matrix = create_z_matrix(res_ids, bb_coords, coords, atom_names, add_backbone_distance=False)
    
    for i in range(n):
        for j in range(i+1, n):
            # Use distance matrix for edge creation
            if dist_matrix[i, j] > 0:  # Valid distance
                edge_list.append([i, j])
                edge_list.append([j, i])  # Bidirectional
                
                # Create edge attributes: discretized distance one-hot + residue relationships
                discrete_dist = int(dist_matrix[i, j])
                
                # Add random perturbation to discrete distance
                # Randomly add or subtract 1
                perturbation = np.random.choice([-1, 1])
                perturbed_dist = discrete_dist + perturbation
                # Ensure within valid range [0, 11]
                perturbed_dist = max(0, min(11, perturbed_dist))
                
                dist_one_hot = np.zeros(12)  # Assuming 12 distance bins
                if 0 <= perturbed_dist < 12:
                    dist_one_hot[perturbed_dist] = 1
                
                # Residue relationship features
                residue_features = z_matrix[i, j]  # [distant_residue, same_residue, neighbor_residue]
                
                # Combine features: distance + residue
                edge_attr = np.concatenate([dist_one_hot, residue_features])
                edge_attrs.append(edge_attr)
                edge_attrs.append(edge_attr)
                
                # Edge mask: 1 if edge is valid
                edge_masks.append(1.0)
                edge_masks.append(1.0)
    
    if len(edge_list) == 0:
        # Fallback: create a minimal graph
        edge_list = [[0, 1], [1, 0]] if n > 1 else [[0, 0]]
        # Create default edge attributes
        default_dist_one_hot = np.zeros(12)
        default_dist_one_hot[0] = 1  # First bin
        default_residue_features = np.array([1, 0, 0])  # distant_residue
        default_edge_attr = np.concatenate([default_dist_one_hot, default_residue_features])
        edge_attrs = [default_edge_attr, default_edge_attr] if n > 1 else [default_edge_attr]
        edge_masks = [1.0, 1.0] if n > 1 else [1.0]
    
    return np.array(edge_list).T, np.array(edge_attrs), np.array(edge_masks)

def create_node_attributes(mol_types, codes):
    """Create node attributes for EDM2."""
    # Combine molecular type, PDB atom type, and element type as node attributes
    node_attrs = []
    for i in range(len(mol_types)):
        # For ligand atoms (mol_types[i][2] == 1), use 'X' for element type in node_attr
        if mol_types[i][2] == 1:  # ligand atom
            element_type = get_element_one_hot('X')  # Use 'X' for ligand atoms
        else:  # protein/NA atoms
            element_type = codes[i][1]  # Use real element type
        
        # Concatenate mol_types, PDB atom type, and element type
        node_attr = np.concatenate([mol_types[i], codes[i][0], element_type])
        node_attrs.append(node_attr)
    
    return np.array(node_attrs)

def min_dist(coords1, coords2):
    d = float('inf')
    for coord1 in coords1:
        for coord2 in coords2:
            dist = np.linalg.norm(coord1 - coord2)
            d = min(d, dist)
    return d

def create_dist_features(pocket_pdb, ligand_mol):
    """Create distance-based features from protein pocket and ligand."""
    # seq: [is_protein, is_NA, is_other, element_one_hot, atom_one_hot] ligand element and atom are always zero
    # z: [distant_residue, same_residue, neighbor_residue, backbone_distance]
    # dist: [distance_discretization]
    
    result = parse_structure_and_ligand(pocket_pdb, ligand_mol, True)
    if result is None:
        # Return None to skip this sample
        return None
    
    bb_coords, coords, mol_types, codes, atom_names, res_ids = result
    
    res_ids = np.array(res_ids)
    coords = np.array(coords)
    mol_types = np.array(mol_types)
    n = len(coords)
    seq = np.concatenate([mol_types, np.array([code[0] for code in codes]), np.array([code[1] for code in codes])], axis=1)
            
    # Create distance matrix
    dist_matrix = create_dist_matrix(coords, discretization_config='b12')
    
    # Create z matrix: [distant_residue, same_residue, neighbor_residue, distance]
    z_matrix = create_z_matrix(res_ids, bb_coords, coords, codes, add_backbone_distance=True)
                
    # Create masks
    seq_mask = np.ones(n)
    pair_mask = np.ones((n, n))

    features = {
        'seq': seq,
        'z': z_matrix,
        'dist': dist_matrix,
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
    }
    
    return features

def create_dist_matrix(coords, discretization_config='b12'):
    """Helper function to create distance matrix from coordinates"""
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


