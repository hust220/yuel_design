import os
import numpy as np
import torch

from torch.utils.data import Dataset
from rdkit import Chem
from src import const
from src.pdb_utils import Structure
from src.cache import FileCache
from src.db_utils import db_connection
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

def create_ligand_coords_features(pocket_pdb, ligand_mol, item_id=None):
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
    receptor_atoms = pocket_info['atoms']
    receptor_reduced_coords = pocket_info['reduced_coords']
    receptor_full_coords = pocket_info['full_coords']
    
    # Create interaction index pairs between reduced atoms
    # int_index contains (receptor_full_idx, ligand_full_idx) pairs
    int_index = create_interaction_index(
        receptor_reduced_coords=receptor_reduced_coords,
        ligand_reduced_coords=ligand_reduced_coords,
        receptor_atoms=receptor_atoms,
    )
    
    # Return None if int_index is empty
    if len(int_index) == 0:
        return None
    
    # Prepare full coordinates for distance calculation
    full_coords = np.concatenate([receptor_full_coords, ligand_full_coords], axis=0)
    
    features = init_ligand_coords_features(
        int_index=int_index,
        receptor_atoms=receptor_atoms,
        ligand_fixed_atoms=ligand_reduced_atoms,
        ligand_size=ligand_size
    )
    
    features['x'] = full_coords.astype(np.float32)

    return features

def init_ligand_coords_features(int_index, receptor_atoms, ligand_fixed_atoms, ligand_size):
    """Initialize features from interaction index and receptor/ligand information.
    
    Args:
        int_index: List of tuples, each tuple is (receptor_full_idx, ligand_full_idx)
            representing interactions between reduced receptor and ligand atoms
            but with indices in full atoms
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
        ligand_fixed_atoms: List of ligand atom types for reduced ligand atoms
        ligand_size: Number of ligand atoms
    
    Returns:
        dict with features including 'h', 'edge_index', 'edge_attr', 'mask'
    """
    # Build full receptor atom list
    full_receptor_atoms = []
    for reduced_atoms, full_atoms in receptor_atoms:
        full_receptor_atoms.extend(full_atoms)
    
    n_receptor = len(full_receptor_atoms)
    n_total = n_receptor + ligand_size
    
    node_attr = build_nodes(full_receptor_atoms, ligand_fixed_atoms, ligand_size)
    
    # Build pair features
    z_matrix = build_pair_features(
        int_index=int_index,
        receptor_atoms=receptor_atoms,
        ligand_size=ligand_size
    )
    
    # Create masks
    mask = np.zeros(n_total, dtype=np.float32)
    mask[:n_receptor] = 1.0
    seq_mask = np.ones(n_total, dtype=np.float32)
    pair_mask = np.ones((n_total, n_total), dtype=np.float32)
    
    features = {
        'seq': node_attr.astype(np.float32),
        'z': z_matrix.astype(np.float32),
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
        'mask': mask,
    }
    
    return features

def build_pair_features(int_index, receptor_atoms, ligand_size):
    """Build pair features tensor combining residue and interaction information."""
    # Build residue indices for each receptor atom
    res_indices = []
    for res_idx, (reduced_atoms, full_atoms) in enumerate(receptor_atoms):
        res_indices.extend([res_idx] * len(full_atoms))
    
    n_receptor = len(res_indices)
    n = n_receptor + ligand_size
    
    # Extend residue indices for ligand (use -1 for ligand atoms)
    res_indices = np.array(res_indices + [-1] * ligand_size, dtype=np.int64)
    
    # Convert int_index to a set for fast lookup (both directions)
    int_index_set = set(int_index)
    int_index_set.update((lig_idx, rec_idx) for rec_idx, lig_idx in int_index)
    
    z = np.zeros((n, n, 2), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            res_i, res_j = res_indices[i], res_indices[j]
            is_same_residue = 1.0 if (res_i == res_j and res_i != -1) else 0.0
            is_interaction = 1.0 if (i, j) in int_index_set else 0.0
            z[i, j, 0] = is_same_residue
            z[i, j, 1] = is_interaction
    
    return z

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
    c_atoms = []
    c_coords = []
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
       
        # Add to reduced atoms if non-C
        if atom.GetSymbol() != 'C':
            reduced_atoms.append(atom_symbol)
            reduced_coords.append(coord)
        else:
            c_atoms.append(atom_symbol)
            c_coords.append(coord)
    
    # Detect and add ring centers
    ring_centers = detect_ligand_rings(mol)
    for ring_type, ring_center in ring_centers:
        reduced_atoms.append(ring_type)
        reduced_coords.append(ring_center)
    
    full_atoms = reduced_atoms + c_atoms
    full_coords = reduced_coords + c_coords
    
    return reduced_atoms, np.array(reduced_coords), full_atoms, np.array(full_coords)

def create_interaction_index(receptor_reduced_coords, ligand_reduced_coords, receptor_atoms, interaction_threshold=5):
    """Create interaction index pairs between reduced receptor and ligand atoms.
    
    Args:
        receptor_reduced_coords: Array of reduced receptor atom coordinates (CA + non-C + ring centers)
        ligand_reduced_coords: Array of reduced ligand atom coordinates (non-C + ring centers)
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
        interaction_threshold: Distance threshold for interaction (in Angstrom)
    
    Returns:
        List of tuples: [(receptor_full_idx, ligand_full_idx), ...]
            where indices are in full atoms (receptor full atoms + ligand full atoms)
    """
    # Build mapping from reduced receptor index to full receptor index
    reduced_to_full_receptor = {}
    reduced_idx = 0
    full_idx_offset = 0
    
    for reduced_atoms, full_atoms in receptor_atoms:
        for atom_name in reduced_atoms:
            if atom_name in full_atoms:
                full_atom_idx = full_atoms.index(atom_name)
                reduced_to_full_receptor[reduced_idx] = full_idx_offset + full_atom_idx
            reduced_idx += 1
        
        full_idx_offset += len(full_atoms)
    
    n_receptor_full = full_idx_offset
    
    # Ligand reduced atoms are already at the front in full_atoms
    # So ligand reduced index i maps to ligand full index i
    # Ligand full atoms: [reduced_atoms (n_reduced), C_atoms (n_c)]
    # So reduced index i -> full index i
    
    # Calculate distances between reduced atoms
    int_index = []
    for i, rec_coord in enumerate(receptor_reduced_coords):
        for j, lig_coord in enumerate(ligand_reduced_coords):
            distance = np.linalg.norm(rec_coord - lig_coord)
            if distance <= interaction_threshold:
                # Map to full indices
                rec_full_idx = reduced_to_full_receptor[i]
                lig_full_idx = n_receptor_full + j  # ligand reduced atoms are at the front
                int_index.append((rec_full_idx, lig_full_idx))
    
    return int_index

class LigandDataset(Dataset):
    """Dataset for predicting continuous features from distance matrix."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='ligand')
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
                    features = create_ligand_coords_features(pocket_pdb, ligand_mol, item_id=item_id)
                    if features is None:
                        # Skip this sample and try a random one
                        import random
                        random_item = random.randint(0, len(self.ids) - 1)
                        return self.__getitem__(random_item)
                    self.cache.set(item_id, (features, ligand_name))
        else:
            features, ligand_name = cached_data
        
        return self._to_torch(features)

    @staticmethod
    def features_to_tensors(features):
        """Convert features dict to tensor representation."""
        return {
            'x': torch.tensor(features['x'], dtype=const.TORCH_FLOAT),
            'seq': torch.tensor(features['seq'], dtype=const.TORCH_FLOAT),
            'z': torch.tensor(features['z'], dtype=const.TORCH_FLOAT),
            'seq_mask': torch.tensor(features['seq_mask'], dtype=const.TORCH_FLOAT),
            'pair_mask': torch.tensor(features['pair_mask'], dtype=const.TORCH_FLOAT),
            'mask': torch.tensor(features['mask'], dtype=const.TORCH_FLOAT),
        }
    
    def _to_torch(self, features):
        return self.features_to_tensors(features)
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for LigandDataset producing padded tensors."""
        max_n = max(sample['seq'].size(0) for sample in batch_data)
        batch_size = len(batch_data)
        seq_dim = batch_data[0]['seq'].size(-1)
        z_dim = batch_data[0]['z'].size(-1)
        
        seq = torch.zeros(batch_size, max_n, seq_dim, dtype=torch.float32)
        z = torch.zeros(batch_size, max_n, max_n, z_dim, dtype=torch.float32)
        seq_mask = torch.zeros(batch_size, max_n, dtype=torch.float32)
        pair_mask = torch.zeros(batch_size, max_n, max_n, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_n, dtype=torch.float32)
        x = torch.zeros(batch_size, max_n, 3, dtype=torch.float32)
        
        for i, sample in enumerate(batch_data):
            n = sample['seq'].size(0)
            seq[i, :n] = sample['seq']
            z[i, :n, :n] = sample['z']
            seq_mask[i, :n] = sample['seq_mask']
            pair_mask[i, :n, :n] = sample['pair_mask']
            mask[i, :n] = sample['mask']
            x[i, :n] = sample['x']
        
        return {
            'seq': seq,
            'z': z,
            'seq_mask': seq_mask,
            'pair_mask': pair_mask,
            'mask': mask,
            'x': x,
        }
