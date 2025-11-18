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
protein_atoms = []
for element, atom_names in PROTEIN_ATOM_TYPES.items():
    protein_atoms.extend(atom_names)

# Ring center virtual atoms (by ring size) - shared by both protein and ligand
RING_CENTERS = ['RING_3', 'RING_4', 'RING_5', 'RING_6', 'RING_X']  # 3, 4, 5, 6, and larger rings

# Protein atom types: CA + all non-C atoms + ring center virtual atoms
protein_atoms = protein_atoms + RING_CENTERS

# Ligand elements: common elements (prefixed with _ to avoid conflict with protein atoms)
ligand_elements = ['_C', '_O', '_N', '_F', '_S', '_P', '_Cl', '_Br', '_I',
                   '_ZN', '_MG', '_FE', '_CU', '_MN', '_CO', '_NI', '_MO', '_W', '_SE']

# Allowed atom types for node features (only protein atoms + X for ligand)
ALLOWED_ATOM_TYPES = ['X'] + protein_atoms

# Create mappings
ATOM2IDX = {atom: idx for idx, atom in enumerate(ALLOWED_ATOM_TYPES)}

# Ligand atom types mapping (for ground truth labels)
LIGAND_ATOM_TYPES = ['X'] + ligand_elements
LIGAND_ATOM2IDX = {atom: idx for idx, atom in enumerate(LIGAND_ATOM_TYPES)}

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
    
    # Prepare full coordinates for distance calculation
    full_coords = np.concatenate([receptor_full_coords, ligand_full_coords], axis=0)
    
    # Check receptor-ligand proximity and get interaction mask
    receptor_interaction = check_receptor_ligand_interaction(
        receptor_full_coords=receptor_full_coords,
        ligand_full_coords=ligand_full_coords,
        receptor_atoms=receptor_atoms,
        distance_threshold=5.0
    )
    
    # Return None if no interaction (no receptor atom within 5A of ligand)
    if receptor_interaction.sum() == 0:
        return None
    
    features = init_ligand_coords_features(
        receptor_atoms=receptor_atoms,
        ligand_size=ligand_size,
        full_coords=full_coords
    )
    
    features['x'] = full_coords
    
    # Add ligand atom types as class indices (same size as full_coords)
    n_receptor = len(receptor_full_coords)
    n_total = len(full_coords)
    ligand_atom_types_full = np.zeros(n_total, dtype=np.int64)  # default to 0 (X/unknown)
    
    # Fill in ligand atom types
    for idx, atom_symbol in enumerate(ligand_full_atoms):
        if atom_symbol in LIGAND_ATOM2IDX:
            ligand_atom_types_full[n_receptor + idx] = LIGAND_ATOM2IDX[atom_symbol]
        else:
            ligand_atom_types_full[n_receptor + idx] = LIGAND_ATOM2IDX['X']  # fallback to unknown
    
    features['ligand_atoms'] = ligand_atom_types_full
    
    # Add receptor interaction mask (extend to full size with ligand part as 0)
    receptor_interaction_full = np.zeros(n_total, dtype=np.int64)
    receptor_interaction_full[:n_receptor] = receptor_interaction
    features['receptor_interaction'] = receptor_interaction_full

    return features

def init_ligand_coords_features(receptor_atoms, ligand_size, full_coords):
    """Initialize features from receptor/ligand information.
    
    Args:
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
            e.g., [(['CA', 'RING_6'], ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'RING_6']), ...]
        ligand_size: Number of ligand atoms
        full_coords: Array of all atom coordinates (receptor + ligand)
    
    Returns:
        dict with features including 'h', 'z', 'receptor_mask'
    """
    # Build full receptor atom list
    full_receptor_atoms = []
    for reduced_atoms, full_atoms in receptor_atoms:
        full_receptor_atoms.extend(full_atoms)
    
    n_receptor = len(full_receptor_atoms)
    n_total = n_receptor + ligand_size
    
    node_attr = build_nodes(full_receptor_atoms, ligand_size)
    
    # Build pair features (dense matrix)
    z_matrix = build_pair_features(
        receptor_atoms=receptor_atoms,
        ligand_size=ligand_size,
        full_coords=full_coords
    )
    
    # Create receptor mask: 1 for receptor atoms, 0 for ligand atoms
    receptor_mask = np.zeros(n_total, dtype=np.int64)
    receptor_mask[:n_receptor] = 1
    
    # Create anchor mask: 1 for all CA atoms, 0 for others
    anchor_mask = np.zeros(n_total, dtype=np.int64)
    for i, atom_name in enumerate(full_receptor_atoms):
        if atom_name == 'CA':
            anchor_mask[i] = 1
    
    # For unbatched data (single sample), seq_mask and pair_mask are all 1s (no padding)
    seq_mask = np.ones(n_total, dtype=np.int64)
    pair_mask = np.ones((n_total, n_total), dtype=np.int64)
    
    features = {
        'h': node_attr,
        'z': z_matrix,
        'receptor_mask': receptor_mask,
        'anchor_mask': anchor_mask,
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
    }
    
    return features

def build_pair_features(receptor_atoms, ligand_size, full_coords):
    """Build dense pair feature matrix.
    
    Args:
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
        ligand_size: Number of ligand atoms
        full_coords: Array of all atom coordinates (receptor + ligand) - unused, kept for API compatibility
    
    Returns:
        numpy array of shape (n, n, 1) with [is_same_residue] feature
    """
    # Build residue indices for each receptor atom
    res_indices = []
    for res_idx, (reduced_atoms, full_atoms) in enumerate(receptor_atoms):
        res_indices.extend([res_idx] * len(full_atoms))
    
    n_receptor = len(res_indices)
    n = n_receptor + ligand_size
    
    # Extend residue indices for ligand (use -1 for ligand atoms)
    res_indices = np.array(res_indices + [-1] * ligand_size, dtype=np.int64)
    
    # Build dense pair feature matrix
    z = np.zeros((n, n, 1), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            res_i, res_j = res_indices[i], res_indices[j]
            is_same_residue = 1.0 if (res_i == res_j and res_i != -1) else 0.0
            z[i, j, 0] = is_same_residue
    
    return z

def build_nodes(receptor_atoms, ligand_size):
    """Build node attributes from receptor atoms and ligand atoms.
    
    Args:
        receptor_atoms: List of receptor atom names (e.g., 'CA', 'RING_3', etc.)
        ligand_size: Total number of ligand atoms
    
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
    
    # Build ligand node attributes - all atom types are unknown
    for _ in range(ligand_size):
        mol_type = [0, 0, 1]  # ligand
        atom_one_hot = get_atom_one_hot('X')  # unknown atom type
        node_attrs.append(np.concatenate([mol_type, atom_one_hot]))
    
    return np.array(node_attrs)

def get_ligand_atoms_and_coords(mol):
    """Get ligand atoms and coordinates from molecule.
    
    Returns:
        tuple: (reduced_atoms, reduced_coords, full_atoms, full_coords)
            reduced_atoms: List of reduced atom types (non-C atoms only, no ring centers)
            reduced_coords: Coordinates of reduced atoms
            full_atoms: List of all atom types (all atoms including C atoms, no ring centers)
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
    
    # No longer add ring centers for ligand
    full_atoms = reduced_atoms + c_atoms
    full_coords = reduced_coords + c_coords
    
    return reduced_atoms, np.array(reduced_coords), full_atoms, np.array(full_coords)

def check_receptor_ligand_interaction(receptor_full_coords, ligand_full_coords, receptor_atoms, distance_threshold=5.0):
    """Check which receptor atoms are within threshold distance of any ligand atom.
    Only considers receptor non-C atoms and RING atoms.
    
    Args:
        receptor_full_coords: Array of full receptor atom coordinates
        ligand_full_coords: Array of all ligand atom coordinates
        receptor_atoms: List of tuples, each tuple is ([reduced_atoms...], [full_atoms...])
        distance_threshold: Distance threshold in Angstrom
    
    Returns:
        numpy array of shape (n_receptor,) with:
            - 1 for receptor non-C/RING atoms within threshold distance of any ligand atom
            - 0 for receptor C atoms or atoms not within threshold
    """
    # Build full receptor atom list with names
    full_receptor_atom_names = []
    for reduced_atoms, full_atoms in receptor_atoms:
        full_receptor_atom_names.extend(full_atoms)
    
    n_receptor = len(full_receptor_atom_names)
    
    # Initialize mask (only for receptor atoms)
    proximity_mask = np.zeros(n_receptor, dtype=np.int64)
    
    # For each receptor atom
    for i, atom_name in enumerate(full_receptor_atom_names):
        # Only consider non-C atoms or RING atoms
        if atom_name.startswith('RING_') or (atom_name[0] != 'C' and atom_name != 'CA'):
            # Check distance to all ligand atoms
            rec_coord = receptor_full_coords[i]
            for lig_coord in ligand_full_coords:
                distance = np.linalg.norm(rec_coord - lig_coord)
                if distance <= distance_threshold:
                    proximity_mask[i] = 1
                    break
    
    return proximity_mask

class E2EDataset(Dataset):
    """Dataset for predicting continuous features from distance matrix."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='e2efix')
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
        
        return self.features_to_tensors(features)

    @staticmethod
    def features_to_tensors(features):
        """Convert features dict to tensor dict for Toformer.
        
        Args:
            features: dict with required keys 'x', 'h', 'z', 'receptor_mask', 'anchor_mask', 'seq_mask', 'pair_mask'
                     and optional keys 'ligand_atoms', 'receptor_interaction'
        
        Returns:
            Dict with tensor values
        """
        result = {
            'x': torch.tensor(features['x'], dtype=const.TORCH_FLOAT),
            'h': torch.tensor(features['h'], dtype=const.TORCH_FLOAT),
            'z': torch.tensor(features['z'], dtype=const.TORCH_FLOAT),
            'receptor_mask': torch.tensor(features['receptor_mask'], dtype=torch.long),
            'anchor_mask': torch.tensor(features['anchor_mask'], dtype=torch.long),
            'seq_mask': torch.tensor(features['seq_mask'], dtype=torch.long),
            'pair_mask': torch.tensor(features['pair_mask'], dtype=torch.long),
        }
        
        # Optional fields (only needed for training)
        if 'ligand_atoms' in features:
            result['ligand_atoms'] = torch.tensor(features['ligand_atoms'], dtype=torch.long)
        if 'receptor_interaction' in features:
            result['receptor_interaction'] = torch.tensor(features['receptor_interaction'], dtype=torch.long)
        
        return result
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for ContDataset producing padded tensors.
        
        Since Toformer requires dense pair matrices, we pad all samples to the same size.
        
        Args:
            batch_data: List of dicts, each with keys 'x', 'h', 'z', 'receptor_mask', 'anchor_mask', 'ligand_atoms', 'receptor_interaction'
        
        Returns:
            Dict with batched and padded tensors:
                - x: [batch_size, max_n, 3] - coordinates
                - h: [batch_size, max_n, h_dim] - node features
                - z: [batch_size, max_n, max_n, z_dim] - pair features
                - receptor_mask: [batch_size, max_n] - mask for receptor atoms (1=receptor, 0=ligand/padding)
                - anchor_mask: [batch_size, max_n] - mask for anchor atoms (1=CA atoms, 0=others)
                - seq_mask: [batch_size, max_n] - mask for non-padding nodes (1=valid, 0=padding)
                - pair_mask: [batch_size, max_n, max_n] - mask for non-padding pairs (1=valid, 0=padding)
                - ligand_atoms: [batch_size, max_n] - ligand atom type class indices (0 for receptor/padding)
                - receptor_interaction: [batch_size, max_n] - receptor atoms interacting with ligand
        """
        max_n = max(data['x'].size(0) for data in batch_data)
        batch_size = len(batch_data)
        h_dim = batch_data[0]['h'].size(-1)
        z_dim = batch_data[0]['z'].size(-1)
        
        # Initialize padded tensors
        x_batch = torch.zeros(batch_size, max_n, 3, dtype=torch.float32)
        h_batch = torch.zeros(batch_size, max_n, h_dim, dtype=torch.float32)
        z_batch = torch.zeros(batch_size, max_n, max_n, z_dim, dtype=torch.float32)
        receptor_mask_batch = torch.zeros(batch_size, max_n, dtype=torch.long)  # 1 for receptor, 0 for ligand/padding
        anchor_mask_batch = torch.zeros(batch_size, max_n, dtype=torch.long)  # 1 for CA atoms, 0 for others
        seq_mask_batch = torch.zeros(batch_size, max_n, dtype=torch.long)  # 1 for valid, 0 for padding
        pair_mask_batch = torch.zeros(batch_size, max_n, max_n, dtype=torch.long)  # 1 for valid pairs, 0 for padding
        ligand_atoms_batch = torch.zeros(batch_size, max_n, dtype=torch.long)  # ligand atom classes (same size as nodes)
        receptor_interaction_batch = torch.zeros(batch_size, max_n, dtype=torch.long)  # receptor-ligand interaction
        
        for i, data in enumerate(batch_data):
            n = data['x'].size(0)
            x_batch[i, :n] = data['x']
            h_batch[i, :n] = data['h']
            z_batch[i, :n, :n] = data['z']
            receptor_mask_batch[i, :n] = data['receptor_mask']
            anchor_mask_batch[i, :n] = data['anchor_mask']
            seq_mask_batch[i, :n] = 1  # Mark valid nodes
            pair_mask_batch[i, :n, :n] = 1  # Mark valid pairs
            ligand_atoms_batch[i, :n] = data['ligand_atoms']  # Already same size as nodes
            receptor_interaction_batch[i, :n] = data['receptor_interaction']
        
        return {
            'x': x_batch,
            'h': h_batch,
            'z': z_batch,
            'receptor_mask': receptor_mask_batch,
            'anchor_mask': anchor_mask_batch,
            'seq_mask': seq_mask_batch,
            'pair_mask': pair_mask_batch,
            'ligand_atoms': ligand_atoms_batch,
            'receptor_interaction': receptor_interaction_batch,
        }
