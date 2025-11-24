import os
import numpy as np
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset
from src import const
from src.pdb_utils import Structure
from src.cache import FileCache
from src.db_utils import db_connection
from src.distance_discretization import discretize_distance_numpy
from src.const import ALLOWED_PDB_ATOM_TYPES, PDB_ATOM2IDX, ELEMENT2IDX, PROTEIN_ATOM_TYPES

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

# Ligand elements: common elements (non-C atoms only, prefixed with _ to avoid conflict with protein atoms)
ligand_elements = ['_O', '_N', '_F', '_S', '_P', '_Cl', '_Br', '_I',
                   '_ZN', '_MG', '_FE', '_CU', '_MN', '_CO', '_NI', '_MO', '_W', '_SE']

# Ligand atom types: elements + ring centers
LIGAND_ATOM_TYPES = ligand_elements + RING_CENTERS

# Ligand bond types: no_bond, bonded (all bond types merged)
LIGAND_BOND_TYPES = ['NO_BOND', 'BONDED']

# Create mappings
LIGAND_ATOM_TYPE2IDX = {atom_type: idx for idx, atom_type in enumerate(LIGAND_ATOM_TYPES)}
LIGAND_BOND_TYPE2IDX = {bond_type: idx for idx, bond_type in enumerate(LIGAND_BOND_TYPES)}

def get_ligand_atom_type(atom_symbol):
    """Get ligand atom type index from atom symbol.
    Converts RDKit atom symbol (e.g., 'O', 'N') to ligand format with _ prefix (e.g., '_O', '_N').
    """
    # Convert to ligand format with _ prefix
    ligand_symbol = '_' + atom_symbol if not atom_symbol.startswith('_') else atom_symbol
    if ligand_symbol in LIGAND_ATOM_TYPE2IDX:
        return LIGAND_ATOM_TYPE2IDX[ligand_symbol]
    return LIGAND_ATOM_TYPE2IDX[LIGAND_ATOM_TYPES[0]]

def get_ligand_bond_type(bond_type):
    """Get ligand bond type index from RDKit bond type.
    All bond types (SINGLE, DOUBLE, TRIPLE, AROMATIC) are merged into BONDED.
    """
    if bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]:
        return LIGAND_BOND_TYPE2IDX['BONDED']
    return LIGAND_BOND_TYPE2IDX['NO_BOND']

# Combined atom types: X (unknown) + protein atoms + ligand elements
# Note: RING_CENTERS are already in protein_atoms, so we only add ligand_elements (not LIGAND_ATOM_TYPES)
# to avoid duplicating ring centers
ALLOWED_ATOM_TYPES = ['X'] + protein_atoms + ligand_elements

# Create mapping
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

def detect_protein_ring(res_name, residue_atoms):
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
        
        for name, coord in residue_atoms:
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
    """Parse protein structure: keep all CA and non-C atoms, plus ring centers as virtual atoms.
    Only supports protein residues, does not handle RNA/DNA.
    """
    coords, mol_types, codes, atom_names, res_ids, res_names = [], [], [], [], [], []
    ires = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_atoms = []
                ca_coord = None
                res_name = residue.res_name
                
                for atom in residue:
                    if atom.atom_name[0] == 'H':
                        continue
                    if atom.atom_name == 'CA':
                        ca_coord = atom.get_coord()
                    residue_atoms.append((atom.atom_name, atom.get_coord()))
                
                if ca_coord is not None:
                    # Always add CA node
                    coords.append(ca_coord)
                    atom_names.append('CA')
                    res_ids.append(ires)
                    res_names.append(res_name)
                    codes.append(get_atom_one_hot('CA'))
                    mol_types.append([1, 0, 0])
                    
                    # Add all non-C atoms (excluding CA and C atoms, but including backbone N and O)
                    for name, coord in residue_atoms:
                        if name and name[0] != 'C':  # Non-C atom (includes backbone N, O, OXT and side chain non-C atoms)
                            coords.append(coord)
                            atom_names.append(name)
                            res_ids.append(ires)
                            res_names.append(res_name)
                            codes.append(get_atom_one_hot(name))
                            mol_types.append([0, 1, 0])
                    
                    # Detect and add ring centers as virtual atoms
                    ring_centers = detect_protein_ring(res_name, residue_atoms)
                    for ring_type, ring_center in ring_centers:
                        coords.append(ring_center)
                        atom_names.append(ring_type)  # Use ring type (RING_3, RING_4, etc.)
                        res_ids.append(ires)
                        res_names.append(res_name)
                        codes.append(get_atom_one_hot(ring_type))  # Use ring type for encoding
                        mol_types.append([0, 1, 0])
                    
                    ires += 1
    
    return {
        'coords': coords,
        'mol_types': mol_types,
        'codes': codes,
        'atom_names': atom_names,
        'res_ids': res_ids,
        'residue_names': res_names,
    }

def create_z_matrix(res_ids, receptor_coords, ligand_size):
    n = len(receptor_coords) + ligand_size
    res_ids = res_ids + ([res_ids[-1] + 1] * ligand_size)
    z = np.zeros((n, n, 1), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                ri, rj = res_ids[i], res_ids[j]
                is_same_residue = 1 if ri == rj else 0
                z[i, j, 0] = is_same_residue
    return z

def detect_ligand_rings(mol):
    """Detect rings in ligand molecule and return ring centers with ring types.
    Returns list of (ring_type, ring_center_coord), where ring_type is based on ring size.
    """
    ring_centers = []
    ring_info = mol.GetRingInfo()
    
    if ring_info is None:
        return ring_centers
    
    conf = mol.GetConformer()
    
    # Get all rings
    rings = ring_info.AtomRings()
    for ring_atom_indices in rings:
        ring_size = len(ring_atom_indices)
        if ring_size >= 3:
            # Calculate ring center
            ring_coords = []
            for atom_idx in ring_atom_indices:
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() != 'H':
                    coord = conf.GetAtomPosition(atom_idx)
                    # Convert RDKit Point3D to numpy array
                    ring_coords.append(np.array([coord.x, coord.y, coord.z]))
            
            if ring_coords:
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
                    ring_type = 'RING_X'  # Larger rings (7+)
                ring_centers.append((ring_type, ring_center))
    
    return ring_centers


def get_ligand_atoms_and_coords(mol, pocket_info):
    """Get ligand atoms (as atom type indices), coordinates, and bond matrix from molecule.
    
    Args:
        mol: RDKit molecule object
        pocket_info: Optional dict with pocket information containing 'coords' and 'atom_names'
                    If provided, applies filtering rules based on distance to protein atoms.
    
    Returns:
        tuple: (ligand_atoms, ligand_coords, bond_matrix)
    """
    conf = mol.GetConformer()
    
    # Extract protein non-C atoms and ring centers for filtering
    protein_coords = np.array(pocket_info['coords'])

    protein_atom_names = pocket_info['atom_names']
    protein_non_c_indices = [i for i, name in enumerate(protein_atom_names) if name[0] != 'C']
    protein_ring_centers_indices = [i for i, name in enumerate(protein_atom_names) if name.startswith('RING_')]
    
    protein_non_c_coords = protein_coords[protein_non_c_indices]
    protein_ring_centers = protein_coords[protein_ring_centers_indices]
    
    # Collect all non-C atoms from ligand
    ligand_non_c_coords, ligand_non_c_types = [], []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx = atom.GetIdx()
        coord = conf.GetAtomPosition(atom_idx)
        # Convert RDKit Point3D to numpy array
        coord = np.array([coord.x, coord.y, coord.z])
        symbol = atom.GetSymbol()
        
        if symbol != 'C':
            ligand_non_c_coords.append(coord)
            ligand_non_c_types.append(get_ligand_atom_type(symbol))
    
    # Filter non-C atoms: keep if distance to any protein non-C atom < 3.5
    filtered_ligand_coords, filtered_ligand_types = [], []
    for coord, atom_type in zip(ligand_non_c_coords, ligand_non_c_types):
        distances = np.linalg.norm(protein_non_c_coords - coord, axis=1)
        if np.any(distances < 3.5):
            filtered_ligand_coords.append(coord)
            filtered_ligand_types.append(atom_type)
    
    # Detect and filter ring centers
    ligand_ring_centers = detect_ligand_rings(mol)
    for ligand_ring_center_type, ligand_ring_center_coord in ligand_ring_centers:
        distances = np.linalg.norm(protein_ring_centers - ligand_ring_center_coord, axis=1)
        if np.any(distances < 4.0):
            filtered_ligand_coords.append(ligand_ring_center_coord)
            filtered_ligand_types.append(get_ligand_atom_type(ligand_ring_center_type))
    
    # Convert to numpy array with correct shape [N, 3]
    if len(filtered_ligand_coords) == 0:
        filtered_ligand_coords = np.array([]).reshape(0, 3)
    else:
        filtered_ligand_coords = np.array(filtered_ligand_coords)
            
    return filtered_ligand_types, filtered_ligand_coords

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

    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        return None

    # Get ligand atoms with filtering based on pocket_info
    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol, pocket_info)
    ligand_size = len(ligand_atoms)

    dist_matrix = create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12')

    features = init_dist_features(pocket_info, ligand_size)

    features['dist'] = dist_matrix # [N, N] - distance class indices

    protein_size = len(pocket_info['coords'])
    features['ligand_atoms'] = torch.zeros((protein_size + ligand_size,), dtype=torch.int64)
    features['ligand_atoms'][protein_size:] = torch.tensor(ligand_atoms, dtype=torch.int64)

    return features

def init_dist_features(pocket_info, ligand_size):
    receptor_coords = np.array(pocket_info['coords'])
    mol_types = pocket_info['mol_types']
    codes = pocket_info['codes']
    receptor_atoms = pocket_info['atom_names']
    receptor_residues = pocket_info['res_ids']
    protein_size = len(receptor_coords)

    mol_types.extend([[0, 0, 1]]*ligand_size)
    codes.extend([get_atom_one_hot('X')]*ligand_size)
    seq = np.concatenate([mol_types, codes], axis=1)
            
    z_matrix = create_z_matrix(receptor_residues, receptor_coords, ligand_size)

    # Build bb_dist with same shape as z_matrix: (n, n)
    # CA-CA pairs use b12 discretized distance category in [1..12], others are 0
    n = protein_size + ligand_size
    bb_dist = np.zeros((n, n), dtype=np.int64)
    for i in range(protein_size):
        if receptor_atoms[i] != 'CA':
            continue
        for j in range(protein_size):
            if i == j:
                continue
            if receptor_atoms[j] != 'CA':
                continue
            d = np.linalg.norm(receptor_coords[i] - receptor_coords[j])
            bb_dist[i, j] = discretize_distance_numpy(d, 'b12') + 1

    # Create masks
    seq_mask = np.ones(n)
    pair_mask = np.ones((n, n))
    seq_ligand_mask = np.zeros(n)
    for i in range(protein_size, n):
        seq_ligand_mask[i] = 1

    features = {
        'seq': seq, # [is_CA, is_SC, is_Ligand, one_hot]
        'z': z_matrix, # [same_residue]
        'bb_dist': bb_dist, # [b12(+1)_distance_class]
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
        'seq_ligand_mask': seq_ligand_mask,
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

class DiscDataset(Dataset):
    """Dataset for discrete features prediction."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache', no_dist_bins=12):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='disc2')
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
        return {
            'seq': torch.tensor(features['seq'], dtype=const.TORCH_FLOAT),
            'z': torch.tensor(features['z'], dtype=const.TORCH_FLOAT),
            'dist': torch.tensor(features['dist'], dtype=const.TORCH_INT),
            'bb_dist': torch.tensor(features['bb_dist'], dtype=const.TORCH_INT),
            'seq_mask': torch.tensor(features['seq_mask'], dtype=const.TORCH_FLOAT),
            'pair_mask': torch.tensor(features['pair_mask'], dtype=const.TORCH_FLOAT),
            'seq_ligand_mask': torch.tensor(features['seq_ligand_mask'], dtype=const.TORCH_FLOAT),
            'ligand_atoms': torch.tensor(features['ligand_atoms'], dtype=const.TORCH_INT),
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function for DiscDataset"""
        # batch: list of dicts with variable N
        max_n = max(sample['seq'].size(0) for sample in batch)
        
        # Determine feature sizes
        bsz = len(batch)
        feat_seq = batch[0]['seq'].size(-1)
        feat_z = batch[0]['z'].size(-1)
        
        seq = torch.zeros(bsz, max_n, feat_seq, dtype=torch.float32)
        z = torch.zeros(bsz, max_n, max_n, feat_z, dtype=torch.float32)
        dist = torch.zeros(bsz, max_n, max_n, dtype=torch.int64)
        bb_dist = torch.zeros(bsz, max_n, max_n, dtype=torch.int64)
        seq_mask = torch.zeros(bsz, max_n, dtype=torch.float32)
        pair_mask = torch.zeros(bsz, max_n, max_n, dtype=torch.float32)
        ligand_atoms = torch.zeros(bsz, max_n, dtype=torch.int64)
        seq_ligand_mask = torch.zeros(bsz, max_n, dtype=torch.float32)
        
        for i, sample in enumerate(batch):
            n = sample['seq'].size(0)
            seq[i, :n] = sample['seq']
            z[i, :n, :n] = sample['z']
            dist[i, :n, :n] = sample['dist']
            bb_dist[i, :n, :n] = sample['bb_dist']
            seq_mask[i, :n] = sample['seq_mask']
            pair_mask[i, :n, :n] = sample['pair_mask']
            ligand_atoms[i, :n] = sample['ligand_atoms']
            seq_ligand_mask[i, :n] = sample['seq_ligand_mask']

        return {
            # inputs
            'seq': seq,
            'z': z,
            'bb_dist': bb_dist,
            # masks
            'seq_mask': seq_mask,
            'pair_mask': pair_mask,
            'seq_ligand_mask': seq_ligand_mask,
            # oututs
            'dist': dist,
            'ligand_atoms': ligand_atoms,
        }

