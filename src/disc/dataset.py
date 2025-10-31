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
from src.const import ALLOWED_PDB_ATOM_TYPES, PDB_ATOM2IDX, ELEMENT2IDX

# Define unified atom types for coarse-grained representation
# Protein atoms: CA + 20 standard amino acid side chains + X_SC for unknown
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
protein_atoms = ['CA'] + [f'{aa}_SC' for aa in STANDARD_AMINO_ACIDS]  # Includes GLY_SC

# Ligand elements: common elements
ligand_elements = ['C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I',
                   'ZN', 'MG', 'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'W', 'SE']

# Ligand atom types: same as ligand elements
LIGAND_ATOM_TYPES = ligand_elements.copy()

# Ligand bond types: no_bond, single, double, triple, aromatic
LIGAND_BOND_TYPES = ['NO_BOND', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

# Create mappings
LIGAND_ATOM_TYPE2IDX = {atom_type: idx for idx, atom_type in enumerate(LIGAND_ATOM_TYPES)}
LIGAND_BOND_TYPE2IDX = {bond_type: idx for idx, bond_type in enumerate(LIGAND_BOND_TYPES)}

def get_ligand_atom_type(atom_symbol):
    """Get ligand atom type index from atom symbol."""
    if atom_symbol in LIGAND_ATOM_TYPE2IDX:
        return LIGAND_ATOM_TYPE2IDX[atom_symbol]
    return LIGAND_ATOM_TYPE2IDX[LIGAND_ATOM_TYPES[0]]

def get_ligand_bond_type(bond_type):
    """Get ligand bond type index from RDKit bond type."""
    if bond_type == Chem.BondType.SINGLE:
        return LIGAND_BOND_TYPE2IDX['SINGLE']
    elif bond_type == Chem.BondType.DOUBLE:
        return LIGAND_BOND_TYPE2IDX['DOUBLE']
    elif bond_type == Chem.BondType.TRIPLE:
        return LIGAND_BOND_TYPE2IDX['TRIPLE']
    elif bond_type == Chem.BondType.AROMATIC:
        return LIGAND_BOND_TYPE2IDX['AROMATIC']
    return LIGAND_BOND_TYPE2IDX['SINGLE']

# Combined atom types
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
    Supports protein atoms (CA, ALA_SC, ARG_SC, etc.) and ligand elements.
    """
    if atom_name in ATOM2IDX:
        return get_one_hot(atom_name, ATOM2IDX)
    else:
        return get_one_hot('X', ATOM2IDX)

def parse_pocket(structure):
    """Parse protein structure using coarse-grained representation.
    Each residue is simplified to two atoms: CA (backbone) and side chain center.
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
                    # CA node
                    coords.append(ca_coord)
                    atom_names.append('CA')
                    res_ids.append(ires)
                    res_names.append(res_name)
                    codes.append(get_atom_one_hot('CA'))
                    mol_types.append([1, 0, 0])

                    # side chain center excluding backbone atoms
                    sc_coords = [c for name, c in residue_atoms if name not in ['CA', 'N', 'C', 'O']]
                    if sc_coords:
                        sc_center = np.mean(sc_coords, axis=0)
                    else:
                        # gly: approximate side chain along bisector of N-CA-C
                        n_coord = next((c for name, c in residue_atoms if name == 'N'), None)
                        c_coord = next((c for name, c in residue_atoms if name == 'C'), None)
                        if n_coord is not None and c_coord is not None:
                            n_vec = n_coord - ca_coord
                            c_vec = c_coord - ca_coord
                            vec = (n_vec + c_vec)
                            norm = np.linalg.norm(vec)
                            sc_center = ca_coord + (vec / norm * 1.5 if norm > 0 else np.array([1.5, 0.0, 0.0]))
                        else:
                            sc_center = ca_coord + np.array([1.5, 0.0, 0.0])
                    coords.append(sc_center)
                    atom_name = f'{res_name}_SC'
                    atom_names.append(atom_name)
                    res_ids.append(ires)
                    res_names.append(res_name)
                    codes.append(get_atom_one_hot(atom_name))
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

def get_ligand_atoms_and_coords(mol):
    """Get ligand atoms (as atom type indices), coordinates, and bond matrix from molecule."""
    ligand_atoms = []
    ligand_coords = []
    atom_idx_map = {}
    
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx_map[atom.GetIdx()] = len(ligand_atoms)
        ligand_atoms.append(get_ligand_atom_type(atom.GetSymbol()))
        ligand_coords.append(conf.GetAtomPosition(atom.GetIdx()))
    
    ligand_size = len(ligand_atoms)
    bond_matrix = np.full((ligand_size, ligand_size), LIGAND_BOND_TYPE2IDX['NO_BOND'], dtype=np.int64)
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(begin_idx).GetSymbol() == 'H' or mol.GetAtomWithIdx(end_idx).GetSymbol() == 'H':
            continue
        if begin_idx in atom_idx_map and end_idx in atom_idx_map:
            i, j = atom_idx_map[begin_idx], atom_idx_map[end_idx]
            bond_type_idx = get_ligand_bond_type(bond.GetBondType())
            bond_matrix[i, j] = bond_matrix[j, i] = bond_type_idx
    
    return ligand_atoms, ligand_coords, bond_matrix

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

    ligand_atoms, ligand_coords, ligand_bond_matrix = get_ligand_atoms_and_coords(mol)
    ligand_coords = np.array(ligand_coords)
    ligand_size = len(ligand_atoms)
    
    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        return None

    dist_matrix = create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12')

    features = init_dist_features(pocket_info, ligand_size)

    features['dist'] = dist_matrix # [N, N] - distance class indices

    protein_size = len(pocket_info['coords'])
    features['ligand_atoms'] = torch.zeros((protein_size + ligand_size,), dtype=torch.int64)
    features['ligand_atoms'][protein_size:] = torch.tensor(ligand_atoms, dtype=torch.int64)

    features['ligand_bonds'] = torch.zeros((protein_size + ligand_size, protein_size + ligand_size), dtype=torch.int64)
    features['ligand_bonds'][protein_size:, protein_size:] = torch.tensor(ligand_bond_matrix, dtype=torch.int64)

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
    pair_ligand_mask = np.zeros((n, n))
    for i in range(protein_size, n):
        for j in range(protein_size, n):
            pair_ligand_mask[i, j] = 1
            pair_ligand_mask[j, i] = 1
        seq_ligand_mask[i] = 1

    features = {
        'seq': seq, # [is_CA, is_SC, is_Ligand, one_hot]
        'z': z_matrix, # [same_residue]
        'bb_dist': bb_dist, # [b12(+1)_distance_class]
        'seq_mask': seq_mask,
        'pair_mask': pair_mask,
        'seq_ligand_mask': seq_ligand_mask,
        'pair_ligand_mask': pair_ligand_mask,
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
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='disc')
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
            'pair_ligand_mask': torch.tensor(features['pair_ligand_mask'], dtype=const.TORCH_FLOAT),
            'ligand_atoms': torch.tensor(features['ligand_atoms'], dtype=const.TORCH_INT),
            'ligand_bonds': torch.tensor(features['ligand_bonds'], dtype=const.TORCH_INT),
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
        ligand_bonds = torch.zeros(bsz, max_n, max_n, dtype=torch.int64)
        seq_ligand_mask = torch.zeros(bsz, max_n, dtype=torch.float32)
        pair_ligand_mask = torch.zeros(bsz, max_n, max_n, dtype=torch.float32)
        
        for i, sample in enumerate(batch):
            n = sample['seq'].size(0)
            seq[i, :n] = sample['seq']
            z[i, :n, :n] = sample['z']
            dist[i, :n, :n] = sample['dist']
            bb_dist[i, :n, :n] = sample['bb_dist']
            seq_mask[i, :n] = sample['seq_mask']
            pair_mask[i, :n, :n] = sample['pair_mask']
            ligand_atoms[i, :n] = sample['ligand_atoms']
            ligand_bonds[i, :n, :n] = sample['ligand_bonds']
            seq_ligand_mask[i, :n] = sample['seq_ligand_mask']
            pair_ligand_mask[i, :n, :n] = sample['pair_ligand_mask']

        return {
            # inputs
            'seq': seq,
            'z': z,
            'bb_dist': bb_dist,
            # masks
            'seq_mask': seq_mask,
            'pair_mask': pair_mask,
            'seq_ligand_mask': seq_ligand_mask,
            'pair_ligand_mask': pair_ligand_mask,
            # oututs
            'dist': dist,
            'ligand_atoms': ligand_atoms,
            'ligand_bonds': ligand_bonds,
        }

