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

# Define unified atom types for coarse-grained representation
# Protein atoms: CA + 20 standard amino acid side chains + X_SC for unknown
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
protein_cg_atoms = [f'{aa}_SC' for aa in STANDARD_AMINO_ACIDS]  # Includes GLY_SC

ALL_PROTEIN_ATOM_TYPES = {
    # Carbon atoms
    'C': [
        # Standard backbone and side chain
        'CA', 'CB', 'CG', 'CD', 'CE', 'CZ', 'CH', 'CH1', 'CH2', 'CH3',
        # Aromatic rings
        'CE1', 'CE2', 'CE3', 'CD1', 'CD2', 'CD3', 'CG1', 'CG2', 'CG3',
        # Modified carbon atoms
        'CM', 'CM1', 'CM2', 'CM3', 'CA1', 'CA2', 'CA3',
        # Nucleic acid carbons
        'C1\'', 'C2\'', 'C3\'', 'C4\'', 'C5\'', 'C2', 'C4', 'C5', 'C6', 'C8',
    ],
    
    # Nitrogen atoms
    'N': [
        # Standard backbone and side chain
        'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
        # Nucleic acid nitrogens
        'N1', 'N2', 'N3', 'N6', 'N7', 'N9',
    ],
    
    # Oxygen atoms
    'O': [
        # Standard backbone and side chain
        'O', 'OXT', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OG2', 'OH',
        # Modified oxygen atoms
        'OH1', 'OH2',
        # Nucleic acid oxygens
        'O2\'', 'O3\'', 'O4\'', 'O5\'', 'O2P', 'O3P', 'O1P', 'O2', 'O4', 'O6',
    ],
    
    # Phosphorus atoms
    'P': [
        # Phosphorylation and nucleic acid phosphates
        'P', 'OP1', 'OP2', 'OP3',
    ],
    
    # Sulfur atoms
    'S': [
        # Standard sulfur atoms
        'SG', 'SD',
    ],
}

protein_aa_atoms = [atom for atoms in ALL_PROTEIN_ATOM_TYPES.values() for atom in atoms]

# Ligand elements: common elements (prefixed with _ to avoid conflict with protein atoms)
ligand_elements = ['_C', '_O', '_N', '_F', '_S', '_P', '_Cl', '_Br', '_I',
                   '_ZN', '_MG', '_FE', '_CU', '_MN', '_CO', '_NI', '_MO', '_W', '_SE']

# Ligand bond types: no_bond, single, double, triple, aromatic
LIGAND_BOND_TYPES = ['NO_BOND', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

# Combined atom types: CG atoms + all protein atoms + ligand elements
ALLOWED_ATOM_TYPES = ['X'] + protein_cg_atoms + protein_aa_atoms + ligand_elements

# Create mappings
ATOM2IDX = {atom: idx for idx, atom in enumerate(ALLOWED_ATOM_TYPES)}
LIGAND_BOND_TYPE2IDX = {bond_type: idx for idx, bond_type in enumerate(LIGAND_BOND_TYPES)}

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

# def parse_structure_and_ligand(pocket_pdb, ligand_size, mask_ligand=True):
def parse_pocket(structure):
    """Parse protein structure including all atoms plus coarse-grained side chain centers.
    For each residue, stores all non-hydrogen atoms plus the side chain center.
    Only supports protein residues, does not handle RNA/DNA."""
    
    coords, mol_types, codes, atom_names, res_ids, res_names = [], [], [], [], [], []
    ires = 0
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Collect all non-hydrogen atoms for this residue
                residue_atoms = []
                ca_coord = None
                res_name = residue.res_name
                
                for atom in residue:
                    if atom.atom_name[0] == 'H':
                        continue
                    
                    atom_name = atom.atom_name
                    coord = atom.get_coord()
                    
                    # Store CA atom for backbone
                    if atom_name == 'CA':
                        ca_coord = coord
                    
                    residue_atoms.append((atom_name, coord))
                
                # Only process protein residues (with CA atom)
                if ca_coord is not None:
                    # Add all atoms for this residue
                    for atom_name, coord in residue_atoms:
                        coords.append(coord)
                        atom_names.append(atom_name)
                        res_ids.append(ires)
                        res_names.append(res_name)
                        codes.append(get_atom_one_hot(atom_name))
                        
                        # Determine mol_type: backbone or side chain
                        if atom_name in ['N', 'CA', 'C', 'O', 'OXT']:
                            # Backbone atoms
                            mol_types.append([1, 0, 0])
                        else:
                            # Side chain atoms
                            mol_types.append([0, 1, 0])
                    
                    # Calculate side chain center (exclude backbone atoms: CA, N, C, O)
                    side_chain_atoms = [coord for name, coord in residue_atoms 
                                        if name not in ['CA', 'N', 'C', 'O', 'OXT']]
                    
                    if side_chain_atoms:
                        # Side chain center (all amino acids except glycine)
                        side_chain_center = np.mean(side_chain_atoms, axis=0)
                        coords.append(side_chain_center)
                        atom_names.append(f'{res_name}_SC')
                        res_ids.append(ires)
                        res_names.append(res_name)
                        # Use specific side chain type if known, otherwise X
                        sc_type = f'{res_name}_SC'
                        codes.append(get_atom_one_hot(sc_type))
                        mol_types.append([0, 1, 0])  # side chain center
                    else:
                        # Glycine (GLY) has no side chain atoms
                        # Use CA's direction vector to place side chain center
                        n_coord = next((coord for name, coord in residue_atoms if name == 'N'), None)
                        c_coord = next((coord for name, coord in residue_atoms if name == 'C'), None)
                        
                        if n_coord is not None and c_coord is not None:
                            # Create a small offset along the bisector of N-CA-C angle
                            n_vec = n_coord - ca_coord
                            c_vec = c_coord - ca_coord
                            avg_vec = (n_vec + c_vec) / 2
                            norm = np.linalg.norm(avg_vec)
                            if norm > 0:
                                avg_vec = avg_vec / norm * 1.5
                                sc_coord = ca_coord + avg_vec
                            else:
                                avg_vec = np.array([1.0, 0.0, 0.0]) * 1.5
                                sc_coord = ca_coord + avg_vec
                        else:
                            # Fallback: use a small fixed offset
                            sc_coord = ca_coord + np.array([1.5, 0.0, 0.0])
                        
                        coords.append(sc_coord)
                        atom_names.append('GLY_SC')
                        res_ids.append(ires)
                        res_names.append(res_name)
                        codes.append(get_atom_one_hot('GLY_SC'))
                        mol_types.append([0, 1, 0])  # side chain center
                    
                    ires += 1
                # Skip residues without CA (not protein residues)

    return {
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
    ligand_atoms, ligand_coords, ligand_bond_matrix = get_ligand_atoms_and_coords(mol)
    ligand_coords = np.array(ligand_coords)
    ligand_size = len(ligand_coords)

    # Extract only CA, SC, and construct reduced coordinates for distance matrix
    atom_names = pocket_info['atom_names']
    ca_or_sc_coords = []
    for idx, (coord, name) in enumerate(zip(pocket_info['coords'], atom_names)):
        if name == 'CA' or name.endswith('_SC'):
            ca_or_sc_coords.append(coord)
    ca_or_sc_coords = np.array(ca_or_sc_coords)
    
    # Create distance matrix only for CA, SC, and ligand
    dist_matrix = create_dist_matrix(ca_or_sc_coords, ligand_coords, discretization_config='b12')

    features = init_dist_to_coords_features(pocket_info, ligand_size, dist_matrix, ligand_bond_matrix, ligand_atoms)
    features['x'] = np.concatenate([pocket_info['coords'], ligand_coords], axis=0)
    # features['atom_type'] = np.array(list(pocket_info['codes']) + [get_atom_one_hot(atom) for atom in ligand_atoms])

    return features

def init_dist_to_coords_features(pocket_info, ligand_size, dist_matrix, ligand_bond_matrix, ligand_atoms):
    receptor_coords = np.array(pocket_info['coords'])
    mol_types = pocket_info['mol_types']
    codes = pocket_info['codes']
    receptor_residues = pocket_info['res_ids']
    atom_names = pocket_info['atom_names']

    # Create ligand bonds matrix for all atoms
    protein_size = len(receptor_coords)
    ligand_bonds = np.zeros((protein_size + ligand_size, protein_size + ligand_size), dtype=np.int64)
    ligand_bonds[protein_size:, protein_size:] = ligand_bond_matrix
    
    node_attr = build_nodes(mol_types, codes, ligand_atoms)
    edge_index, edge_dist, edge_attr, edge_bonds = build_edges(receptor_coords, ligand_size, dist_matrix, receptor_residues, atom_names, ligand_bonds)
    
    features = {
        'h': node_attr,
        'edge_index': edge_index,
        'edge_dist': edge_dist,
        'edge_attr': edge_attr,
        'ligand_bonds': edge_bonds,
    }
    
    return features

def expand_dist_matrix(dist_matrix, receptor_coords, ligand_size, atom_names):
    """Expand reduced distance matrix (CA, SC, ligand only) to full node set.
    
    dist_matrix has shape (n_ca_sc + n_ligand, n_ca_sc + n_ligand)
    Returns full matrix with shape (n_receptor + n_ligand, n_receptor + n_ligand)
    where distances for CA/SC/ligand pairs are dist_matrix[i,j] + 1, others are 0.
    """
    n = len(receptor_coords) + ligand_size
    full_dist_matrix = np.zeros((n, n), dtype=np.int64)
    
    # Create mapping from full node index to reduced index in dist_matrix
    # Same order as used in create_dist_to_coords_features
    full_to_reduced = {}
    reduced_idx = 0
    for idx, name in enumerate(atom_names):
        if name == 'CA' or name.endswith('_SC'):
            full_to_reduced[idx] = reduced_idx
            reduced_idx += 1
    # Add ligand indices in order
    for idx in range(len(receptor_coords), n):
        full_to_reduced[idx] = reduced_idx
        reduced_idx += 1
    
    # Fill full matrix with reduced distances + 1 for CA/SC/ligand pairs
    for i in full_to_reduced:
        for j in full_to_reduced:
            if i != j:
                reduced_i, reduced_j = full_to_reduced[i], full_to_reduced[j]
                full_dist_matrix[i, j] = int(dist_matrix[reduced_i, reduced_j]) + 1
    
    return full_dist_matrix

def build_edges(receptor_coords, ligand_size, dist_matrix, receptor_residues, atom_names, ligand_bonds):
    """Build graph edges from coordinates and distance matrix.
    Creates a fully connected graph with residue relationship features.
    dist_matrix only contains distances between CA atoms, side chain centers, and ligand atoms.
    """
    n = len(receptor_coords) + ligand_size
    edge_list, edge_dist, edge_attr, edge_bonds_list = [], [], [], []
    
    # Extend data structures for ligand
    res_ids = receptor_residues + [receptor_residues[-1]+1 if receptor_residues else 0]*ligand_size
    
    # Expand reduced distance matrix to full node set
    full_dist_matrix = expand_dist_matrix(dist_matrix, receptor_coords, ligand_size, atom_names)
    
    for i in range(n):
        for j in range(i+1, n):
            # Bidirectional edges
            edge_list.append([i, j])
            edge_list.append([j, i])
            
            # Store discrete distance from expanded matrix
            discrete_dist = int(full_dist_matrix[i, j])
            edge_dist.append(discrete_dist)
            edge_dist.append(discrete_dist)
            
            res_i, res_j = res_ids[i], res_ids[j]
            is_same_residue = 1 if res_i == res_j else 0
            edge_attr.append([is_same_residue])
            edge_attr.append([is_same_residue])
            
            # Store ligand bond type
            bond_type = int(ligand_bonds[i, j])
            edge_bonds_list.append(bond_type)
            edge_bonds_list.append(bond_type)
        
    return np.array(edge_list).T, np.array(edge_dist), np.array(edge_attr), np.array(edge_bonds_list)

def build_nodes(mol_types, codes, ligand_atoms):
    node_attrs = []
    for i in range(len(mol_types)):
        node_attrs.append(np.concatenate([mol_types[i], codes[i]]))

    for atom_symbol in ligand_atoms:
        node_attrs.append(np.concatenate([[0, 0, 1], get_atom_one_hot(atom_symbol)]))
    
    return np.array(node_attrs)

def get_ligand_atoms_and_coords(mol):
    """Get ligand atoms, coordinates, and bond matrix from molecule."""
    ligand_atoms = []
    ligand_coords = []
    atom_idx_map = {}
    
    # Get the conformer from the molecule
    conf = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx_map[atom.GetIdx()] = len(ligand_atoms)
        # Convert atom symbol to ligand element format with _ prefix
        atom_symbol = '_' + atom.GetSymbol()
        ligand_atoms.append(atom_symbol)
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

def create_dist_matrix(receptor_coords, ligand_coords, discretization_config='b12'):
    """Helper function to create distance matrix from coordinates.
    
    Args:
        receptor_coords: Array of receptor atom coordinates (CA and SC centers)
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

class ContDataset(Dataset):
    """Dataset for predicting continuous features from distance matrix."""

    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='cont')
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
        # g.ndata['atom_type'] = torch.tensor(features['atom_type'], dtype=const.TORCH_FLOAT)
        g.ndata['h'] = torch.tensor(features['h'], dtype=const.TORCH_FLOAT)
        # g.ndata['anchor_mask'] = torch.tensor(features['anchor_mask'], dtype=const.TORCH_FLOAT)
        
        # Add edge data to edata - separate distance and residue attributes
        g.edata['edge_dist'] = torch.tensor(edge_dist, dtype=torch.long)
        g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
        g.edata['ligand_bonds'] = torch.tensor(features['ligand_bonds'], dtype=torch.long)
        
        return g
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for CoordsDataset using custom batch functionality"""
        return gnn.batch(batch_data)



