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
protein_atoms = ['CA'] + [f'{aa}_SC' for aa in STANDARD_AMINO_ACIDS]  # Includes GLY_SC

# Ligand elements: common elements
ligand_elements = ['C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I',
                   'ZN', 'MG', 'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'W', 'SE']

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

# def parse_structure_and_ligand(pocket_pdb, ligand_size, mask_ligand=True):
def parse_pocket(structure):
    """Parse protein structure using coarse-grained representation.
    Each residue is simplified to two atoms: CA (backbone) and side chain center.
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
                    # Add CA atom
                    coords.append(ca_coord)
                    atom_names.append('CA')
                    res_ids.append(ires)
                    res_names.append(res_name)
                    codes.append(get_atom_one_hot('CA'))
                    mol_types.append([1, 0, 0])  # backbone 
                    
                    # Calculate side chain center (exclude backbone atoms: CA, N, C, O)
                    side_chain_atoms = [coord for name, coord in residue_atoms 
                                        if name not in ['CA', 'N', 'C', 'O']]
                    
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
                        mol_types.append([0, 1, 0])  # side chain
                        
                        # Store CA coordinate for backbone
                    else:
                        # Glycine (GLY) has no side chain atoms
                        # Option: Use CA's direction vector to place side chain
                        # Get C and N atom positions for direction calculation
                        n_coord = next((coord for name, coord in residue_atoms if name == 'N'), None)
                        c_coord = next((coord for name, coord in residue_atoms if name == 'C'), None)
                        
                        if n_coord is not None and c_coord is not None:
                            # Create a small offset along the bisector of N-CA-C angle
                            # This gives a reasonable position for the "missing" side chain
                            n_vec = n_coord - ca_coord
                            c_vec = c_coord - ca_coord
                            # Average direction
                            avg_vec = (n_vec + c_vec) / 2
                            # Normalize and scale by typical side chain length (1.5 Angstroms)
                            norm = np.linalg.norm(avg_vec)
                            if norm > 0:
                                avg_vec = avg_vec / norm * 1.5
                                sc_coord = ca_coord + avg_vec
                            else:
                                # Avoid zero norm: use a small random offset
                                avg_vec = np.array([1.0, 0.0, 0.0]) * 1.5
                                sc_coord = ca_coord + avg_vec
                        else:
                            # Fallback: use a small fixed offset to avoid overlap
                            sc_coord = ca_coord + np.array([1.5, 0.0, 0.0])
                        
                        coords.append(sc_coord)
                        atom_names.append('GLY_SC')
                        res_ids.append(ires)
                        res_names.append(res_name)
                        codes.append(get_atom_one_hot('GLY_SC'))
                        mol_types.append([0, 0, 1])  # side chain
                    
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
    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol)
    ligand_coords = np.array(ligand_coords)
    ligand_size = len(ligand_coords)

    features = init_dist_to_coords_features(pocket_info, ligand_size)
    features['x'] = np.concatenate([pocket_info['coords'], ligand_coords], axis=0)

    return features

def init_dist_to_coords_features(pocket_info, ligand_size):
    receptor_coords = np.array(pocket_info['coords'])
    mol_types = pocket_info['mol_types']
    codes = pocket_info['codes']
    receptor_residues = pocket_info['res_ids']

    node_attr, anchor_mask = build_nodes(mol_types, codes, ligand_size)
    edge_index, edge_attr = build_edges(receptor_coords, ligand_size, receptor_residues)
    
    features = {
        'h': node_attr,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'anchor_mask': anchor_mask,
    }
    
    return features

def build_edges(receptor_coords, ligand_size, receptor_residues):
    """Build graph edges from coordinates and distance matrix.
    Creates a fully connected graph with residue relationship features.
    """
    n = len(receptor_coords) + ligand_size
    edge_list, edge_attr = [], []
    
    # Extend data structures for ligand
    res_ids = receptor_residues + [receptor_residues[-1]+1 if receptor_residues else 0]*ligand_size
    
    for i in range(n):
        for j in range(i+1, n):
            # Bidirectional edges
            edge_list.append([i, j])
            edge_list.append([j, i])
                        
            res_i, res_j = res_ids[i], res_ids[j]
            is_same_residue = 1 if res_i == res_j else 0
            edge_attr.append([is_same_residue])
            edge_attr.append([is_same_residue])
        
    return np.array(edge_list).T, np.array(edge_attr)

def build_nodes(mol_types, codes, ligand_size):
    node_attrs, anchor_mask = [], []
    for i in range(len(mol_types)):
        node_attrs.append(np.concatenate([mol_types[i], codes[i]]))
        # Only CA (backbone) should be anchors; exclude side chains
        is_backbone = 1 if mol_types[i][0] == 1 else 0
        anchor_mask.append(is_backbone)

    for i in range(ligand_size):
        node_attrs.append(np.concatenate([[0, 0, 1], get_atom_one_hot('X')]))
        anchor_mask.append(0)
    
    return np.array(node_attrs), np.array(anchor_mask)

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

class E2EDataset(Dataset):
    """Dataset for predicting coordinates from distance matrix."""
    
    def __init__(self, split='train', cache_mode='memory', cache_dir='cache'):
        self.cache = FileCache(cache_mode=cache_mode, cache_dir=cache_dir, dataset_name='e2e')
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
                
        # Create custom graph
        g = gnn.graph(edge_index, num_nodes=x.shape[0])
        
        # Add node data to ndata
        g.ndata['x'] = x
        g.ndata['h'] = torch.tensor(features['h'], dtype=const.TORCH_FLOAT)
        g.ndata['anchor_mask'] = torch.tensor(features['anchor_mask'], dtype=const.TORCH_FLOAT)
        g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
        
        return g
    
    @staticmethod
    def collate_fn(batch_data):
        """Collate function for CoordsDataset using custom batch functionality"""
        return gnn.batch(batch_data)



