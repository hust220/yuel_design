import os
import numpy as np
import pandas as pd
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const
from Bio.PDB import PDBParser
from pdb import set_trace

def parse_residues(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            # atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                # pocket_types.append(atom_type)
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
    n1 = const.GEOM_NUMBER_OF_RESIDUE_TYPES
    n2 = const.GEOM_NUMBER_OF_ATOM_TYPES
    one_hot = np.zeros(n1 + n2)
    one_hot[n1 + const.GEOM_ATOM2IDX[atom]] = 1
    return one_hot

# one hot for amino acids
def aa_one_hot(residue):
    n1 = const.GEOM_NUMBER_OF_RESIDUE_TYPES
    n2 = const.GEOM_NUMBER_OF_ATOM_TYPES
    one_hot = np.zeros(n1 + n2)
    one_hot[const.GEOM_RESIDUE2IDX[residue]] = 1
    return one_hot

def molecule_feat_mask():
    n1 = const.GEOM_NUMBER_OF_RESIDUE_TYPES
    n2 = const.GEOM_NUMBER_OF_ATOM_TYPES
    mask = np.zeros(n1 + n2)
    mask[n1:] = 1
    return mask

def parse_molecule(mol):
    one_hot = []
    charges = []
    for atom in mol.GetAtoms():
        one_hot.append(atom_one_hot(atom.GetSymbol()))
        charges.append(const.GEOM_CHARGES[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)

def parse_pocket(rs):
    pocket_coords = []
    pocket_types = []

    for residue in rs:
        residue_name = residue.get_resname()
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            # atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            if atom_name == 'CA':
                pocket_coords.append(atom_coord.tolist())
                # pocket_types.append(atom_type)
                pocket_types.append(residue_name)

    pocket_one_hot = []
    for _type in pocket_types:
        pocket_one_hot.append(aa_one_hot(_type))
    pocket_one_hot = np.array(pocket_one_hot)

    return pocket_coords, pocket_one_hot

def get_pocket(mol, pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for ir,residue in enumerate(struct.get_residues()):
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(ir)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(axis=1) <= 6)[0]])

    return parse_pocket([r for (ir, r) in enumerate(struct.get_residues()) if ir in contact_residues])

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

class MOADDataset(Dataset):
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
            # print(222,one_hot)

            fragment_mask = np.zeros(pocket_size + mol_size)
            linker_mask = np.zeros(pocket_size + mol_size)
            fragment_mask[:pocket_size] = 1
            linker_mask[pocket_size:] = 1

            data.append({
                'name': molecule_name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
            })

        return data


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

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    # 为什么MOAD的edge_mask是batch_idx呢？
    # batch_mask = torch.cat([
    #     torch.ones(n_nodes, dtype=const.TORCH_INT) * i
    #     for i in range(batch_size)
    # ]).to(atom_mask.device)
    # out['edge_mask'] = batch_mask

    edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)

def create_templates_for_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes
    """
    decoupled_data = []
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data['fragment_mask'][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                # Should write fragment-related data + (zeros x linker_size)
                fill_value = 1 if k == 'linker_mask' else 0
                template = create_template(v[i], fragment_size, linker_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)

