import torch
from rdkit import Chem

def generate_mappings(items_list):
    item2idx = {item: idx for idx, item in enumerate(items_list)}
    idx2item = {idx: item for idx, item in enumerate(items_list)}
    
    return item2idx, idx2item

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

ALLOWED_ATOM_TYPES = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P']
ALLOWED_RESIDUE_TYPES = [
    # Standard amino acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    
    # Modified amino acids (less common)
    'SEC', 'PYL', 'SEP', 'TPO', 'PTR',
    
    # Nucleotides (if working with DNA/RNA interfaces)
    'A', 'C', 'G', 'T', 'U', 'DA', 'DC', 'DG', 'DT',
    
    # Common cofactors/metals
    'HEM', 'FAD', 'NAD', 'ATP', 'GTP', 
    # 'HOH', 'ZN', 'MG', 'CA', 'FE'
]
GEOM_ATOM2IDX, GEOM_IDX2ATOM = generate_mappings(ALLOWED_ATOM_TYPES)
GEOM_RESIDUE2IDX, GEOM_IDX2RESIDUE = generate_mappings(ALLOWED_RESIDUE_TYPES)
GEOM_NUMBER_OF_ATOM_TYPES = len(GEOM_ATOM2IDX)
GEOM_NUMBER_OF_RESIDUE_TYPES = len(GEOM_RESIDUE2IDX)

# Atomic numbers (Z)
GEOM_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}

# One-hot atom types

# Dataset keys
DATA_LIST_ATTRS = {
    'uuid', 'name', 'protein_smi', 'ligand_smi', 'num_atoms'
}
DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot', 'anchors', 'protein_mask', 'ligand_mask', 'pocket_mask', 'protein_only_mask'
}
DATA_ATTRS_TO_ADD_LAST_DIM = {
    'anchors', 'protein_mask', 'ligand_mask', 'pocket_mask', 'protein_only_mask'
}


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

MARGINS_EDM = [10, 5, 2]
