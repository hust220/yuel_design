import torch
from rdkit import Chem
import numpy as np

def generate_mappings(items_list):
    item2idx = {item: idx for idx, item in enumerate(items_list)}
    idx2item = {idx: item for idx, item in enumerate(items_list)}
    
    return item2idx, idx2item, len(items_list)

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

# All protein atom types - comprehensive dictionary of atom names by element (excluding hydrogen atoms)
PROTEIN_ATOM_TYPES = {
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

COMMON_PDB_ELEMENT_TYPES = [element for element, _ in PROTEIN_ATOM_TYPES.items()]
ALLOWED_PDB_ATOM_TYPES = ['X'] + [atom for _, atoms in PROTEIN_ATOM_TYPES.items() for atom in atoms]
PDB_ATOM2IDX, IDX2PDB_ATOM, N_PDB_ATOM_TYPES = generate_mappings(ALLOWED_PDB_ATOM_TYPES)

# Allowed element types (merged from ALLOWED_ATOM_TYPES and additional elements)
ALLOWED_ELEMENT_TYPES = [
    # Other elements
    'X',
    # Basic elements
    'C', 'O', 'N', 'F', 'S', 'P',
    # Halogens
    'Cl', 'Br', 'I',
    # Metals
    'ZN', 'MG', 'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'W', 'CA',
    # Other elements
    'SE'
]
ELEMENT2IDX, IDX2ELEMENT, N_ELEMENT_TYPES = generate_mappings(ALLOWED_ELEMENT_TYPES)

# Keep ALLOWED_ATOM_TYPES for backward compatibility (will be redefined below)
ALLOWED_ATOM_TYPES_OLD = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P']
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
RESIDUE2IDX, IDX2RESIDUE, N_RESIDUE_TYPES = generate_mappings(ALLOWED_RESIDUE_TYPES)

# Define unified atom types for coarse-grained representation
# Protein atoms: CA + 20 standard amino acid side chains + X_SC for unknown
protein_atoms = ['CA'] + [f'{aa}_SC' for aa in ALLOWED_RESIDUE_TYPES if len(aa) == 3] + ['X_SC']
# Ligand elements: common elements from ALLOWED_ELEMENT_TYPES + X for unknown
ligand_elements = ['X'] + [e for e in ALLOWED_ELEMENT_TYPES if e != 'X']
# Combined atom types
ALLOWED_ATOM_TYPES = protein_atoms + ligand_elements
ATOM2IDX, IDX2ATOM, N_ATOM_TYPES = generate_mappings(ALLOWED_ATOM_TYPES)

# Flatten PROTEIN_ATOM_TYPES dictionary to list for mapping generation
PROTEIN_ATOM_LIST = []
for element, atom_names in PROTEIN_ATOM_TYPES.items():
    PROTEIN_ATOM_LIST.extend(atom_names)

PROTEIN_ATOM2IDX, IDX2PROTEIN_ATOM, N_PROTEIN_ATOM_TYPES = generate_mappings(PROTEIN_ATOM_LIST)

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

# Standard amino acid atom names in PDB format (excluding hydrogen atoms)
# Each residue includes backbone atoms (N, CA, C, O) and side chain atoms
# Order: backbone atoms first, then side chain atoms
AMINO_ACID_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
}

# For unknown residues, use a default set
AMINO_ACID_ATOMS['UNK'] = ['N', 'CA', 'C', 'O']
