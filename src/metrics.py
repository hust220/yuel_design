import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from src import const
from src.molecule_builder import get_bond_order
from scipy.stats import wasserstein_distance
from src.delinker_utils import frag_utils
import math
import numpy as np
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from pdb import set_trace

import math
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

def sas(mol):
    """
    Custom Synthetic Accessibility Score (SAS) implementation based on:
    1. Molecular complexity (rings, branches, stereocenters)
    2. Fragment contributions
    3. Penalties for problematic groups
    
    Returns: Score between 1 (easy to make) and 10 (very hard)
    """
    if not mol or mol.GetNumAtoms() == 0:
        return 10.0  # Worst score for invalid molecules
    
    # 1. Basic molecular properties
    n_atoms = mol.GetNumAtoms()
    n_rings = len(Chem.GetSymmSSSR(mol))
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_rot_bonds = Lipinski.NumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)
    
    # 2. Fragment-based complexity (modified Ertl's method)
    fragment_score = 0
    frags = Chem.GetMolFrags(mol, asMols=True)
    for frag in frags:
        frag_size = frag.GetNumAtoms()
        if frag_size <= 2:
            fragment_score += 1
        elif frag_size <= 5:
            fragment_score += 2
        else:
            fragment_score += 3 + math.log(frag_size - 4, 2)
    
    # 3. Problematic groups penalty
    penalty = 0
    problematic_smarts = [
        '[#16;D2](=[#8])(=[#8])',  # Sulfones
        '[#7;!R]=[#8;!R]',         # Nitro groups
        '[#6](=[#8])[#6](=[#8])',  # 1,2-dicarbonyl
        '[#7;R]=[#8]',             # Cyclic amides
        '[#6]#[#6]',               # Alkynes
        '[#16]',                   # Sulfur atoms
        '[#15]',                   # Phosphorus atoms
    ]
    
    for smarts in problematic_smarts:
        penalty += len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
    
    # 4. Calculate raw score (lower = easier)
    raw_score = (
        0.5 * n_rings + 
        0.5 * n_stereo + 
        0.3 * n_rot_bonds + 
        0.1 * fragment_score + 
        0.8 * penalty +
        math.log(mw / 100, 2)
    )
    
    # 5. Normalize to 1-10 scale (empirically determined)
    sas = min(10.0, max(1.0, 1.0 + (raw_score / 2.5)))
    
    return round(sas, 2)

def safe_exp(x):
    """Numerically stable exponential function with clipping"""
    try:
        return math.exp(min(max(x, -700), 700))  # Prevent overflow
    except:
        return 0.0 if x < 0 else float('inf')

def sigmoid_transform(x, a, b, c, d):
    """Numerically stable double sigmoid transform"""
    # First sigmoid: 1/(1 + exp(-(x-a)/b))
    try:
        term1 = 1.0 / (1.0 + safe_exp(-(x - a)/b))
    except:
        term1 = 0.0 if x < a else 1.0
    
    # Second sigmoid: 1/(1 + exp((x-c)/d))
    try:
        term2 = 1.0 / (1.0 + safe_exp((x - c)/d))
    except:
        term2 = 0.0 if x > c else 1.0
    
    return term1 * term2

def qed(mol):
    """Fixed version with stable numerical transforms"""
    if not mol:
        return 0.0
    
    # Adjusted parameters for real-world ranges
    sig_params = {
        'MW': [100, 50, 400, 50],      # MW typically 0-500 Da
        'ALOGP': [-1, 1, 5, 1],        # LogP typically -2 to 6
        'HBA': [0, 1, 8, 1],            # HBA typically 0-10
        'HBD': [0, 1, 5, 1],            # HBD typically 0-5
        'PSA': [0, 10, 150, 10],        # PSA typically 0-200 Å²
        'ROTB': [0, 1, 8, 1],           # Rotatable bonds typically 0-10
        'AROM': [0, 1, 4, 1],           # Aromatic rings typically 0-5
        'ALERTS': [0, 0.1, 1, 0.1],     # Binary-like (0-1)
    }
    
    # Calculate descriptors (same as before)
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'ALOGP': Descriptors.MolLogP(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'PSA': rdMolDescriptors.CalcTPSA(mol),
        'ROTB': Lipinski.NumRotatableBonds(mol),
        'AROM': Lipinski.NumAromaticRings(mol),
        'ALERTS': 0,
    }
    
    # Apply stable transformation
    transformed = {}
    for key, value in descriptors.items():
        a, b, c, d = sig_params[key]
        transformed[key] = sigmoid_transform(value, a, b, c, d)
    
    # Geometric mean
    product = 1.0
    for t in transformed.values():
        product *= max(0.001, t)  # Prevent zeros
    
    return product ** (1/len(transformed))


def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


def is_connected(mol):
    try:
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
    except Chem.rdchem.AtomValenceException:
        return False
    if len(mol_frags) != 1:
        return False
    return True


def get_valid_molecules(molecules):
    valid = []
    for mol in molecules:
        if is_valid(mol):
            valid.append(mol)
    return valid


def get_connected_molecules(molecules):
    connected = []
    for mol in molecules:
        if is_connected(mol):
            connected.append(mol)
    return connected


def get_unique_smiles(valid_molecules):
    unique = set()
    for mol in valid_molecules:
        unique.add(Chem.MolToSmiles(mol))
    return list(unique)


def get_novel_smiles(unique_true_smiles, unique_pred_smiles):
    return list(set(unique_pred_smiles).difference(set(unique_true_smiles)))


def compute_energy(mol):
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    energy = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0).CalcEnergy()
    return energy


def wasserstein_distance_between_energies(true_molecules, pred_molecules):
    true_energy_dist = []
    for mol in true_molecules:
        try:
            energy = compute_energy(mol)
            true_energy_dist.append(energy)
        except:
            continue

    pred_energy_dist = []
    for mol in pred_molecules:
        try:
            energy = compute_energy(mol)
            pred_energy_dist.append(energy)
        except:
            continue

    if len(true_energy_dist) > 0 and len(pred_energy_dist) > 0:
        return wasserstein_distance(true_energy_dist, pred_energy_dist)
    else:
        return 0


def compute_metrics(pred_molecules, true_molecules):
    if len(pred_molecules) == 0:
        return None

    # Passing rdkit.Chem.Sanitize filter
    true_valid = get_valid_molecules(true_molecules)
    pred_valid = get_valid_molecules(pred_molecules)
    validity = len(pred_valid) / len(pred_molecules)

    # Checking if molecule consists of a single connected part
    true_valid_and_connected = get_connected_molecules(true_valid)
    pred_valid_and_connected = get_connected_molecules(pred_valid)
    validity_and_connectivity = len(pred_valid_and_connected) / len(pred_molecules)

    # Unique molecules
    true_unique = get_unique_smiles(true_valid_and_connected)
    pred_unique = get_unique_smiles(pred_valid_and_connected)
    uniqueness = len(pred_unique) / len(pred_valid_and_connected) if len(pred_valid_and_connected) > 0 else 0

    # Novel molecules
    pred_novel = get_novel_smiles(true_unique, pred_unique)
    novelty = len(pred_novel) / len(pred_unique) if len(pred_unique) > 0 else 0

    # Difference between Energy distributions
    energies = wasserstein_distance_between_energies(true_valid_and_connected, pred_valid_and_connected)

    # calculate the average qed and sas score
#    QEDs = []
#    SASs = []
#    for mol in pred_molecules:
#        QEDs.append(qed(mol))
#        SASs.append(sas(mol))
#    mean_qed = np.mean(QEDs)
#    mean_sas = np.mean(SASs)

    return {
        'validity': validity,
        'validity_and_connectivity': validity_and_connectivity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'energies': energies,
#        'qed': mean_qed,
#        'sas': mean_sas,
    }


# def check_stability(positions, atom_types):
#     assert len(positions.shape) == 2
#     assert positions.shape[1] == 3
#     x = positions[:, 0]
#     y = positions[:, 1]
#     z = positions[:, 2]
#
#     nr_bonds = np.zeros(len(x), dtype='int')
#     for i in range(len(x)):
#         for j in range(i + 1, len(x)):
#             p1 = np.array([x[i], y[i], z[i]])
#             p2 = np.array([x[j], y[j], z[j]])
#             dist = np.sqrt(np.sum((p1 - p2) ** 2))
#             atom1, atom2 = const.IDX2ATOM[atom_types[i].item()], const.IDX2ATOM[atom_types[j].item()]
#             order = get_bond_order(atom1, atom2, dist)
#             nr_bonds[i] += order
#             nr_bonds[j] += order
#     nr_stable_bonds = 0
#     for atom_type_i, nr_bonds_i in zip(atom_types, nr_bonds):
#         possible_bonds = const.ALLOWED_BONDS[const.IDX2ATOM[atom_type_i.item()]]
#         if type(possible_bonds) == int:
#             is_stable = possible_bonds == nr_bonds_i
#         else:
#             is_stable = nr_bonds_i in possible_bonds
#         nr_stable_bonds += int(is_stable)
#
#     molecule_stable = nr_stable_bonds == len(x)
#     return molecule_stable, nr_stable_bonds, len(x)
#
#
# def count_stable_molecules(one_hot, x, node_mask):
#     stable_molecules = 0
#     for i in range(len(one_hot)):
#         mol_size = node_mask[i].sum()
#         atom_types = one_hot[i][:mol_size, :].argmax(dim=1).detach().cpu()
#         positions = x[i][:mol_size, :].detach().cpu()
#         stable, _, _ = check_stability(positions, atom_types)
#         stable_molecules += int(stable)
#
#     return stable_molecules

