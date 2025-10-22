#%%
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger
# Disable all RDKit logging (including warnings and errors)
RDLogger.DisableLog('rdApp.*')

def calculate_similarities(sdf_path, mol_path):
    """
    Calculate FP2 similarities between molecules in an SDF file and a reference MOL file.
    Skips molecules that raise errors when reading.
    """
    # Load the reference molecule
    try:
        ref_mol = Chem.MolFromMolFile(mol_path)
        if ref_mol is None:
            raise ValueError(f"Could not read reference molecule from {mol_path}")
        ref_fp = AllChem.GetMorganFingerprint(ref_mol, 2)  # FP2 is Morgan radius=2
    except Exception as e:
        print(f"Error loading reference molecule: {e}")
        return None

    similarities = []

    # Read the SDF file
    supplier = Chem.SDMolSupplier(sdf_path)
    for mol in supplier:
        if mol is None:
            continue  # Skip molecules that couldn't be read
            
        try:
            # Generate fingerprint for current molecule
            mol_fp = AllChem.GetMorganFingerprint(mol, 2)
            
            # Calculate similarity
            similarity = DataStructs.TanimotoSimilarity(ref_fp, mol_fp)
            similarities.append(similarity)
        except Exception as e:
            print(f"Skipping molecule due to error: {e}")
            continue
    
    return similarities

def plot_similarity_distribution(similarities, title):
    """Plot the distribution of similarity scores."""
    if not similarities:
        print(f"No valid similarities to plot for {title}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'FP2 Similarity Distribution: {title}')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.2)
    plt.show()

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs

def get_fingerprint(mol, fp_type="MACCS", ignore_atom_types=False, ignore_bond_types=False, ignore_bonds=False):
    if ignore_bonds:
        atoms = [a.GetSymbol() if not ignore_atom_types else 'C' for a in mol.GetAtoms()]
        smiles = '.'.join(atoms)
        mol = Chem.MolFromSmiles(smiles)

    if ignore_atom_types or ignore_bond_types:
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if ignore_atom_types:
                atom.SetAtomicNum(6)
        for bond in mol.GetBonds():
            if ignore_bond_types:
                bond.SetBondType(Chem.BondType.SINGLE)

    # 选择指纹类型
    if fp_type == "MACCS":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "Topological":
        return Chem.RDKFingerprint(mol)
    elif fp_type == "AtomPair":
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
    elif fp_type == "Torsion":
        return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")

def compare_molecules(mol1, mol2, fp_type="Topological", ignore_atom_types=False, ignore_bond_types=False, ignore_bonds=False):
    fp1 = get_fingerprint(mol1, fp_type=fp_type, ignore_atom_types=ignore_atom_types, ignore_bond_types=ignore_bond_types, ignore_bonds=ignore_bonds)
    fp2 = get_fingerprint(mol2, fp_type=fp_type, ignore_atom_types=ignore_atom_types, ignore_bond_types=ignore_bond_types, ignore_bonds=ignore_bonds)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# 示例分子：Aspirin vs Paracetamol
smiles1 = 'CC(=O)OC1=CC=CC=C1C(=O)O'
smiles2 = 'CC(=O)NC1=CC=C(C=C1)O'

mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)

# 测试不同指纹
for fp_type in ["MACCS", "Topological", "AtomPair", "Torsion"]:
    sim_all = compare_molecules(mol1, mol2, fp_type=fp_type)
    sim_ignore_atom_types = compare_molecules(mol1, mol2, fp_type=fp_type, ignore_atom_types=True)
    sim_ignore_bond_types = compare_molecules(mol1, mol2, fp_type=fp_type, ignore_bond_types=True)
    sim_ignore_atom_types_and_bond_types = compare_molecules(mol1, mol2, fp_type=fp_type, ignore_atom_types=True, ignore_bond_types=True)
    sim_ignore_bonds = compare_molecules(mol1, mol2, fp_type=fp_type, ignore_bonds=True)
    print(f"{fp_type} 相似性: {sim_all:.4f}, 忽略原子类型: {sim_ignore_atom_types:.4f}, 忽略键类型: {sim_ignore_bond_types:.4f}, 忽略原子和键类型: {sim_ignore_atom_types_and_bond_types:.4f}, 忽略所有键连接: {sim_ignore_bonds:.4f}")



#%%
from collections import defaultdict

def compare_atom_types_only(mol1, mol2):
    # 统计每个分子的原子类型计数
    def get_atom_counts(mol):
        counts = defaultdict(int)
        for atom in mol.GetAtoms():
            counts[atom.GetAtomicNum()] += 1
        return counts
    
    counts1 = get_atom_counts(mol1)
    counts2 = get_atom_counts(mol2)
    
    # 计算Jaccard相似度
    all_atoms = set(counts1.keys()).union(set(counts2.keys()))
    intersection = sum(min(counts1.get(a,0), counts2.get(a,0)) for a in all_atoms)
    union = sum(max(counts1.get(a,0), counts2.get(a,0)) for a in all_atoms)
    
    return intersection / union if union > 0 else 0

aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
paracetamol_smiles = 'CC(=O)NC1=CC=C(C=C1)O'

mol1 = Chem.MolFromSmiles(aspirin_smiles)
mol2 = Chem.MolFromSmiles(paracetamol_smiles)

# 正常对比
fp1 = get_fingerprint(mol1)
fp2 = get_fingerprint(mol2)
print("正常模式下相似性:", tanimoto_similarity(fp1, fp2))

# 忽略原子类型
fp1 = get_fingerprint(mol1, ignore_atom_types=True)
fp2 = get_fingerprint(mol2, ignore_atom_types=True)
print("忽略原子类型相似性:", tanimoto_similarity(fp1, fp2))

# 忽略原子和键类型
fp1 = get_fingerprint(mol1, ignore_atom_types=True, ignore_bond_types=True)
fp2 = get_fingerprint(mol2, ignore_atom_types=True, ignore_bond_types=True)
print("忽略原子和键类型相似性:", tanimoto_similarity(fp1, fp2))

# 完全忽略所有连接
print("忽略所有键连接的相似性:", compare_atom_types_only(mol1, mol2))


#%%


import os
from rdkit import Chem
from collections import defaultdict
from tqdm import tqdm

def find_duplicate_molecules(sdf_path):
    """Find duplicate molecules in an SDF file using InChIKeys."""
    # 使用InChIKey作为唯一标识符
    inchi_keys = []
    duplicates = defaultdict(list)
    
    supplier = Chem.SDMolSupplier(sdf_path)
    for idx, mol in tqdm(enumerate(supplier)):
        if mol is None:
            continue
            
        try:
            # 生成InChIKey
            inchi_key = Chem.MolToInchiKey(mol)
            duplicates[inchi_key].append(idx)
        except Exception as e:
            print(f"Skipping molecule {idx} due to error: {e}")
            continue
    
    # 筛选出有重复的分子
    duplicate_molecules = {k: v for k, v in duplicates.items() if len(v) > 1}
    
    return duplicates, duplicate_molecules

def analyze_5izj():
    # Define paths (modify these according to your actual directory structure)
    base_dir = '../tests'  # Assuming current directory, adjust as needed

    # Pair 1: 5izj_0_11_1000/xxx.sdf vs 5izj_data/5izj_0.mol
    sdf_path_0 = os.path.join(base_dir, '5izj_0_11_1000', 'all_yuel.sdf')  # Adjust filename if needed
    mol_path_0 = os.path.join(base_dir, '5izj_data', '5izj_0.mol')

    # Pair 2: 5izj_1_15_1000/xxx.sdf vs 5izj_data/5izj_1.mol
    sdf_path_1 = os.path.join(base_dir, '5izj_1_15_1000', 'all_yuel.sdf')  # Adjust filename if needed
    mol_path_1 = os.path.join(base_dir, '5izj_data', '5izj_1.mol')

    # Calculate and plot similarities for first pair
    print("Calculating similarities for 5izj_0...")
    similarities_0 = calculate_similarities(sdf_path_0, mol_path_0)
    if similarities_0:
        plot_similarity_distribution(similarities_0, '5izj_0_11_1000 vs 5izj_0.mol')
        print(f"Average similarity: {np.mean(similarities_0):.3f}")
    else:
        print("No valid similarities calculated for first pair")

    # Calculate and plot similarities for second pair
    print("\nCalculating similarities for 5izj_1...")
    similarities_1 = calculate_similarities(sdf_path_1, mol_path_1)
    if similarities_1:
        plot_similarity_distribution(similarities_1, '5izj_1_15_1000 vs 5izj_1.mol')
        print(f"Average similarity: {np.mean(similarities_1):.3f}")
    else:
        print("No valid similarities calculated for second pair")
# %%

duplicates, duplicate_molecules = find_duplicate_molecules('../tests/5izj_0_11_1000/all_yuel.sdf')

#%%
print(f"found {len(duplicates)} sets of distinct molecules:")
print(f"{len(duplicate_molecules)} of them have duplicate molecules:")
# for inchi_key, indices in duplicates.items():
#     print(f"InChIKey: {inchi_key} - Found at positions: {indices}")
# %%
analyze_5izj()
# %%
