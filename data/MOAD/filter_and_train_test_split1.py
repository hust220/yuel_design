import argparse
import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from tqdm import tqdm
import os
import sys
sys.path.append('../../')
from src.const import ALLOWED_ATOM_TYPES, ALLOWED_RESIDUE_TYPES

script_dir = os.path.dirname(os.path.abspath(__file__))
TEST_PDBS_PATH = os.path.join(script_dir, '../../resources/moad_test_pdbs.txt')
VAL_PDBS_PATH = os.path.join(script_dir, '../../resources/moad_val_pdbs.txt')

def assign_dataset(name, test_pdbs, val_pdbs):
    pdb = name.split('_')[0]
    if pdb in test_pdbs:
        return 'test'
    if pdb in val_pdbs:
        return 'val'
    return 'train'

def filter_and_split(input):
    mols_sdf = Chem.SDMolSupplier(mol_path, sanitize=False)
    pocket_data = pickle.load(open(pockets_path, 'rb'))

    table = pd.read_csv(table_path)

    # 1. Filter by molecule size
    table.loc[(table.pocket_bb_size + table.molecule_size) >= 1000, 'discard'] = True

    # 2. Filter by allowed ligand atom types
    for i, mol in tqdm(enumerate(mols_sdf), total=len(mols_sdf)):
        types = set()
        for atom in mol.GetAtoms():
            types.add(atom.GetSymbol())

        if len(types.difference(ALLOWED_ATOM_TYPES)) > 0:
            table.loc[i, 'discard'] = True

    # 3. Filter by pocket atom types
    for i, pdata in tqdm(enumerate(pocket_data), total=len(pocket_data)):
        types = set(pdata['types'])
        if len(types.difference(ALLOWED_RESIDUE_TYPES)) > 0:
            table.loc[i, 'discard'] = True

    # 4. Filter by pocket size
    for i, pdata in tqdm(enumerate(pocket_data), total=len(pocket_data)):
        if len(pdata['coord']) == 0:
            table.loc[i, 'discard'] = True

    # Split in train, test, val
    test_pdbs = np.loadtxt(TEST_PDBS_PATH, dtype='str')
    val_pdbs = np.loadtxt(VAL_PDBS_PATH, dtype='str')
    table['dataset'] = table['molecule_name'].apply(lambda x: assign_dataset(x, test_pdbs, val_pdbs))
    print('Train:', len(table[(~table.discard) & (table.dataset == 'train')]))
    print('Test:', len(table[(~table.discard) & (table.dataset == 'test')]))
    print('Val:', len(table[(~table.discard) & (table.dataset == 'val')]))

    mols = {
        'train': [],
        'val': [],
        'test': [],
    }
    pockets = {
        'train': [],
        'val': [],
        'test': [],
    }
    idx = {
        'train': [],
        'val': [],
        'test': [],
    }

    for i, (m, p) in tqdm(enumerate(zip(mols_sdf, pocket_data)), total=len(mols_sdf)):
        discard = table.loc[i, 'discard']
        dataset = table.loc[i, 'dataset']
        if discard:
            continue

        mols[dataset].append(m)
        pockets[dataset].append(p)
        idx[dataset].append(i)

    tables = {
        'train': table.loc[idx['train']].copy().reset_index(drop=True),
        'val': table.loc[idx['val']].copy().reset_index(drop=True),
        'test': table.loc[idx['test']].copy().reset_index(drop=True),
    }

    # Saving datasets
    template = mol_path.replace('_mol.sdf', '')
    for dataset in ['train', 'val', 'test']:
        mols_len = len(mols[dataset])
        pockets_len = len(pockets[dataset])
        table_len = len(tables[dataset])
        assert len({mols_len, pockets_len, table_len}) == 1

        mol_sdf_path = f'{template}_{dataset}_mol.sdf'
        pockets_sdf_path = f'{template}_{dataset}_pockets.pkl'
        table_out_path = f'{template}_{dataset}_table.csv'

        with Chem.SDWriter(open(mol_sdf_path, 'w')) as writer:
            for mol in tqdm(mols[dataset], desc=dataset):
                writer.write(mol)

        with open(pockets_sdf_path, 'wb') as f:
            pickle.dump(pockets[dataset], f)

        tables[dataset].to_csv(table_out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle', action='store', type=str, required=True)
    args = parser.parse_args()

    filter_and_split(
        input=args.pickle,
    )
