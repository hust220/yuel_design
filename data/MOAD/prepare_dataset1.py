import argparse
import numpy as np
import pandas as pd
import os, sys
import pickle
from multiprocessing import Pool
from functools import partial
from rdkit import Chem
from tqdm import tqdm

sys.path.append('../../')
from src.datasets import parse_molecule, get_pocket

def read_sdf_blocks(sdf_path):
    """Read and split SDF file into individual molecule blocks"""
    with open(sdf_path, 'r') as f:
        content = f.read().split('$$$$\n')
    return [block.strip() for block in content if block.strip()]

def process_sdf_block(sdf_block, proteins_path):
    """Process single SDF block (atomic operation for multiprocessing)"""
    try:
        # Create molecule from block inside worker process
        mol = Chem.MolFromMolBlock(sdf_block)
        if not mol:
            return None

        mol_name = mol.GetProp('_Name')
        mol_smi = Chem.MolToSmiles(mol)
        mol_pos, mol_one_hot = parse_molecule(mol)

        # Create protein pocket
        pdb_code = mol_name.split('_')[0]
        pdb_path = os.path.join(proteins_path, f'{pdb_code}_protein.pdb')
        # print(pdb_path)
        pocket_pos, pocket_one_hot = get_pocket(mol, pdb_path)

        # print(len(mol_pos), len(pocket_pos))
        if len(pocket_pos) == 0 or (len(mol_pos) + len(pocket_pos)) >= 1000:
            return None

        # Store molecule as binary data and temporary metadata
        # print(mol_name)
        return {
            'molecule': mol_name,
            'molecule_pos': mol_pos,
            'molecule_one_hot': mol_one_hot,
            'pocket_pos': pocket_pos,
            'pocket_one_hot': pocket_one_hot,

            'sdf_block': sdf_block,
            'smiles': mol_smi,
            'pocket_size': len(pocket_pos),
            'molecule_size': mol.GetNumAtoms(),
            # 'mol_binary': mol.ToBinary()
        }
    except Exception as e:
        return None

def main(args):
    # Read all SDF blocks first (fast IO operation)
    sdf_blocks = read_sdf_blocks(args.sdf)
    print(f"Loaded {len(sdf_blocks)} molecules")
 
    # Process blocks in parallel
    with Pool(args.num_workers) as pool:
        processor = partial(process_sdf_block, proteins_path=args.proteins)
        results = list(tqdm(
            pool.imap(processor, sdf_blocks), 
            total=len(sdf_blocks),
            desc="Processing molecules"
        ))

    # Reconstruct final data in main process
    data = [result for result in results if result]
    print(f"Successfully processed {len(data)} molecules")
    
    # Save with main process molecules
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf', type=str, required=True)
    parser.add_argument('--proteins', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    main(args)
