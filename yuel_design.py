import argparse
import os
import numpy as np

import torch
from rdkit import Chem
from Bio.PDB import PDBParser

from src import const
from src.datasets import (
    collate, get_dataloader, ProteinLigandDataset, parse_residues, parse_pocket
)
from src.lightning import DDPM
from src.utils import FoundNaNException, set_deterministic
from tqdm import tqdm
from src.molecule_builder import build_molecules
from src import metrics
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pocket', action='store', type=str, required=False,
    help='Path to the file with pocket atoms'
)
parser.add_argument(
    '--dataset', action='store', type=str, required=False,
    help='Path to the dataset'
)
parser.add_argument(
    '--model', action='store', type=str, required=True,
    help='Path to the DiffLinker model'
)
parser.add_argument(
    '--size', action='store', type=str, required=True,
    help='Linker size (int) or allowed size boundaries (comma-separated integers) or path to the size prediction model'
)
parser.add_argument(
    '--output', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules will be saved'
)
parser.add_argument(
    '--n_samples', action='store', type=int, required=False, default=5,
    help='Number of linkers to generate'
)
parser.add_argument(
    '--random_seed', action='store', type=int, required=False, default=None,
    help='Random seed'
)
parser.add_argument(
    '--trajectory', action='store', type=str, required=False, default=None,
    help='trajectory directory'
)

def save_xyz_file(path, one_hot, positions, node_mask, names, metadata=None, append=False):
    idx2atom = const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        if append:
            f = open(os.path.join(path, f'{names[batch_i]}.xyz'), "a")
        else:
            f = open(os.path.join(path, f'{names[batch_i]}.xyz'), "w")

        f.write("%d\n" % n_atoms)
        if metadata is not None:
            f.write("%s\n" % ','.join([f"{k}:{v}" for k, v in metadata.items()]))
        else:
            f.write("\n")
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()

def read_molecule(path):
    if path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)[0]
    raise Exception('Unknown file extension')

def read_pocket(path):
    struct = PDBParser().get_structure('', path)
    return struct.get_residues()

def save_pdb_file(path, one_hot, positions, node_mask, names):
    idx2atom = const.IDX2ATOM
    pdb_path = os.path.join(f'{path}.pdb')
    with open(pdb_path, 'w') as f:
        for batch_i in reversed(range(one_hot.size(0))):
            mask = node_mask[batch_i].squeeze()
            atom_idx = torch.where(mask)[0]
            atoms = torch.argmax(one_hot[batch_i], dim=1)
            
            # Write MODEL header
            f.write(f"MODEL     {batch_i+1}\n")
            f.write(f"REMARK 350 Generated from {names[batch_i]}\n")
            
            # Write atoms
            for serial, atom_i in enumerate(atom_idx, start=1):
                atom_type = idx2atom[atoms[atom_i].item()]
                x, y, z = positions[batch_i, atom_i].tolist()
                
                # Format PDB ATOM line (PDB format specification)
                f.write(
                    "ATOM  {serial:5d} {atom_type:4s} UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}"
                    "  1.00  0.00          {atom_type:>2s}\n".format(
                        serial=serial,
                        atom_type=atom_type,
                        x=x,
                        y=y,
                        z=z
                    )
                )
            
            # Close model section
            f.write("ENDMDL\n")
        
        # Final PDB end marker
        f.write("END\n")

def record_trajectories(directory, imol, chain_batch, node_mask):
    # batch_indices, mol_indices = utils.get_batch_idx_for_animation(self.batch_size, batch_i)
    nframes, batch, n_atoms, n_dims = chain_batch.shape
    # print(chain_batch.shape)
    i = imol
    for bi in range(batch):
        chain = chain_batch[:, bi, :, :]
        name = f'mol_{i:03d}'
        chain_output = os.path.join(directory, name)
        os.makedirs(chain_output, exist_ok=True)

        one_hot = chain[:, :, 3+const.N_RESIDUE_TYPES:]
        positions = chain[:, :, :3]
        chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(nframes)], dim=0)
        names = [f'{name}_{j:03d}' for j in range(nframes)]

        save_pdb_file(chain_output, one_hot, positions, chain_node_mask, names=names)
        i += 1
        # visualize_chain(chain_output, wandb=wandb, mode=name)

def get_sssr_rings(mol):
    sssr_rings = Chem.GetSymmSSSR(mol)  # Get SSSR rings
    return [list(ring) for ring in sssr_rings]

def prepare_single_dataset(pocket_path, device):
    positions = np.empty((0, 3), dtype=np.float32)
    one_hot = np.empty((0, const.N_RESIDUE_TYPES + const.N_ATOM_TYPES), dtype=np.float32)
    pocket_size = 0

    if pocket_path is not None:
        pocket_extension = pocket_path.split('.')[-1]
        if pocket_extension != 'pdb':
            print('Please upload the pocket file in .pdb format')
            return
        try:
            print('Reading pocket:', pocket_path)
            pocket_data = read_pocket(pocket_path)
        except Exception as e:
            print(f'Could not read the file with pocket: {e}')
            return
        pocket_pos, pocket_one_hot = parse_pocket(pocket_data)
        pocket_size = len(pocket_one_hot)
        positions = pocket_pos
        one_hot = pocket_one_hot

    linker_mask = np.zeros(pocket_size)
    fragment_mask = np.ones(pocket_size)

    dataset = [{
        'name': '0',
        'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
    }]
    return ProteinLigandDataset(data=dataset, device=device)

def design_ligands(dataset, model, output_dir, n_samples, mol_size, random_seed, trajectory, device):

    os.makedirs(output_dir, exist_ok=True)

    if random_seed is not None:
        set_deterministic(random_seed)

    print(f'Will generate compounds with {mol_size} atoms')
    mol_size = int(mol_size)

    def sample_fn(_data):
        return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * mol_size

    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    # ddpm.val_dataset = dataset

    dataloader = get_dataloader(
        dataset, batch_size=1, collate_fn=collate
    )

    # Sampling
    print('Sampling...')
    # save a summary csv file in the output directory
    bar = tqdm(total=n_samples*len(dataset), desc='Sampling', unit='molecule')
    summary_file = os.path.join(output_dir, f'summary_size{mol_size}_n{n_samples}_s{random_seed}.csv')
    # if summary file exists, append to it
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f:
            f.write('name,n_samples,mol_size,random_seed,duration,attempts,invalid,unconnected,large_rings\n')

    for data in dataloader:
        n_attempts = 0
        n_attempts_invalid = 0
        n_attempts_unconnected = 0
        n_attempts_large_rings = 0
        start_time = time.time()
        name = data['name'][0]
        imol = 0

        while imol < n_samples:
            n_attempts += 1
            bar.set_postfix({
                'Attempts': n_attempts,
                'Invalid': n_attempts_invalid,
                'Unconnected': n_attempts_unconnected,
                'Large Rings': n_attempts_large_rings
            })

            try:
                chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=100)
            except FoundNaNException:
                continue
            
            node_mask[torch.where(data['fragment_mask'])] = 0

            x = chain[0][:, :, :ddpm.n_dims]
            h = chain[0][:, :, ddpm.n_dims+const.N_RESIDUE_TYPES:]
            
            mol = build_molecules(h, x, node_mask)[0]
            if not metrics.is_valid(mol):
                n_attempts_invalid += 1
                continue
            if not metrics.is_connected(mol):
                n_attempts_unconnected += 1
                continue
            rings = get_sssr_rings(mol)
            if any(len(ring) > 6 for ring in rings):
                n_attempts_large_rings += 1
                continue

            if trajectory is not None:
                record_trajectories(trajectory, imol, chain_batch=chain, node_mask=node_mask)

            names = [f'{name}_size{mol_size}_n{n_samples}_s{random_seed}']
            metadata = {
                'name': name,
                'n_samples': n_samples,
                'mol_size': mol_size,
                'random_seed': random_seed,
                'imol': imol,
            }
            save_xyz_file(output_dir, h, x, node_mask, names=names, metadata=metadata, append=True)

            imol += 1
            bar.update(1)

        tqdm.write(f'Finished sampling for {name}')
        tqdm.write(f'Attempts: {n_attempts}')
        tqdm.write(f'Invalid molecules: {n_attempts_invalid}')
        tqdm.write(f'Unconnected molecules: {n_attempts_unconnected}')
        tqdm.write(f'Large rings: {n_attempts_large_rings}')
        tqdm.write(f'Generated molecules: {imol}')
        tqdm.write(f'Output directory: {output_dir}')
        tqdm.write(f'Duration: {time.time() - start_time}')

        duration = time.time() - start_time
        with open(summary_file, 'a') as f:
            f.write(f'{name},{n_samples},{mol_size},{random_seed},{duration},{n_attempts},{n_attempts_invalid},{n_attempts_unconnected},{n_attempts_large_rings}\n')

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pocket is not None:
        dataset = prepare_single_dataset(args.pocket, device)
    elif args.dataset is not None:
        if '/' in args.dataset:
            data_path, prefix = args.dataset.split('/')
        else:
            data_path = '.'
            prefix = args.dataset
        dataset = ProteinLigandDataset(data_path=data_path, prefix=prefix, device=device)
    else:
        raise ValueError('Either --pocket or --dataset must be provided')

    design_ligands(
        dataset=dataset,
        model=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        mol_size=args.size,
        random_seed=args.random_seed,
        trajectory=args.trajectory,
        device=device
    )
