import pytest
import sys
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from src.cont.app import run_cont_mode, save_coords_pdb, save_coords_trajectory, load_disc_predictions
from src.cont.dataset import (
    get_ligand_atoms_and_coords,
    parse_pocket,
    create_dist_matrix,
    get_ligand_bond_type,
    LIGAND_BOND_TYPE2IDX,
)

def get_random_moad_sample():
    """Get a random sample from moad_pockets table"""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT mp.id, mp.pdb, ml.mol, ml.name
                FROM moad_pockets mp
                JOIN moad_ligands ml ON mp.ligand_name = ml.name
                WHERE mp.split = 'train'
                ORDER BY RANDOM()
                LIMIT 1
                """
            )
            row = c.fetchone()
            pocket_id, pocket_pdb, ligand_mol, ligand_name = row
            return pocket_id, pocket_pdb, ligand_mol, ligand_name


def read_molecule_from_molblock(molblock: str):
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(molblock)
    return Chem.RemoveHs(mol) if mol is not None else None


def create_disc_predictions_from_mol(mol, pocket_info):
    """Create disc project predictions from molecule and pocket info.
    
    This simulates what disc project would output:
    - dist_matrix: (n_ca_ring + ligand_size, n_ca_ring + ligand_size) - distance class indices
    - ligand_atoms: list of atom names (strings like 'C', 'N', 'O')
    - ligand_bonds: (ligand_size, ligand_size) - bond class indices
    """
    ligand_atoms_cont, ligand_coords, ligand_bond_matrix = get_ligand_atoms_and_coords(mol)
    ligand_atoms_names = [atom.replace('_', '') for atom in ligand_atoms_cont]
    ligand_size = len(ligand_atoms_names)
    
    # Extract CA and ring center coordinates from pocket (same logic as dataset.py)
    atom_names = pocket_info['atom_names']
    ca_or_ring_coords = np.array([
        coord for coord, name in zip(pocket_info['coords'], atom_names)
        if name == 'CA' or name.startswith('RING_')
    ])
    
    dist_matrix = create_dist_matrix(ca_or_ring_coords, np.array(ligand_coords), discretization_config='b12')
    
    return dist_matrix, ligand_atoms_names, ligand_bond_matrix


def test_cont_mode(device):
    """Test cont mode with disc project predictions"""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing cont mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break

    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info = parse_pocket(structure)
    
    # Create disc project predictions
    dist_matrix, ligand_atoms_names, ligand_bonds = create_disc_predictions_from_mol(mol, pocket_info)
    
    final_prediction, chain, pocket_info = run_cont_mode(
        pocket_structure=pocket_pdb,
        dist_matrix=dist_matrix,
        ligand_atoms=ligand_atoms_names,
        ligand_bonds=ligand_bonds,
        device=device,
    )

    import os
    output_dir = f"test_outputs/cont_{pocket_id}"
    os.makedirs(output_dir, exist_ok=True)

    original_pdb_path = os.path.join(output_dir, "original_receptor.pdb")
    with open(original_pdb_path, 'w') as f:
        f.write(pocket_pdb)

    ligand_sdf_path = os.path.join(output_dir, "original_ligand.sdf")
    with open(ligand_sdf_path, 'w') as f:
        f.write(ligand_mol)

    predicted_pdb_path = os.path.join(output_dir, "predicted.pdb")
    print(f"Saving predicted coordinates to {predicted_pdb_path}")
    save_coords_pdb(final_prediction, pocket_info, predicted_pdb_path)

    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    print(f"Saving trajectory to {trajectory_path}")
    save_coords_trajectory(chain, pocket_info, trajectory_path)

    print(f"✓ Cont mode test passed! Generated coordinates with shape: {final_prediction.shape}")
    print(f"  Chain length: {len(chain)}")
    print(f"  Pocket atoms: {len(pocket_info['coords'])}")


def test_cont_mode_with_files(device):
    """Test cont mode with disc project prediction files"""
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing cont mode with files: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol:
            break

    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_pdb))
    pocket_info = parse_pocket(structure)
    
    # Create disc project predictions
    dist_matrix, ligand_atoms_names, ligand_bonds = create_disc_predictions_from_mol(mol, pocket_info)
    
    # Create temporary files to simulate disc project output
    with tempfile.TemporaryDirectory() as tmpdir:
        dist_matrix_file = os.path.join(tmpdir, "dist_matrix.txt")
        ligand_atoms_file = os.path.join(tmpdir, "ligand_atoms.txt")
        ligand_bonds_file = os.path.join(tmpdir, "ligand_bonds.txt")
        
        # Save distance matrix (class indices)
        np.savetxt(dist_matrix_file, dist_matrix, fmt='%d')
        
        # Save ligand atoms (one name per line)
        with open(ligand_atoms_file, 'w') as f:
            for atom_name in ligand_atoms_names:
                f.write(f"{atom_name}\n")
        
        # Save ligand bonds (class indices)
        np.savetxt(ligand_bonds_file, ligand_bonds, fmt='%d')
        
        dist_matrix, ligand_atoms, ligand_bonds = load_disc_predictions(
            dist_matrix_file,
            ligand_atoms_file,
            ligand_bonds_file,
        )
        
        final_prediction, chain, pocket_info = run_cont_mode(
            pocket_structure=pocket_pdb,
            dist_matrix=dist_matrix,
            ligand_atoms=ligand_atoms,
            ligand_bonds=ligand_bonds,
            device=device,
        )
        
        assert final_prediction.shape[0] == len(pocket_info['coords']) + len(ligand_atoms_names)
        assert final_prediction.shape[1] == 3
        print(f"✓ Cont mode with files test passed! Generated coordinates with shape: {final_prediction.shape}")


def test_cont_mode_with_dataset(device):
    """Test using ContDataset to load data and ContModel to generate coordinates"""
    from src.cont.dataset import ContDataset
    from src.cont.model import ContModel
    from src.lightning1 import LightningWrapper
    from src.utils import pick_latest
    import src.gnn as gnn
    
    dataset = ContDataset(split='train')
    graph = dataset[0]
    graph = graph.to(device)
    
    cont_checkpoint = pick_latest(['checkpoints/*cont*/*.ckpt'])
    model = LightningWrapper.load_from_checkpoint(cont_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        final_prediction, chain = model.sample_chain(graph=graph)
    
    assert final_prediction.shape[0] == graph.ndata['x'].shape[0]
    assert final_prediction.shape[1] == 3
    print(f"✓ Cont mode with dataset test passed! Generated coordinates with shape: {final_prediction.shape}")


@pytest.fixture
def moad_sample():
    return get_random_moad_sample()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
