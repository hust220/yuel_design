import pytest
import sys
import torch
from pathlib import Path

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from src.coords.app import run_coords_mode, save_coords_pdb, save_coords_trajectory
from src.coords.dataset import (
    get_ligand_atoms_and_coords,
    parse_pocket,
    create_dist_matrix,
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
            if row is None:
                raise ValueError("No samples found in moad_pockets table")
            pocket_id, pocket_pdb, ligand_mol, ligand_name = row
            return pocket_id, pocket_pdb, ligand_mol, ligand_name


def parse_pdb_to_structure(pdb_content: str):
    """Parse PDB content to structure object"""
    from src.pdb_utils import Structure
    from io import StringIO

    structure = Structure()
    structure.read(StringIO(pdb_content))
    return structure


def read_molecule_from_molblock(molblock: str):
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(molblock)
    if mol is not None:
        mol = Chem.RemoveHs(mol)
    return mol


# Use create_dist_matrix from src.coords.dataset for consistency


def test_coords_mode(device):
    while True:
        pocket_id, pocket_pdb, ligand_mol, ligand_name = get_random_moad_sample()
        print(f"Testing coords mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")
        mol = read_molecule_from_molblock(ligand_mol)
        if mol is not None:
            break

    pocket_structure = parse_pdb_to_structure(pocket_pdb)
    pocket_info = parse_pocket(pocket_structure)

    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol)
    ligand_size = len(ligand_atoms)
    print(f"Using ligand size: {ligand_size} atoms (excluding H)")

    dist_matrix = torch.tensor(create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12'))

    final_prediction, chain, pocket_info = run_coords_mode(
        pocket_structure=pocket_structure,
        ligand_size=ligand_size,
        dist_matrix=dist_matrix,
        device=device,
    )

    import os
    output_dir = f"test_outputs/coords_{pocket_id}"
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

    print(f"✓ Coords mode test passed! Generated coordinates with shape: {final_prediction.shape}")
    print(f"  Chain length: {len(chain)}")
    print(f"  Pocket atoms: {len(pocket_info['coords'])}")


@pytest.mark.parametrize("ligand_size", [10, 20, 30])
def test_coords_mode_different_sizes(moad_sample, device, ligand_size):
    pocket_id, pocket_pdb, ligand_mol, ligand_name = moad_sample
    print(f"Testing coords mode with ligand size {ligand_size} for sample: {pocket_id}")

    pocket_structure = parse_pdb_to_structure(pocket_pdb)
    pocket_info = parse_pocket(pocket_structure)
    n_atoms = len(pocket_info['coords']) + ligand_size
    dist_matrix = torch.randint(0, 12, (n_atoms, n_atoms))

    try:
        final_prediction, chain, pocket_info = run_coords_mode(
            pocket_structure=pocket_structure,
            ligand_size=ligand_size,
            dist_matrix=dist_matrix,
            device=device,
        )

        assert final_prediction.shape[0] == n_atoms
        assert final_prediction.shape[1] == 3
        print(f"✓ Ligand size {ligand_size} test passed!")

    except FileNotFoundError:
        pytest.skip("Model checkpoint not found")
    except Exception as e:
        pytest.fail(f"Coords mode test failed for ligand size {ligand_size}: {e}")


@pytest.fixture
def moad_sample():
    try:
        return get_random_moad_sample()
    except Exception as e:
        pytest.skip(f"Could not get MOAD sample: {e}")


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


