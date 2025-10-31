import pytest
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python's path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from yuel_design import run_dist_mode


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


def create_dist_matrix(pocket_coords, ligand_coords, discretization_config='b12'):
    """Create distance matrix from pocket and ligand coordinates"""
    import matplotlib.pyplot as plt  # keep matplotlib import localized for tests
    from scipy.spatial.distance import pdist, squareform

    all_coords = np.vstack([pocket_coords, ligand_coords])
    distances = pdist(all_coords)
    dist_matrix = squareform(distances)

    if discretization_config == 'b12':
        dist_matrix = np.clip(np.round(dist_matrix), 0, 11).astype(int)
    return dist_matrix


def plot_comparison_distance_matrices(original_matrix, predicted_matrix, save_path=None):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im1 = ax1.imshow(original_matrix, cmap='viridis', aspect='equal')
    ax1.set_title('Original Distance Matrix')
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Atom Index')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(predicted_matrix, cmap='viridis', aspect='equal')
    ax2.set_title('Predicted Distance Matrix')
    ax2.set_xlabel('Atom Index')
    ax2.set_ylabel('Atom Index')
    plt.colorbar(im2, ax=ax2)

    diff_matrix = np.abs(original_matrix - predicted_matrix)
    im3 = ax3.imshow(diff_matrix, cmap='Reds', aspect='equal')
    ax3.set_title('Absolute Difference')
    ax3.set_xlabel('Atom Index')
    ax3.set_ylabel('Atom Index')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_ligand_atoms_and_coords(mol):
    from src.datasets import get_ligand_atoms_and_coords as _fn
    return _fn(mol)


def parse_pocket(structure):
    from src.datasets import parse_pocket as _fn
    return _fn(structure)


def test_dist_mode(moad_sample, device):
    pocket_id, pocket_pdb, ligand_mol, ligand_name = moad_sample
    print(f"Testing dist mode with MOAD sample: {pocket_id}, ligand: {ligand_name}")

    from rdkit import Chem
    mol = Chem.MolFromMolBlock(ligand_mol)
    ligand_atoms, ligand_coords = get_ligand_atoms_and_coords(mol)
    ligand_size = len(ligand_atoms)
    print(f"Using ligand size: {ligand_size} atoms (excluding H)")

    pocket_structure = parse_pdb_to_structure(pocket_pdb)
    pocket_info = parse_pocket(pocket_structure)

    try:
        gif_path = f"test_outputs/dist_diffusion_{pocket_id}.gif"
        final_prediction, chain, pocket_info = run_dist_mode(
            pocket_structure=pocket_structure,
            ligand_size=ligand_size,
            device=device,
            save_gif=gif_path,
        )

        original_distance_matrix = create_dist_matrix(pocket_info['coords'], ligand_coords, discretization_config='b12')
        predicted_distance_matrix = final_prediction.cpu().numpy()

        import os
        os.makedirs("test_outputs", exist_ok=True)
        plot_comparison_distance_matrices(original_distance_matrix, predicted_distance_matrix, f"test_outputs/dist_comparison_{pocket_id}.png")

        assert final_prediction.shape[0] == final_prediction.shape[1], "Distance matrix should be square"
        assert final_prediction.min() >= 0 and final_prediction.max() <= 11
        assert len(chain) > 0 and pocket_info is not None

    except FileNotFoundError:
        pytest.skip("Model checkpoint not found")
    except Exception as e:
        pytest.fail(f"Dist mode test failed: {e}")


@pytest.mark.parametrize("ligand_size", [10, 20, 30])
def test_dist_mode_different_sizes(moad_sample, device, ligand_size):
    pocket_id, pocket_pdb, ligand_mol, ligand_name = moad_sample
    print(f"Testing dist mode with ligand size {ligand_size} for sample: {pocket_id}")

    pocket_structure = parse_pdb_to_structure(pocket_pdb)

    try:
        gif_path = f"test_outputs/dist_diffusion_{pocket_id}_size_{ligand_size}.gif"
        final_prediction, chain, pocket_info = run_dist_mode(
            pocket_structure=pocket_structure,
            ligand_size=ligand_size,
            device=device,
            save_gif=gif_path,
        )

        expected_size = len(pocket_info['coords']) + ligand_size
        assert final_prediction.shape[0] == expected_size
        assert final_prediction.shape[0] == final_prediction.shape[1]

    except FileNotFoundError:
        pytest.skip("Model checkpoint not found")
    except Exception as e:
        pytest.fail(f"Dist mode test failed for ligand size {ligand_size}: {e}")


@pytest.fixture
def moad_sample():
    try:
        return get_random_moad_sample()
    except Exception as e:
        pytest.skip(f"Could not get MOAD sample: {e}")


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


