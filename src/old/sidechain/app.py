import os
import torch
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.sidechain.dataset import (
    parse_pocket,
    init_receptor_features,
    expand_dist_matrix,
    SidechainDataset,
)

@torch.no_grad()
def run_sidechain_mode(
    pocket_structure,
    dist_matrix,
    sidechain_checkpoint: str = None,
    device: torch.device = None,
    seed: int = None,
):
    """High-level sidechain coordinate generation using SidechainModel.
    
    Args:
        pocket_structure: PDB string of protein structure
        dist_matrix: Distance matrix with shape (n_reduced_receptor, n_reduced_receptor)
            for reduced receptor atoms (CA + non-C + ring centers)
        sidechain_checkpoint: Path to sidechain model checkpoint (None for auto-detection)
        device: PyTorch device (None for auto-detection)
        seed: Random seed (None for random)
    
    Returns:
        tuple: (final_prediction, chain, pocket_info)
            - final_prediction: [N, 3] tensor of predicted coordinates
            - chain: [T, N, 3] tensor of diffusion chain
            - pocket_info: dict with pocket metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Parse pocket structure
    from src.pdb_utils import Structure
    from io import StringIO
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info = parse_pocket(structure)
    if pocket_info is None:
        raise ValueError("Failed to parse pocket structure")
    
    # Get receptor atoms and coordinates
    receptor_atoms = pocket_info['atoms']
    receptor_reduced_coords = pocket_info['reduced_coords']
    receptor_full_coords = pocket_info['full_coords']
    
    # Validate dist_matrix shape
    n_reduced_receptor = len(receptor_reduced_coords)
    if dist_matrix.shape[0] != n_reduced_receptor or dist_matrix.shape[1] != n_reduced_receptor:
        raise ValueError(
            f"dist_matrix shape {dist_matrix.shape} does not match "
            f"n_reduced_receptor ({n_reduced_receptor})"
        )
    
    # Expand distance matrix to get full_receptor_atoms list (with correct order)
    full_dist_matrix, full_receptor_atoms = expand_dist_matrix(dist_matrix, receptor_atoms)
    
    # Initialize features from distance matrix
    features = init_receptor_features(
        dist_matrix=dist_matrix,
        receptor_atoms=receptor_atoms
    )
    
    # Initialize x: CA atoms use original coordinates, others use random coordinates
    # full_receptor_atoms and receptor_full_coords should have the same order
    # since both are built by iterating receptor_atoms in the same way
    n_receptor = len(full_receptor_atoms)
    assert n_receptor == len(receptor_full_coords), \
        f"Length mismatch: full_receptor_atoms ({n_receptor}) != receptor_full_coords ({len(receptor_full_coords)})"
    
    x = np.random.randn(n_receptor, 3).astype(np.float32)
    
    # Set CA atoms to original coordinates
    # Since full_receptor_atoms is built in the same order as receptor_full_coords,
    # we can directly map by index
    for i, atom_name in enumerate(full_receptor_atoms):
        if atom_name == 'CA':
            x[i] = receptor_full_coords[i]

    features['x'] = x
    
    graph = SidechainDataset.features_to_graph(features)
    graph = graph.to(device)
    
    if sidechain_checkpoint is None:
        sidechain_checkpoint = pick_latest(['checkpoints/*sidechain*/*.ckpt'])
    
    print(f"Loading model from: {sidechain_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(sidechain_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_prediction, chain = model.sample_chain(graph=graph)
    
    return final_prediction, chain, pocket_info

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _extract_pocket_metadata(pocket_info):
    """Extract atom names, residue names, residue IDs, and chain IDs from pocket_info.
    
    Args:
        pocket_info: dict with 'atoms' and 'residues' keys
    
    Returns:
        tuple: (atom_names, residue_names, res_ids, chain_ids)
    """
    atoms = pocket_info.get('atoms', [])
    residues = pocket_info.get('residues', [])
    
    atom_names = []
    residue_names = []
    res_ids = []
    chain_ids = []
    
    for (reduced_atoms, full_atoms), (res_name, res_id, chain_id) in zip(atoms, residues):
        for atom_name in full_atoms:
            atom_names.append(atom_name)
            residue_names.append(res_name)
            res_ids.append(res_id)
            chain_ids.append(chain_id)
    
    return atom_names, residue_names, res_ids, chain_ids

def _save_model(x: np.ndarray, pocket_info: dict, f, start_serial: int = 1):
    """Save a single model to PDB file.
    
    Args:
        x: numpy array of shape [N, 3] with coordinates
        pocket_info: dict with pocket metadata
        f: file handle to write to
        start_serial: starting serial number for atoms
    
    Returns:
        int: next serial number to use
    """
    from src.pdb_utils import pdb_line
    
    pocket_atom_names, pocket_residue_names, pocket_res_ids, pocket_chain_ids = _extract_pocket_metadata(pocket_info)
    
    serial = start_serial
    
    # Receptor atoms (skip ring center virtual atoms)
    for i in range(min(len(pocket_atom_names), x.shape[0])):
        atom_name = pocket_atom_names[i] if i < len(pocket_atom_names) else 'CA'
        # Skip ring center virtual atoms (start with 'RING_')
        if atom_name.startswith('RING_'):
            continue
        res_name = pocket_residue_names[i] if i < len(pocket_residue_names) else 'UNK'
        res_seq = (pocket_res_ids[i] if i < len(pocket_res_ids) else i) + 1
        chain_id = pocket_chain_ids[i] if i < len(pocket_chain_ids) else 'A'
        cx, cy, cz = x[i]
        f.write(pdb_line(serial=serial, atom_name=atom_name, res_name=res_name, chain_id=chain_id, res_seq=res_seq, x=cx, y=cy, z=cz))
        serial += 1
    
    return serial

def save_sidechain_pdb(coords: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB using pocket atom metadata.
    coords: [N,3] tensor predicted by SidechainModel.
    """
    _ensure_parent(output_path)
    x = coords.detach().cpu().numpy()
    
    with open(output_path, 'w') as f:
        f.write("HEADER    SIDECHAIN PREDICTION\n")
        _save_model(x, pocket_info, f)
        f.write("END\n")

def save_sidechain_trajectory(chain: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinate trajectory (frames,N,3) into a multi-model PDB file."""
    _ensure_parent(output_path)
    
    # Accept list or tensor
    if isinstance(chain, list):
        frames = [c.detach().cpu().numpy() for c in chain]
    else:
        arr = chain.detach().cpu().numpy()
        # If shape is (T,N,3)
        if arr.ndim == 3:
            frames = [arr[t] for t in range(arr.shape[0])]
        else:
            frames = [arr]
    
    with open(output_path, 'w') as f:
        for idx, x in enumerate(frames):
            f.write(f"MODEL {idx+1:05d}\n")
            _save_model(x, pocket_info, f)
            f.write("ENDMDL\n")
