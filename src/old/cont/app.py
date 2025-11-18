import os
import glob
import torch
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.cont.dataset import (
    parse_pocket,
    create_ligand_coords_features,
    ContDataset,
    create_interaction_index,
    get_ligand_atoms_and_coords,
)


@torch.no_grad()
def run_cont_mode(
    pocket_structure,
    int_index,
    ligand_fixed_atoms,
    ligand_size,
    cont_checkpoint: str = None,
    device: torch.device = None,
    seed: int = None,
):
    """High-level coordinate generation for all non-CA atoms using ContModel.
    
    Args:
        pocket_structure: PDB string of protein pocket
        int_index: List of tuples, each tuple is (receptor_full_idx, ligand_full_idx)
            representing interactions between reduced receptor and ligand atoms
        ligand_fixed_atoms: List of ligand reduced atom types (e.g., ['_O', '_N', '_C'])
        ligand_size: Total number of ligand atoms (including C atoms)
        cont_checkpoint: Path to cont model checkpoint (None for auto-detection)
        device: PyTorch device (None for auto-detection)
        seed: Random seed (None for random)
    
    Returns:
        tuple: (final_prediction, chain, pocket_info)
            - final_prediction: [N, 3] tensor of predicted coordinates (all atoms including CA)
            - chain: [T, N, 3] tensor of diffusion chain
            - pocket_info: dict with pocket metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    from src.pdb_utils import Structure
    from io import StringIO
    from src.cont.dataset import init_ligand_coords_features
    
    # Parse pocket structure
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info = parse_pocket(structure)
    
    # Get receptor coordinates and build CA mask
    receptor_full_coords = pocket_info['full_coords']
    receptor_atoms = pocket_info['atoms']
    n_receptor = len(receptor_full_coords)
    
    # Build full receptor atom list to identify CA atoms
    full_receptor_atoms = []
    for reduced_atoms, full_atoms in receptor_atoms:
        full_receptor_atoms.extend(full_atoms)
    
    # Initialize coordinates: CA atoms use true coords, all other atoms use random coords
    full_coords = np.zeros((n_receptor + ligand_size, 3), dtype=np.float32)
    for i in range(n_receptor):
        atom_name = full_receptor_atoms[i]
        if atom_name == 'CA':
            full_coords[i] = receptor_full_coords[i]
        else:
            full_coords[i] = np.random.randn(3)
    
    # Initialize ligand coordinates randomly
    full_coords[n_receptor:] = np.random.randn(ligand_size, 3)
    
    # Initialize features
    features = init_ligand_coords_features(
        int_index=int_index,
        receptor_atoms=receptor_atoms,
        ligand_fixed_atoms=ligand_fixed_atoms,
        ligand_size=ligand_size
    )
    features['x'] = full_coords
    
    # Convert to tensors and move to device
    data = ContDataset.features_to_tensors(features)
    data = {k: v.to(device) for k, v in data.items()}
    
    if cont_checkpoint is None:
        cont_checkpoint = pick_latest(['checkpoints/*cont*/*.ckpt'])
    
    print(f"Loading model from: {cont_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(cont_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_prediction, chain = model.sample_chain(data=data)

    # Build ligand atom names list
    ligand_atom_names = ligand_fixed_atoms + ['_C'] * (ligand_size - len(ligand_fixed_atoms))
    pocket_info['ligand_atom_names'] = ligand_atom_names
    
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
    ligand_atom_names = pocket_info.get('ligand_atom_names', [])
    pocket_size = len(pocket_atom_names)
    
    serial = start_serial
    
    # Receptor atoms (skip ring center virtual atoms)
    for i in range(min(pocket_size, x.shape[0])):
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
    
    # Ligand atoms
    for j in range(pocket_size, x.shape[0]):
        cx, cy, cz = x[j]
        # Use ligand atom name if available, otherwise use 'C'
        ligand_idx = j - pocket_size
        atom_name = ligand_atom_names[ligand_idx] if ligand_idx < len(ligand_atom_names) else 'C'
        if atom_name.startswith('_'):
            atom_name = atom_name[1:]
        f.write(pdb_line(record='HETATM', serial=serial, atom_name=atom_name, res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
        serial += 1
    
    return serial

def save_structure_pdb(coords: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB including all atoms (CA fixed, non-CA predicted).
    
    Args:
        coords: [N,3] tensor with all atom coordinates (CA + predicted non-CA + ligand)
        pocket_info: dict with pocket metadata
        output_path: Path to output PDB file
    """
    _ensure_parent(output_path)
    x = coords.detach().cpu().numpy()
    
    with open(output_path, 'w') as f:
        f.write("HEADER    CONTINUOUS COORDINATE PREDICTION\n")
        _save_model(x, pocket_info, f)
        f.write("END\n")

def save_trajectory(chain: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinate trajectory (frames,N,3) into a multi-model PDB file.
    Includes all atoms: CA (fixed) + non-CA atoms (predicted) + ligand (predicted).
    """
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
