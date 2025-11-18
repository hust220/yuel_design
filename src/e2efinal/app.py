import os
import torch
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.e2efinal.dataset import (
    parse_pocket,
    create_ligand_coords_features,
    E2EDataset,
    init_ligand_coords_features,
    LIGAND_ATOM_TYPES,
    LIGAND_ATOM2IDX,
)
from src.pdb_utils import Structure

@torch.no_grad()
def run_e2e_mode(
    pocket_structure,
    ligand_size: int = None,
    e2e_checkpoint: str = None,
    device: torch.device = None,
    seed: int = None,
):
    """High-level end-to-end ligand generation using E2EModel.
    Generates both ligand coordinates and atom types.
    
    Args:
        pocket_structure: PDB string of protein pocket
        ligand_size: Number of ligand atoms (if None, will be estimated from receptor)
        e2e_checkpoint: Path to e2e model checkpoint (None for auto-detection)
        device: PyTorch device (None for auto-detection)
        seed: Random seed (None for random)
    
    Returns:
        tuple: (final_coords, final_atoms, chain, pocket_info)
            - final_coords: [N, 3] tensor of predicted coordinates (all atoms)
            - final_atoms: [N] tensor of predicted atom type indices for ligand atoms
            - chain: [T, N, 3] tensor of diffusion chain (coordinates only)
            - pocket_info: dict with pocket metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    from io import StringIO
    
    # Parse pocket structure
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info = parse_pocket(structure)
    
    # Get receptor coordinates
    receptor_full_coords = pocket_info['full_coords']
    receptor_atoms = pocket_info['atoms']
    n_receptor = len(receptor_full_coords)
    
    # Estimate ligand size if not provided
    if ligand_size is None:
        # Default to reasonable size based on receptor
        ligand_size = min(50, max(10, n_receptor // 10))
    
    # Build initial coordinates: use receptor coords + random ligand coords
    full_coords = np.zeros((n_receptor + ligand_size, 3), dtype=np.float32)
    full_coords[:n_receptor] = receptor_full_coords
    full_coords[n_receptor:] = np.random.randn(ligand_size, 3)
    
    # Initialize features using e2e dataset utilities
    features = init_ligand_coords_features(
        receptor_atoms=receptor_atoms,
        ligand_size=ligand_size,
        full_coords=full_coords
    )
    features['x'] = full_coords
    
    # Convert to tensors and add batch dimension
    # Note: ligand_atoms and receptor_interaction are optional and not needed for inference
    data = E2EDataset.features_to_tensors(features)
    data = {k: v.unsqueeze(0).to(device) for k, v in data.items()}
    
    if e2e_checkpoint is None:
        e2e_checkpoint = pick_latest(['checkpoints/*e2efinal*/*.ckpt'])
    
    print(f"Loading model from: {e2e_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(e2e_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_result, chain = model.sample_chain(data=data)
    
    # Extract final coordinates and atoms from result dictionary
    final_coords = final_result['coords']  # [B, N, 3]
    final_atoms = final_result['atoms']  # [B, N]
    
    # Handle batch dimension (if batch_size > 1, take first sample)
    if final_coords.dim() == 3 and final_coords.shape[0] > 1:
        final_coords = final_coords[0]  # [N, 3]
        final_atoms = final_atoms[0]  # [N]
    elif final_coords.dim() == 3:
        final_coords = final_coords.squeeze(0)  # [N, 3]
        final_atoms = final_atoms.squeeze(0)  # [N]
    
    # Mask out receptor atoms (set to 0)
    receptor_mask = data['receptor_mask']
    if receptor_mask.dim() == 2 and receptor_mask.shape[0] == 1:
        receptor_mask = receptor_mask.squeeze(0)  # [N]
    final_atoms = final_atoms * (1 - receptor_mask.long())
    
    # Store ligand atom names in pocket_info for saving
    ligand_atom_names = []
    for i in range(n_receptor, len(final_atoms)):
        atom_idx = final_atoms[i].item()
        if atom_idx < len(LIGAND_ATOM_TYPES):
            atom_name = LIGAND_ATOM_TYPES[atom_idx]
            # Remove underscore prefix if present
            if atom_name.startswith('_'):
                atom_name = atom_name[1:]
            elif atom_name == 'X':
                atom_name = 'C'  # Default to C for unknown
            ligand_atom_names.append(atom_name)
        else:
            ligand_atom_names.append('C')  # Default fallback
    
    pocket_info['ligand_atom_names'] = ligand_atom_names
    
    return final_coords, final_atoms, chain, pocket_info

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

def _save_model(x: np.ndarray, pocket_info: dict, ligand_atoms: np.ndarray, f, start_serial: int = 1):
    """Save a single model to PDB file.
    
    Args:
        x: numpy array of shape [N, 3] with coordinates
        pocket_info: dict with pocket metadata
        ligand_atoms: numpy array of shape [N] with atom type indices (0 for receptor)
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
        # Use ligand atom name if available
        ligand_idx = j - pocket_size
        if ligand_idx < len(ligand_atom_names):
            atom_name = ligand_atom_names[ligand_idx]
        else:
            # Fallback: convert atom type index to name
            atom_idx = ligand_atoms[j].item() if j < len(ligand_atoms) else 0
            if atom_idx < len(LIGAND_ATOM_TYPES):
                atom_name = LIGAND_ATOM_TYPES[atom_idx]
                if atom_name.startswith('_'):
                    atom_name = atom_name[1:]
                elif atom_name == 'X':
                    atom_name = 'C'
            else:
                atom_name = 'C'
        
        f.write(pdb_line(record='HETATM', serial=serial, atom_name=atom_name, res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
        serial += 1
    
    return serial

def save_structure_pdb(coords: torch.Tensor, ligand_atoms: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB including all atoms (receptor + predicted ligand).
    
    Args:
        coords: [N,3] tensor with all atom coordinates (receptor + ligand)
        ligand_atoms: [N] tensor with atom type indices (0 for receptor atoms)
        pocket_info: dict with pocket metadata
        output_path: Path to output PDB file
    """
    _ensure_parent(output_path)
    x = coords.detach().cpu().numpy()
    atoms = ligand_atoms.detach().cpu().numpy() if isinstance(ligand_atoms, torch.Tensor) else ligand_atoms
    
    with open(output_path, 'w') as f:
        f.write("HEADER    END-TO-END LIGAND GENERATION\n")
        _save_model(x, pocket_info, atoms, f)
        f.write("END\n")

def save_trajectory(chain, ligand_atoms: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinate trajectory into a multi-model PDB file.
    Includes all atoms: receptor (fixed) + ligand (predicted).
    
    Args:
        chain: List of dicts with 'coords' and 'atoms' keys, or tensor with shape (T, N, 3)
        ligand_atoms: [N] tensor with atom type indices (0 for receptor atoms)
        pocket_info: dict with pocket metadata
        output_path: Path to output PDB file
    """
    _ensure_parent(output_path)
    
    # Extract frames from chain (list of dicts with 'coords' and 'atoms')
    if isinstance(chain, list) and len(chain) > 0 and isinstance(chain[0], dict):
        frames = []
        atoms_list = []
        for step_result in chain:
            coords = step_result['coords']  # [B, N, 3] or [N, 3]
            atoms = step_result['atoms']  # [B, N] or [N]
            # Handle batch dimension
            if coords.dim() == 3:
                coords = coords[0] if coords.shape[0] > 1 else coords.squeeze(0)
            if atoms.dim() == 2:
                atoms = atoms[0] if atoms.shape[0] > 1 else atoms.squeeze(0)
            frames.append(coords.detach().cpu().numpy())
            atoms_list.append(atoms.detach().cpu().numpy())
        # Use final atoms for all frames (or could use per-frame atoms)
        atoms = atoms_list[-1] if atoms_list else ligand_atoms.detach().cpu().numpy()
    elif isinstance(chain, list):
        frames = [c.detach().cpu().numpy() for c in chain]
        atoms = ligand_atoms.detach().cpu().numpy() if isinstance(ligand_atoms, torch.Tensor) else ligand_atoms
    else:
        arr = chain.detach().cpu().numpy()
        # If shape is (T,N,3)
        if arr.ndim == 3:
            frames = [arr[t] for t in range(arr.shape[0])]
        else:
            frames = [arr]
        atoms = ligand_atoms.detach().cpu().numpy() if isinstance(ligand_atoms, torch.Tensor) else ligand_atoms
    
    with open(output_path, 'w') as f:
        for idx, x in enumerate(frames):
            f.write(f"MODEL {idx+1:05d}\n")
            _save_model(x, pocket_info, atoms, f)
            f.write("ENDMDL\n")
