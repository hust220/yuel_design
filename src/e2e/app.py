import os
import glob
import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import pick_latest
from src.lightning1 import LightningWrapper
from src.e2e.dataset import parse_pocket, init_dist_to_coords_features
from src.e2e.model import E2EModel
from src.e2e import config as e2e_config

def _build_graph_from_features(features):
    """Create gnn graph compatible with E2EModel from precomputed features."""
    import src.gnn as gnn
    edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
    g = gnn.graph(edge_index, num_nodes=len(features['h']))
    g.ndata['x'] = torch.tensor(features['x'], dtype=torch.float32)
    g.ndata['h'] = torch.tensor(features['h'], dtype=torch.float32)
    g.ndata['anchor_mask'] = torch.tensor(features['anchor_mask'], dtype=torch.float32)
    g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
    return g

@torch.no_grad()
def run_e2e_mode(pocket_structure, ligand_size: int, device: torch.device = None, seed: int = None, checkpoint: str = None):
    """High-level end-to-end coordinate generation using E2EModel and dataset utils."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse pocket and build features/graph
    pocket_info = parse_pocket(pocket_structure)
    features = init_dist_to_coords_features(pocket_info, ligand_size)
    # Build initial coordinates: use pocket coords + zero-initialized ligand coords
    import numpy as np
    receptor_coords = np.array(pocket_info['coords'], dtype=float)
    ligand_zeros = np.zeros((ligand_size, 3), dtype=float)
    features['x'] = np.concatenate([receptor_coords, ligand_zeros], axis=0)
    graph = _build_graph_from_features(features)
    graph = graph.to(device)

    ckpt = checkpoint or pick_latest(['checkpoints/*e2e*/*.ckpt'])
    model = LightningWrapper.load_from_checkpoint(ckpt, map_location='cpu')
    model = model.eval()
    model = model.to(device)

    # Sample coordinates chain
    final_prediction, chain = model.sample_chain(graph=graph)
    return final_prediction, chain, pocket_info

def _ensure_parent(path: str):
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_e2e_pdb(coords: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB using pocket atom metadata; ligand atoms use placeholders.
    coords: [N,3] tensor predicted by CoordsModel.
    """
    from src.pdb_utils import pdb_line

    _ensure_parent(output_path)
    x = coords.detach().cpu().numpy()

    pocket_atom_names = pocket_info.get('atom_names', [])
    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_res_ids = pocket_info.get('res_ids', [])

    pocket_size = len(pocket_atom_names)
    with open(output_path, 'w') as f:
        f.write("HEADER    COORDS PREDICTION\n")
        serial = 1
        # Receptor atoms
        for i in range(pocket_size):
            atom_name = pocket_atom_names[i] if i < len(pocket_atom_names) else 'CA'
            res_name = pocket_residue_names[i] if i < len(pocket_residue_names) else 'UNK'
            res_seq = (pocket_res_ids[i] if i < len(pocket_res_ids) else i) + 1
            cx, cy, cz = x[i]
            f.write(pdb_line(serial=serial, atom_name=atom_name, res_name=res_name, chain_id='A', res_seq=res_seq, x=cx, y=cy, z=cz))
            serial += 1
        # Ligand atoms (placeholder names)
        for j in range(pocket_size, x.shape[0]):
            cx, cy, cz = x[j]
            f.write(pdb_line(record='HETATM', serial=serial, atom_name='C', res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
            serial += 1
        f.write("END\n")

def save_e2e_trajectory(chain: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinate trajectory (frames,N,3) into a multi-model PDB file."""
    from src.pdb_utils import pdb_line

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

    pocket_atom_names = pocket_info.get('atom_names', [])
    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_res_ids = pocket_info.get('res_ids', [])
    pocket_size = len(pocket_atom_names)

    open(output_path, 'w').close()
    with open(output_path, 'a') as f:
        for idx, x in enumerate(frames):
            f.write(f"MODEL {idx+1:05d}\n")
            serial = 1
            # Receptor
            for i in range(min(pocket_size, x.shape[0])):
                atom_name = pocket_atom_names[i] if i < len(pocket_atom_names) else 'CA'
                res_name = pocket_residue_names[i] if i < len(pocket_residue_names) else 'UNK'
                res_seq = (pocket_res_ids[i] if i < len(pocket_res_ids) else i) + 1
                cx, cy, cz = x[i]
                f.write(pdb_line(serial=serial, atom_name=atom_name, res_name=res_name, chain_id='A', res_seq=res_seq, x=cx, y=cy, z=cz))
                serial += 1
            # Ligand
            for j in range(pocket_size, x.shape[0]):
                cx, cy, cz = x[j]
                f.write(pdb_line(record='HETATM', serial=serial, atom_name='C', res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
                serial += 1
            f.write("ENDMDL\n")


