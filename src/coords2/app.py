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
from src.coords2.dataset import (
    parse_pocket,
    init_dist_to_coords_features,
    get_atom_one_hot,
)

def features_to_graph(features):
    """Convert features dict to graph (reusing logic from CoordsDataset._to_torch)."""
    import src.gnn as gnn
    from src import const
    
    x = torch.tensor(features['x'], dtype=const.TORCH_FLOAT)
    edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
    
    g = gnn.graph(edge_index, num_nodes=x.shape[0])
    
    g.ndata['x'] = x
    g.ndata['h'] = torch.tensor(features['h'], dtype=const.TORCH_FLOAT)
    
    g.edata['edge_dist'] = torch.tensor(features['edge_dist'], dtype=torch.long)
    g.edata['edge_attr'] = torch.tensor(features['edge_attr'], dtype=torch.long)
    g.edata['ligand_bonds'] = torch.tensor(features['ligand_bonds'], dtype=torch.long)
    
    return g

def load_coords_predictions(
    dist_matrix_file,
    ligand_atoms_file,
    ligand_bonds_file,
):
    """Load coords project predictions from files.
    
    Args:
        dist_matrix_file: Path to distance matrix file (numpy txt format, class indices)
        ligand_atoms_file: Path to ligand atoms file (text file, one atom name per line like 'C', 'N', 'O')
        ligand_bonds_file: Path to ligand bonds file (numpy txt format, class indices)
    
    Returns:
        tuple: (dist_matrix, ligand_atoms, ligand_bonds)
            - dist_matrix: numpy array of distance matrix class indices
            - ligand_atoms: list of ligand atom names (strings)
            - ligand_bonds: numpy array of ligand bonds class indices
    """
    # Load distance matrix (class indices)
    dist_matrix = np.loadtxt(dist_matrix_file, dtype=np.int64)
    
    # Load ligand atoms (one name per line)
    with open(ligand_atoms_file, 'r') as f:
        lines = f.readlines()
    ligand_atoms = [line.strip() for line in lines if line.strip()]
    
    # Load ligand bonds (class indices)
    ligand_bonds = np.loadtxt(ligand_bonds_file, dtype=np.int64)
    
    return dist_matrix, ligand_atoms, ligand_bonds


@torch.no_grad()
def run_coords_mode(
    pocket_structure,
    dist_matrix,
    ligand_atoms,
    ligand_bonds,
    coords_checkpoint: str = None,
    device: torch.device = None,
    seed: int = None,
):
    """High-level coordinate generation using CoordsModel and coords project predictions.
    
    Args:
        pocket_structure: PDB string of protein pocket
        dist_matrix: numpy array of distance matrix class indices (shape: n_ca_ring + ligand_size, n_ca_ring + ligand_size)
        ligand_atoms: list or array of ligand atom names (strings like 'C', 'N', 'O')
        ligand_bonds: numpy array of ligand bonds class indices (shape: ligand_size, ligand_size)
        cont_checkpoint: Path to cont model checkpoint (None for auto-detection)
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
    
    from src.pdb_utils import Structure
    from io import StringIO
    
    # Parse pocket structure
    structure = Structure()
    structure.read(StringIO(pocket_structure))
    pocket_info = parse_pocket(structure)
    
    # Convert inputs to correct types
    dist_matrix = np.asarray(dist_matrix, dtype=np.int64)
    ligand_bonds = np.asarray(ligand_bonds, dtype=np.int64)
    
    # Convert ligand_atoms to list if needed
    if isinstance(ligand_atoms, np.ndarray):
        ligand_atoms = ligand_atoms.tolist()
    ligand_atoms = list(ligand_atoms)
    
    receptor_coords = np.array(pocket_info['coords'])
    protein_size = len(receptor_coords)
    
    # Extract ligand part if ligand_atoms includes protein atoms
    if len(ligand_atoms) > protein_size:
        ligand_atoms = ligand_atoms[protein_size:]
    
    # Convert ligand atom names to cont format (strings with _ prefix)
    # dataset.py expects ligand atoms with '_' prefix (e.g., '_C', '_N', '_O')
    ligand_atoms_str = ['_' + str(atom).strip().lstrip('_') for atom in ligand_atoms]
    ligand_size = len(ligand_atoms_str)
    
    # Store ligand atom names in pocket_info for later use in PDB saving
    pocket_info['ligand_atom_names'] = ligand_atoms
        
    # Extract receptor residues and atoms corresponding to dist_matrix
    # dist_matrix contains CA/ring centers (receptor) + ligand atoms
    atom_names = pocket_info['atom_names']
    res_ids = pocket_info['res_ids']
    res_names = pocket_info['residue_names']
    receptor_residues = []  # Will store residue names (3-letter codes)
    receptor_atoms = []
    for idx, (name, res_id, res_name) in enumerate(zip(atom_names, res_ids, res_names)):
        if name == 'CA' or name.startswith('RING_'):
            receptor_residues.append(res_name)  # Store residue name, not ID
            receptor_atoms.append(name)
    
    # Convert receptor_residues and receptor_atoms to tuple format: [(res_name, [atom_names]), ...]
    receptor_residues_tuples = []
    current_res_name = None
    current_atom_names = []
    
    for res_name, atom_name in zip(receptor_residues, receptor_atoms):
        if current_res_name is None:
            # First residue
            current_res_name = res_name
            current_atom_names = [atom_name]
        elif res_name == current_res_name:
            # Same residue, add to current atom list
            current_atom_names.append(atom_name)
        else:
            # New residue, save previous and start new
            receptor_residues_tuples.append((current_res_name, current_atom_names))
            current_res_name = res_name
            current_atom_names = [atom_name]
    
    # Don't forget the last residue
    if current_res_name is not None:
        receptor_residues_tuples.append((current_res_name, current_atom_names))
    
    # ligand_fixed_atoms corresponds to the ligand atoms in dist_matrix
    ligand_fixed_atoms = ligand_atoms_str
    
    # Build features using disc predictions
    features = init_dist_to_coords_features(
        dist_matrix=dist_matrix,
        receptor_residues=receptor_residues_tuples,
        ligand_fixed_atoms=ligand_fixed_atoms,
        ligand_size=ligand_size,
    )
    
    # Add initial coordinates (use pocket coords + zero for ligand)
    receptor_coords = np.array(pocket_info['coords'])
    ligand_coords = np.zeros((ligand_size, 3))
    features['x'] = np.concatenate([receptor_coords, ligand_coords], axis=0)
    
    graph = features_to_graph(features)
    graph = graph.to(device)
    
    if coords_checkpoint is None:
        coords_checkpoint = pick_latest(['checkpoints/*coords2*/*.ckpt'])
    
    print(f"Loading model from: {coords_checkpoint}")
    model = LightningWrapper.load_from_checkpoint(coords_checkpoint, map_location='cpu')
    model = model.eval()
    model = model.to(device)
    
    final_prediction, chain = model.sample_chain(graph=graph)
    return final_prediction, chain, pocket_info

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_coords_pdb(coords: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB using pocket atom metadata; ligand atoms use placeholders.
    coords: [N,3] tensor predicted by CoordsModel.
    """
    from src.pdb_utils import pdb_line

    _ensure_parent(output_path)
    x = coords.detach().cpu().numpy()

    pocket_atom_names = pocket_info.get('atom_names', [])
    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_res_ids = pocket_info.get('res_ids', [])
    ligand_atom_names = pocket_info.get('ligand_atom_names', [])

    pocket_size = len(pocket_atom_names)
    with open(output_path, 'w') as f:
        f.write("HEADER    COORDS2 PREDICTION\n")
        serial = 1
        # Receptor atoms (skip ring center virtual atoms)
        for i in range(pocket_size):
            atom_name = pocket_atom_names[i] if i < len(pocket_atom_names) else 'CA'
            # Skip ring center virtual atoms (start with 'RING_')
            if atom_name.startswith('RING_'):
                continue
            res_name = pocket_residue_names[i] if i < len(pocket_residue_names) else 'UNK'
            res_seq = (pocket_res_ids[i] if i < len(pocket_res_ids) else i) + 1
            cx, cy, cz = x[i]
            f.write(pdb_line(serial=serial, atom_name=atom_name, res_name=res_name, chain_id='A', res_seq=res_seq, x=cx, y=cy, z=cz))
            serial += 1
        # Ligand atoms
        for j in range(pocket_size, x.shape[0]):
            cx, cy, cz = x[j]
            # Use ligand atom name if available, otherwise use 'C'
            ligand_idx = j - pocket_size
            atom_name = ligand_atom_names[ligand_idx] if ligand_idx < len(ligand_atom_names) else 'C'
            f.write(pdb_line(record='HETATM', serial=serial, atom_name=atom_name, res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
            serial += 1
        f.write("END\n")

def save_coords_trajectory(chain: torch.Tensor, pocket_info: dict, output_path: str):
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
    ligand_atom_names = pocket_info.get('ligand_atom_names', [])
    pocket_size = len(pocket_atom_names)

    open(output_path, 'w').close()
    with open(output_path, 'a') as f:
        for idx, x in enumerate(frames):
            f.write(f"MODEL {idx+1:05d}\n")
            serial = 1
            # Receptor (skip ring center virtual atoms)
            for i in range(min(pocket_size, x.shape[0])):
                atom_name = pocket_atom_names[i] if i < len(pocket_atom_names) else 'CA'
                # Skip ring center virtual atoms (start with 'RING_')
                if atom_name.startswith('RING_'):
                    continue
                res_name = pocket_residue_names[i] if i < len(pocket_residue_names) else 'UNK'
                res_seq = (pocket_res_ids[i] if i < len(pocket_res_ids) else i) + 1
                cx, cy, cz = x[i]
                f.write(pdb_line(serial=serial, atom_name=atom_name, res_name=res_name, chain_id='A', res_seq=res_seq, x=cx, y=cy, z=cz))
                serial += 1
            # Ligand
            for j in range(pocket_size, x.shape[0]):
                cx, cy, cz = x[j]
                # Use ligand atom name if available, otherwise use 'C'
                ligand_idx = j - pocket_size
                atom_name = ligand_atom_names[ligand_idx] if ligand_idx < len(ligand_atom_names) else 'C'
                f.write(pdb_line(record='HETATM', serial=serial, atom_name=atom_name, res_name='LIG', chain_id='B', res_seq=(pocket_res_ids[-1] + 2) if pocket_res_ids else 1, x=cx, y=cy, z=cz))
                serial += 1
            f.write("ENDMDL\n")
