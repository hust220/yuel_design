import argparse
import os
import random
import torch
from pathlib import Path

# Yuel Design imports
from src.datasets import DistDataset, CoordsDataset, DATASET_CONFIGS, init_dist_features, init_dist_to_coords_features
from src.lightning1 import YuelDesign
from src.DistD3PM import D3PM as DistD3PM
from src.CoordsEDM import EDM as EDM2
from src.console import section, info, success, warn, error, spinner
from src.utils import pick_latest


def parse_args():
    p = argparse.ArgumentParser(description='Yuel Design: Protein-Ligand Design Pipeline')

    # Pipeline stages
    p.add_argument('--pipeline', type=str, default='dist:coords', help='Pipeline stages: dist, coords, or dist:coords')

    
    # Model checkpoints
    p.add_argument('--dist_checkpoint', type=str, default=None, help='Checkpoint path for distance prediction model')
    p.add_argument('--coords_checkpoint', type=str, default=None, help='Checkpoint path for coordinate prediction model')

    # Input/Output
    p.add_argument('--input_pdb', type=str, required=True, help='Input protein PDB file')
    p.add_argument('--input_ligand', type=str, default=None, help='Input ligand SMILES string')
    p.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    # Save controls
    p.add_argument('--save_dist', type=str, default=None, help='Save distance matrix as PNG image')
    p.add_argument('--save_coords_pdb', type=str, default='output/coords.pdb', help='Save predicted coordinates as PDB')
    p.add_argument('--save_trajectory', type=str, default=None, help='Save generation trajectory')
    p.add_argument('--save_coords', type=str, default=None, help='Save coordinates as PDB file')
    p.add_argument('--save_traj', type=str, default=None, help='Save coordinate trajectory as PDB files')

    # Generation parameters
    p.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate')
    p.add_argument('--ligand_size', type=int, required=True, help='Number of atoms in the ligand to generate')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    p.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'], help='Device to use')

    return p.parse_args()


def parse_pipeline(pipeline_str: str) -> list:
    """Parse pipeline string like 'dist:coords' or 'dist' into list of stages"""
    # Stage name mappings
    stage_mapping = {
        'dist': 'dist', 
        'distance': 'dist',
        'coords': 'coords',
        'coordinates': 'coords',
    }
    
    if ':' in pipeline_str:
        stages = pipeline_str.split(':')
    else:
        stages = [pipeline_str]
    
    parsed_stages = []
    for stage in stages:
        parsed_stage = stage_mapping.get(stage.strip().lower())
        if parsed_stage is None:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {list(stage_mapping.keys())}")
        parsed_stages.append(parsed_stage)
    
    return parsed_stages


def pick_device(user_choice: str) -> torch.device:
    if user_choice == 'cpu':
        return torch.device('cpu')
    if user_choice == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_parent(path: str):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_model(model_class, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    # Use pick_latest to find checkpoint
    actual_checkpoint_path = pick_latest([checkpoint_path])
    info(f"Found checkpoint: {actual_checkpoint_path}")

    # Load model using YuelDesign wrapper
    from src.lightning1 import YuelDesign
    model = YuelDesign.load_from_checkpoint(actual_checkpoint_path, map_location='cpu').eval()
    if device.type == 'cuda':
        model = model.to(device)
    
    info(f"Loaded model from {actual_checkpoint_path}")
    return model


def parse_pocket_from_pdb(pdb_content: str):
    """Parse pocket info from PDB content"""
    from src.pdb_utils import Structure
    from src.datasets import parse_pocket
    from io import StringIO
    
    structure = Structure()
    structure.read(StringIO(pdb_content))
    pocket_info = parse_pocket(structure)
    return pocket_info


def create_dist_features_data(pocket_info, ligand_size):
    """Create distance features data for model input"""
    features = init_dist_features(pocket_info, ligand_size)
    
    batch_size = 1
    seq_length = features['seq'].shape[0]
    
    data = {
        'seq': torch.tensor(features['seq'], dtype=torch.float32).unsqueeze(0),
        'z': torch.tensor(features['z'], dtype=torch.float32).unsqueeze(0),
        'dist': torch.randint(0, 12, (batch_size, seq_length, seq_length)),
        'seq_mask': torch.tensor(features['seq_mask'], dtype=torch.float32).unsqueeze(0),
        'pair_mask': torch.tensor(features['pair_mask'], dtype=torch.float32).unsqueeze(0)
    }
    
    return data, features


def create_coords_features_data(pocket_info, ligand_size, dist_matrix):
    """Create coordinate features data for model input"""
    features = init_dist_to_coords_features(pocket_info, ligand_size, dist_matrix.cpu().numpy())
    
    from src import gnn
    edge_index = torch.tensor(features['edge_index'], dtype=torch.long)
    graph = gnn.Graph(edge_index, num_nodes=features['positions'].shape[0])
    
    # Add node data
    graph.add_node_data('positions', torch.tensor(features['positions'], dtype=torch.float32))
    graph.add_node_data('one_hot', torch.tensor(features['one_hot'], dtype=torch.float32))
    graph.add_node_data('node_attr', torch.tensor(features['node_attr'], dtype=torch.float32))
    graph.add_node_data('node_mask', torch.tensor(features['node_mask'], dtype=torch.float32))
    graph.add_node_data('anchor_mask', torch.tensor(features['anchor_mask'], dtype=torch.float32))
    
    graph.add_edge_data('edge_dist', torch.tensor(features['edge_dist'], dtype=torch.long))
    graph.add_edge_data('edge_residue', torch.tensor(features['edge_residue'], dtype=torch.long))
    graph.add_edge_data('edge_mask', torch.tensor(features['edge_mask'], dtype=torch.float32))
    
    return graph, features


def run_dist_mode(pocket_structure, ligand_size: int, dist_checkpoint: str = None, device: torch.device = None, seed: int = None, save_gif: str = None):
    """
    Run distance prediction mode
    
    Args:
        pocket_structure: Parsed pocket structure object
        ligand_size: Number of atoms in the ligand to generate
        dist_checkpoint: Path to distance model checkpoint (optional, will use default pattern)
        device: Device to use (optional, will auto-detect)
        seed: Random seed (optional)
        save_gif: Path to save GIF animation of diffusion process (optional)
    
    Returns:
        tuple: (final_prediction, chain, pocket_info)
            - final_prediction: Final distance matrix (torch.Tensor)
            - chain: Full diffusion chain (list of torch.Tensor)
            - pocket_info: Parsed pocket information (dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Load distance prediction model
    if dist_checkpoint is None:
        dist_checkpoint = "models/*dist*/*.ckpt"
    dist_model = load_model(DistD3PM, dist_checkpoint, device)
    
    # Parse pocket info from structure
    from src.datasets import parse_pocket
    pocket_info = parse_pocket(pocket_structure)
    
    # Create distance features data
    data, features = create_dist_features_data(pocket_info, ligand_size)
    
    # Move data to the same device as model
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    
    # Generate distance matrix using the model
    with torch.no_grad():
        final_pred, chain = dist_model.sample_chain(data=data)
        final_pred = final_pred.view(features['seq'].shape[0], features['seq'].shape[0])
    
    # Save chain as GIF if requested
    if save_gif:
        save_chain_as_gif(chain, save_gif, title="Distance Matrix Diffusion Process")
    
    return final_pred, chain, pocket_info


def run_coords_mode(pocket_structure, ligand_size: int, dist_matrix: torch.Tensor, coords_checkpoint: str = None, device: torch.device = None, seed: int = None):
    """
    Run coordinate prediction mode
    
    Args:
        pocket_structure: Parsed pocket structure object
        ligand_size: Number of atoms in the ligand to generate
        dist_matrix: Distance matrix from dist mode (torch.Tensor)
        coords_checkpoint: Path to coordinate model checkpoint (optional, will use default pattern)
        device: Device to use (optional, will auto-detect)
        seed: Random seed (optional)
    
    Returns:
        tuple: (final_prediction, chain, pocket_info)
            - final_prediction: Final coordinates (torch.Tensor)
            - chain: Full diffusion chain (list of torch.Tensor)
            - pocket_info: Parsed pocket information (dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Load coordinate prediction model
    if coords_checkpoint is None:
        coords_checkpoint = "models/*coords*/*.ckpt"
    coords_model = load_model(EDM2, coords_checkpoint, device)
    
    # Parse pocket info from structure
    from src.datasets import parse_pocket
    pocket_info = parse_pocket(pocket_structure)
    
    # Create coordinate features data
    graph, features = create_coords_features_data(pocket_info, ligand_size, dist_matrix)
    
    # Move graph data to the same device as model
    graph.to(device)
    
    # Generate coordinates using the model
    with torch.no_grad():
        final_prediction, chain = coords_model.sample_chain(graph=graph)
    
    return final_prediction, chain, pocket_info

def save_distance_matrix(dist_matrix: torch.Tensor, output_path: str):
    """Save distance matrix as PNG image"""
    ensure_parent(output_path)
    dist_np = dist_matrix.cpu().numpy()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_np, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance (Å)')
    plt.title('Distance Matrix')
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    success(f"Saved distance matrix to {output_path}")

def save_coordinates_pdb(coords: torch.Tensor, atom_names: list, output_path: str):
    """Save coordinates as PDB file"""
    ensure_parent(output_path)
    coords_np = coords.cpu().numpy()
    
    with open(output_path, 'w') as f:
        f.write("HEADER    YUEL DESIGN PREDICTION\n")
        for i, (coord, atom_name) in enumerate(zip(coords_np, atom_names)):
            f.write(f"ATOM  {i+1:5d}  {atom_name:4s} UNK A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    
    success(f"Saved coordinates to {output_path}")


def _process_coords_prediction_data(pred):
    """Process prediction tensor to extract coordinates and atom names"""
    coords_np = pred[:, :3].cpu().numpy()
    atom_name_logits = pred[:, 3:]
    atom_name_classes = torch.argmax(atom_name_logits, dim=1).cpu().numpy()
    
    # Import constants for atom name mapping
    from src import const
    predicted_atom_names = [const.IDX2PDB_ATOM[atom_name_class] for atom_name_class in atom_name_classes]
    
    return coords_np, predicted_atom_names

def _write_pdb_atoms(coords_np, predicted_atom_names, pocket_info, file_handle):
    """Write atoms to PDB file - common function for both single and trajectory PDB writing"""
    from src.pdb_utils import pdb_line

    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_atom_names = pocket_info.get('atom_names', [])
    pocket_residue_ids = pocket_info.get('res_ids', [])
    
    pocket_size = len(pocket_residue_names)
    
    # Write pocket atoms with predicted atom names but original residue names
    i = 0
    for coord, atom_name, residue_name in zip(coords_np[:pocket_size], pocket_atom_names[:pocket_size], pocket_residue_names):
        file_handle.write(pdb_line(serial=i+1, atom_name=atom_name, res_name=residue_name, chain_id='A', res_seq=pocket_residue_ids[i]+1, x=coord[0], y=coord[1], z=coord[2]))
        i += 1
    
    # Write ligand atoms with predicted atom names
    for coord, atom_name in zip(coords_np[pocket_size:], predicted_atom_names[pocket_size:]):
        file_handle.write(pdb_line(record='HETATM', serial=i+1, atom_name=atom_name, res_name='LIG', chain_id='B', res_seq=pocket_residue_ids[-1]+2, x=coord[0], y=coord[1], z=coord[2]))
        i += 1

def save_coords_pdb(pred: torch.Tensor, pocket_info: dict, output_path: str):
    """Save coordinates as PDB file with proper atom names"""
    ensure_parent(output_path)

    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_atom_names = pocket_info.get('atom_names', [])
    coords_np, predicted_atom_names = _process_coords_prediction_data(pred)
    
    with open(output_path, 'w') as f:
        f.write("HEADER    YUEL DESIGN PREDICTION\n")
        _write_pdb_atoms(coords_np, predicted_atom_names, pocket_info, f)
        f.write("END\n")
    
    success(f"Saved coordinates to {output_path}")

def save_coords_trajectory(chain: list, pocket_info: dict, output_path: str):
    """Save coordinate trajectory as multiple PDB files"""
    ensure_parent(output_path)
    
    # Get pocket residue names from pocket_info (keep original residue names)
    pocket_residue_names = pocket_info.get('residue_names', [])
    pocket_atom_names = pocket_info.get('atom_names', [])

    open(output_path, 'w').close()
    with open(output_path, 'a') as f:
        for i, pred in enumerate(chain):
            coords_np, predicted_atom_names = _process_coords_prediction_data(pred)
            
            f.write(f"MODEL {i+1:05d}\n")
            _write_pdb_atoms(coords_np, predicted_atom_names, pocket_info, f)
            f.write("END\n")
    
    success(f"Saved trajectory with {len(chain)} frames to {output_path}")

def save_chain_as_gif(chain: torch.Tensor, output_path: str, title: str = "Diffusion Process", 
                      figsize: tuple = (8, 6), fps: int = 10, dpi: int = 100):
    """Save diffusion chain as GIF animation"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image
    import numpy as np
    
    ensure_parent(output_path)
    
    # Convert chain to numpy if it's a tensor
    if isinstance(chain, torch.Tensor):
        chain_np = chain.cpu().numpy()
    else:
        chain_np = chain
    
    # Ensure chain is 3D: (frames, height, width)
    if chain_np.ndim == 4:  # (frames, batch, height, width)
        chain_np = chain_np[:, 0, :, :]  # Take first batch
    elif chain_np.ndim == 3:  # (frames, height, width)
        pass
    else:
        raise ValueError(f"Chain should be 3D or 4D, got {chain_np.ndim}D")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize with first frame
    im = ax.imshow(chain_np[0], cmap='viridis', aspect='equal')
    ax.set_title(f"{title} - Frame 0/{len(chain_np)-1}")
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('Atom Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance')
    
    def animate(frame):
        im.set_array(chain_np[frame])
        ax.set_title(f"{title} - Frame {frame}/{len(chain_np)-1}")
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(chain_np), 
                                   interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    
    success(f"Saved diffusion chain animation to {output_path}")


def run_pipeline(args, device, stages, sample_idx=None):
    """Run a single pipeline execution"""
    # Use provided seed or generate random one
    current_seed = args.seed or random.randint(0, 2**32 - 1)
    
    # Set random seed
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    
    # Add sample suffix to output files if n_samples > 1
    def add_sample_suffix(filename, sample_idx):
        base, ext = os.path.splitext(filename)
        return f"{base}_{sample_idx:03d}{ext}"
    
    # Helper function to create spinner text with sample info
    def spinner_text(base_text):
        if sample_idx is not None:
            return f"Sample {sample_idx}: {base_text}"
        return base_text
    
    # Stage: dist (Distance prediction)
    if 'dist' in stages:
        section("Distance prediction")
        
        # Read PDB file and parse structure
        with spinner(spinner_text("Loading and parsing protein structure")):
            with open(args.input_pdb, 'r') as f:
                pdb_content = f.read()
            
            from src.pdb_utils import Structure
            from io import StringIO
            pocket_structure = Structure()
            pocket_structure.read(StringIO(pdb_content))
            info(f"Parsed protein structure with {len(pocket_structure.atoms)} atoms")
        
        # Use the high-level function
        with spinner(spinner_text("Generating distance matrix")):
            dist_matrix, chain, pocket_info = run_dist_mode(
                pocket_structure=pocket_structure,
                ligand_size=args.ligand_size,
                dist_checkpoint=args.dist_checkpoint,
                device=device,
                seed=current_seed
            )
        
        # Save distance matrix if requested
        if args.save_dist is not None:
            save_distance_matrix(dist_matrix, args.save_dist)
        
        success("Distance prediction completed")

    # Stage: coords (Coordinate prediction)
    if 'coords' in stages:
        section("Coordinate prediction")
        
        # Use the high-level function
        with spinner(spinner_text("Generating coordinates")):
            final_pred, chain, pocket_info = run_coords_mode(
                pocket_structure=pocket_structure,
                ligand_size=args.ligand_size,
                dist_matrix=dist_matrix,
                coords_checkpoint=args.coords_checkpoint,
                device=device,
                seed=current_seed
            )
        
        # Save coordinates if requested
        if args.save_coords is not None:
            save_coords_pdb(final_pred, pocket_info, args.save_coords)

        if args.save_traj is not None:
            save_coords_trajectory(chain, pocket_info, args.save_traj)
        
        success("Coordinate prediction completed")


def main():
    args = parse_args()
    device = pick_device(args.device)
    
    # Validate inputs
    if args.n_samples < 1:
        error("n_samples must be at least 1")
        return
    
    # Parse pipeline string
    try:
        stages = parse_pipeline(args.pipeline)
    except ValueError as e:
        error(f"Invalid pipeline configuration: {e}")
        return
    
    section("Yuel Design Pipeline")
    info(f"Stages: {' → '.join(stages)}")
    info(f"Device: {device.type}")
    
    
    # Use provided seed or generate random one
    args.seed = args.seed or random.randint(0, 2**32 - 1)
    info(f"Using seed {args.seed}")
    
    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        info(f"Created output directory: {args.output_dir}")
    
    # Run the pipeline n_samples times with different seeds
    for sample_idx in range(args.n_samples):
        if args.n_samples > 1:
            section(f"Sample {sample_idx + 1}/{args.n_samples}")
            # Generate a new random seed for this sample
            sample_seed = random.randint(0, 2**32 - 1)
            args.seed = sample_seed
            info(f"Sample {sample_idx + 1}: Using seed {sample_seed}")
        
        try:
            # Run the pipeline for this sample
            run_pipeline(args, device, stages, sample_idx if args.n_samples > 1 else None)
        except Exception as e:
            error(f"Pipeline failed for sample {sample_idx + 1}: {e}")
            if args.n_samples == 1:
                return  # Exit on first failure if only one sample
    
    if args.n_samples > 1:
        success(f"Completed {args.n_samples} pipeline runs")
    else:
        success("Pipeline completed")


if __name__ == '__main__':
    main()
