import argparse
import os
import random
import torch
import yaml
from pathlib import Path

# Yuel Design imports
from src.datasets import DistDataset, CoordsDataset, DATASET_CONFIGS
from src.lightning1 import YuelDesign
from src.DistD3PM import D3PM as DistD3PM
from src.edm2 import EDM as EDM2
from src.console import section, info, success, warn, error, spinner


def parse_args():
    p = argparse.ArgumentParser(description='Yuel Design: Protein-Ligand Design Pipeline')

    # Pipeline stages
    p.add_argument('--pipeline', type=str, default='dist:coords', help='Pipeline stages: dist, coords, or dist:coords')

    # Model configurations
    p.add_argument('--dist_config', type=str, default='configs/train_dist.yml', help='Config file for distance prediction model')
    p.add_argument('--coords_config', type=str, default='configs/train_coords.yml', help='Config file for coordinate prediction model')
    
    # Model checkpoints
    p.add_argument('--dist_checkpoint', type=str, default=None, help='Checkpoint path for distance prediction model')
    p.add_argument('--coords_checkpoint', type=str, default=None, help='Checkpoint path for coordinate prediction model')

    # Input/Output
    p.add_argument('--input_pdb', type=str, default=None, help='Input protein PDB file')
    p.add_argument('--input_ligand', type=str, default=None, help='Input ligand SMILES string')
    p.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    # Save controls
    p.add_argument('--save_dist_matrix', type=str, default=None, help='Save distance matrix as numpy file')
    p.add_argument('--save_coords_pdb', type=str, default='output/coords.pdb', help='Save predicted coordinates as PDB')
    p.add_argument('--save_trajectory', type=str, default=None, help='Save generation trajectory')

    # Generation parameters
    p.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate')
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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_class, config: dict, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    model = model_class(**config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        info(f"Loaded model from {checkpoint_path}")
    else:
        warn(f"No checkpoint found at {checkpoint_path}, using random initialization")
    model.to(device)
    model.eval()
    return model


def save_distance_matrix(dist_matrix: torch.Tensor, output_path: str):
    """Save distance matrix as numpy file"""
    ensure_parent(output_path)
    dist_np = dist_matrix.cpu().numpy()
    import numpy as np
    np.save(output_path, dist_np)
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


def run_pipeline(args, device, stages, sample_idx=None):
    """Run a single pipeline execution"""
    # Generate random seed if not provided
    current_seed = args.seed
    if current_seed is None:
        current_seed = random.randint(0, 2**32 - 1)
        if sample_idx is not None:
            info(f"Sample {sample_idx}: Generated random seed {current_seed}")
        else:
            info(f"Generated random seed {current_seed}")
    
    # Set random seed
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    
    # Add sample suffix to output files if n_samples > 1
    def add_sample_suffix(filename, sample_idx):
        if filename is None or sample_idx is None:
            return filename
        base, ext = os.path.splitext(filename)
        return f"{base}_{sample_idx:03d}{ext}"
    
    # Helper function to create spinner text with sample info
    def spinner_text(base_text):
        if sample_idx is not None:
            return f"Sample {sample_idx}: {base_text}"
        return base_text
    
    # Modify output file names if this is a multi-sample run
    dist_matrix_file = add_sample_suffix(args.save_dist_matrix, sample_idx)
    coords_pdb_file = add_sample_suffix(args.save_coords_pdb, sample_idx)
    trajectory_file = add_sample_suffix(args.save_trajectory, sample_idx)

    # Containers passed between stages
    dist_matrix = None
    coordinates = None
    atom_names = None

    # Stage: dist (Distance prediction)
    if 'dist' in stages:
        section("Distance prediction")
        
        # Load distance prediction model
        dist_config = load_config(args.dist_config)
        with spinner(spinner_text("Loading distance prediction model")):
            dist_model = load_model(DistD3PM, dist_config, args.dist_checkpoint, device)
        
        # Create sample data for distance prediction
        # This is a placeholder - in practice, you would load real protein-ligand data
        with spinner(spinner_text("Preparing input data")):
            # Create dummy data for demonstration
            batch_size = 1
            seq_length = 50  # Example sequence length
            
            # Create dummy sequence features
            seq = torch.randn(batch_size, seq_length, 35)  # mol_types(3) + codes(32)
            z = torch.randn(batch_size, seq_length, seq_length, 4)  # residue relationships
            seq_mask = torch.ones(batch_size, seq_length)
            pair_mask = torch.ones(batch_size, seq_length, seq_length)
            
            # Create dummy distance target (this would be real data in practice)
            dist_target = torch.randint(0, 12, (batch_size, seq_length, seq_length))
            
            data = {
                'seq': seq,
                'z': z,
                'dist': dist_target,
                'seq_mask': seq_mask,
                'pair_mask': pair_mask
            }
        
        # Generate distance matrix using the model
        with spinner(spinner_text("Generating distance matrix")):
            with torch.no_grad():
                # Sample from the model
                chain, _ = dist_model.sample_chain(data)
                dist_matrix = chain[-1].view(seq_length, seq_length)
        
        # Save distance matrix if requested
        if dist_matrix_file:
            save_distance_matrix(dist_matrix, dist_matrix_file)
        
        success("Distance prediction completed")

    # Stage: coords (Coordinate prediction)
    if 'coords' in stages:
        section("Coordinate prediction")
        
        # Load coordinate prediction model
        coords_config = load_config(args.coords_config)
        with spinner(spinner_text("Loading coordinate prediction model")):
            coords_model = load_model(EDM2, coords_config, args.coords_checkpoint, device)
        
        # Use distance matrix from previous stage or create dummy data
        if dist_matrix is None:
            warn("No distance matrix from previous stage, using dummy data")
            dist_matrix = torch.randint(0, 12, (50, 50))
        
        # Create sample data for coordinate prediction
        with spinner(spinner_text("Preparing coordinate prediction data")):
            # Create dummy graph data for EDM2
            n_atoms = dist_matrix.shape[0]
            positions = torch.randn(n_atoms, 3)  # Random initial positions
            one_hot = torch.randn(n_atoms, 10)  # Element type features
            
            # Create edge index (all pairs)
            edge_index = []
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    edge_index.append([i, j])
                    edge_index.append([j, i])
            edge_index = torch.tensor(edge_index).T
            
            # Create edge attributes
            edge_attr = torch.randn(edge_index.shape[1], 15 + 2 * 10)  # distance + residue + 2*element
            
            # Create node attributes
            node_attr = torch.randn(n_atoms, 3 + 100)  # mol_types + PDB atom types
            
            # Create masks
            node_mask = torch.ones(n_atoms)
            anchor_mask = torch.zeros(n_atoms)
            edge_mask = torch.ones(edge_index.shape[1])
            
            data = {
                'positions': positions,
                'one_hot': one_hot,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'node_attr': node_attr,
                'node_mask': node_mask,
                'anchor_mask': anchor_mask,
                'edge_mask': edge_mask
            }
        
        # Generate coordinates using the model
        with spinner(spinner_text("Generating coordinates")):
            with torch.no_grad():
                # Sample coordinates from the model
                coordinates = coords_model.sample(data)
        
        # Create dummy atom names
        atom_names = [f'ATOM_{i}' for i in range(n_atoms)]
        
        # Save coordinates if requested
        if coords_pdb_file:
            save_coordinates_pdb(coordinates, atom_names, coords_pdb_file)
        
        success("Coordinate prediction completed")


def main():
    args = parse_args()
    device = pick_device(args.device)
    
    # Parse pipeline string
    stages = parse_pipeline(args.pipeline)
    section("Yuel Design Pipeline")
    info(f"Stages: {' → '.join(stages)}")
    info(f"Device: {device.type}")
    
    # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        info(f"Generated random seed {args.seed}")
    
    # Run the pipeline n_samples times with different seeds
    for sample_idx in range(args.n_samples):
        if args.n_samples > 1:
            section(f"Sample {sample_idx + 1}/{args.n_samples}")
            # Generate a new random seed for this sample
            sample_seed = random.randint(0, 2**32 - 1)
            args.seed = sample_seed
            info(f"Sample {sample_idx + 1}: Using seed {sample_seed}")
        
        # Run the pipeline for this sample
        run_pipeline(args, device, stages, sample_idx if args.n_samples > 1 else None)
    
    if args.n_samples > 1:
        success(f"Completed {args.n_samples} pipeline runs")
    else:
        success("Pipeline completed")


if __name__ == '__main__':
    main()
