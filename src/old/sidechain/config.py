# Project settings
project = "yuel_design"
exp_name = "sidechain"
wandb_entity = None  # Set to your wandb username/team, or None for default
enable_progress_bar = True

# Training settings
n_epochs = 1000
batch_size = 1
lr = 2e-4
logs = "logs"
log_iterations = 20
checkpoints = "checkpoints"
seed = 42
device = "cuda"
resume = None  # Set to experiment name to resume training

# Memory settings
cache_mode = "file"
cache_dir = "cache"
low_memory = False

# Dataset settings
num_workers = 8

# Model parameters
diffusion_steps = 100
diffusion_noise_schedule = 'polynomial_2'
diffusion_noise_precision = 1e-5
t_weight_power = 0.0
n_blocks = 16
hidden_nf = 64
in_node_features = 52  # 2 (mol_types: backbone, sidechain) + 50 (atom_one_hot: X + protein_atoms)
in_edge_features = 14  # 13 (edge_dist one-hot encoded: 0-12) + 1 (edge_attr: is_same_residue)
