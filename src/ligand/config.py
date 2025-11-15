# Project settings
project = "yuel_design"
exp_name = "ligand"
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
cache_mode = "file"  # No caching (recompute features each time)
cache_dir = "cache"
low_memory = False

# Dataset settings
num_workers = 8  # Will become useful with larger batch_size

# Model parameters
diffusion_steps = 100
diffusion_noise_schedule = 'polynomial_2'
diffusion_noise_precision = 1e-5
t_weight_power = 0.0

# E3former parameters
n_blocks = 8  # Number of E3former blocks
hidden_nf = 64  # Latent dimension for sequence/pair representations
in_node_features = 72  # 3 (mol_types) + 69 (atom_one_hot: X + protein_atoms + ligand_elements)
in_edge_features = 2  # pair features: [is_same_residue, is_interaction]
no_heads_seq = 4
no_heads_pair = 2
transition_n = 4
blocks_per_ckpt = 4
chunk_size = 4
