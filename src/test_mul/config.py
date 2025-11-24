# Project settings
project = "yuel_design"
exp_name = "test_mul"
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
num_workers = 0  # Will become useful with larger batch_size

# Model parameters
diffusion_steps = 100
diffusion_noise_schedule = 'polynomial_2'
diffusion_noise_precision = 1e-5
t_weight_power = 0.0

# E3former parameters
n_blocks = 8  # Number of E3former blocks
hidden_nf = 64  # Latent dimension for sequence/pair representations
in_node_features = 88  # 3 (mol_types: backbone, side_chain, ligand) + 51 (atom_onehot: X + protein_atoms)
in_edge_features = 1  # pair features: [is_same_residue, ca_distance]
num_ligand_atom_types = 20  # Number of ligand atom types (X + 19 ligand elements)
# Note: seq_input_dim in model will be in_node_features + num_ligand_atom_types + 2 = 54 + 20 + 2 = 76
# (the +2 is for receptor_mask and time step)
no_heads_seq = 4
no_heads_pair = 2
transition_n = 4
blocks_per_ckpt = 4
chunk_size = 4
