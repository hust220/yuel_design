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

# EGNN parameters
n_blocks = 16  # Number of GNN blocks (n_layers in EGNN)
hidden_nf = 64  # Hidden feature dimension for nodes and edges
in_node_features = 72  # 3 (mol_types) + 69 (atom_one_hot: X + protein_atoms + ligand_elements)
in_edge_features = 2  # edge_attr: [is_same_residue, is_interaction]
# Note: EGNN internally adds distance to edge features before embedding
# Mask and time step are added via node_attr (node_attr_nf=2)
# EGNN expects edge_feat_nf = in_edge_features = 2 (distance will be added internally)
