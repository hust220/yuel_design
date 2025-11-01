# Project settings
project = "yuel_design"
exp_name = "cont"
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
num_workers = 4

# Model parameters
diffusion_steps = 100
diffusion_noise_schedule = 'polynomial_2'
diffusion_noise_precision = 1e-5
t_weight_power = 0.0
n_blocks = 16
hidden_nf = 64
in_node_features = 122 # 3 (mol_types) + 1 (X) + 20 (protein cg atom types) + 79 (protein aa atom types) + 19 (ligand atoms)
in_edge_features = 19 # 13 (distance) + 5 (ligand bond type) + 1 (is_same_residue)
