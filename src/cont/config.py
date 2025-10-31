# Project settings
project = "yuel_design"
exp_name = "coords"
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
# resume = 'coords_bs1_date30-10_time07-35-25.132397'  # Set to experiment name to resume training
resume = 'coords_bs1_date30-10_time10-51-35.275969'  # Set to experiment name to resume training

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
in_node_features = 44 # 3 (mol_types) + 1 (CA) + 20 (side chain types) + 19 (ligand atoms) + 1 ('X')
in_edge_features = 1
