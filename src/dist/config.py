# Project settings
project = "yuel_design"
exp_name = "dist"
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
resume = None
save_every_n_steps = 1000  # Save checkpoint every 1000 training steps
num_workers = 8

# Memory settings
cache_mode = "file"
cache_dir = "cache"

# Model parameters
c_m = 64
c_z = 64
c_hidden_seq_att = 32
c_hidden_opm = 32
c_hidden_mul = 32
c_hidden_pair_att = 32
c_s = 1
no_heads_seq = 8
no_heads_pair = 4
no_blocks = 16
transition_n = 4
blocks_per_ckpt = 4
inf = 1e9
eps = 1e-10

timesteps = 100
forward_type = "uniform"
hybrid_loss_coeff = 0.0

no_dist_bins = 12
seq_input_dim = 44 # 3 (mol_types) + 1 (CA) + 20 (side chain types) + 19 (ligand atoms) + 1 ('X')
z_input_dim = 1
bb_dist_bins = 13