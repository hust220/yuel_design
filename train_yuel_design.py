import argparse
import os
import sys
import yaml
import torch
import wandb
import multiprocessing

from datetime import datetime
from pytorch_lightning import Trainer, callbacks, loggers

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from src.lightning1 import YuelDesign
from src.edm2 import EDM as EDM2
from src.DistD3PM import D3PM as DistD3PM
from src.utils import disable_rdkit_logging, set_deterministic, Logger
from src.datasets import DATASET_CONFIGS, DistDataset, CoordsDataset

def args_to_dict(args):
    """Convert argparse.Namespace to dictionary"""
    return vars(args)

def init_wandb(args, experiment):
    """Initialize wandb logger"""
    # Login to wandb using API key from file
    wandb_api_key_file = "wandb_api_key.txt"
    if os.path.exists(wandb_api_key_file):
        try:
            with open(wandb_api_key_file, 'r') as f:
                api_key = f.read().strip()
            os.environ['WANDB_API_KEY'] = api_key
            wandb.login(key=api_key, relogin=True)
            print("Successfully logged in to wandb")
        except Exception as e:
            print(f"Warning: Failed to login to wandb: {e}")
    else:
        print(f"Warning: {wandb_api_key_file} not found. Please ensure wandb is logged in manually.")
    
    wandb_logger = loggers.WandbLogger(
        save_dir=args['logs'],
        project='yuel_design',
        name=experiment,
        id=experiment,
        resume='must' if args['resume'] is not None else 'allow',
        entity=args['wandb_entity'],
    )
    
    return wandb_logger

def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt') and '=' in fname
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)

def main(args):
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{args["exp_name"]}_bs{args["batch_size"]}_{start_time}'
    experiment = run_name if args["resume"] is None else args["resume"]
    checkpoints_dir = os.path.join(args["checkpoints"], experiment)
    print(f'Checkpoints directory: {checkpoints_dir}')
    os.makedirs(os.path.join(args["logs"], "general_logs", experiment),exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(args["logs"], "general_logs", experiment, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(args["logs"], "general_logs", experiment, f'log.log'), syspart=sys.stderr)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args["logs"], exist_ok=True)

    set_deterministic(args["seed"])
    device = args["device"]
    torch_device = torch.device(device)
    args['device'] = torch_device
    
    # Initialize wandb
    wandb_logger = init_wandb(args, experiment)
    
    # Handle dataset configuration
    if args['dataset'] in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[args['dataset']]
        for k, v in dataset_config.items():
            args[k] = v
    args['device'] = torch_device
    
    if args['dataset'] == 'dist':
        # Distance prediction mode using DistD3PM + DistDataset
        model = YuelDesign(model_class=DistD3PM, **args)
    elif args['dataset'] == 'coords':
        # Coordinate prediction mode using EDM2 + CoordsDataset
        model = YuelDesign(model_class=EDM2, **args)
    else:
        raise ValueError(f"Invalid dataset: {args['dataset']}. Supported datasets: 'dist', 'coords'")

    # Regular checkpoint saving (every epoch)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_{epoch:02d}',
        monitor='loss/val',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )
    
    # Frequent checkpoint saving (every N iterations)
    frequent_checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_iter_{step:06d}',
        every_n_train_steps=args.get('save_every_n_steps', 1000),  # Save every 1000 steps by default
        monitor='loss/train',
        save_top_k=3,  # Keep only the last 3 frequent checkpoints
        save_last=True,
    )

    # Smart device detection for different platforms
    if device == 'cuda':
        accelerator = 'gpu'
    elif device == 'mps':
        accelerator = 'mps'  # Mac MPS support
    else:
        accelerator = 'cpu'

    print("Device: ", device, "Accelerator: ", accelerator)
    
    trainer = Trainer(
        max_epochs=args['n_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, frequent_checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=args['enable_progress_bar'],
    )

    if args['resume'] is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training')
    trainer.fit(model=model, ckpt_path=last_checkpoint)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RNA Design')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/train_dist.yml')
    p.add_argument('--data', action='store', type=str,  default="datasets")
    p.add_argument('--dataset', action='store', type=str,  default="coords", help='dist | coords')
    p.add_argument('--discretization_config', type=str, default='b12', help='Discretization configuration: b5, b8, b12, b16')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--device', action='store', type=str, default='cuda')
    p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
    p.add_argument('--log_iterations', action='store', type=int, default=20)
    p.add_argument('--coord_scale_factor', type=float, default=0.1, help='Scale factor for the coordinates')

    p.add_argument('--exp_name', type=str, default='YourName')
    p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')
    p.add_argument('--t_weight_power', type=float, default=0.0, help='Power for weighting t sampling (higher values favor larger t)')
    p.add_argument('--forward_type', type=str, default='uniform', help='Forward process type for D3PM (uniform)')
    p.add_argument('--hybrid_loss_coeff', type=float, default=0.0, help='Coefficient for hybrid loss in D3PM')
    p.add_argument('--free_all_nodes_early', type=eval, default=False, help='Free all nodes in early timesteps (t <= 0.25) for training and late timesteps (s <= 0.25) for sampling')

    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--brute_force', type=eval, default=False,help='True | False')
    p.add_argument('--actnorm', type=eval, default=True,help='True | False')
    p.add_argument('--break_train_epoch', type=eval, default=False,help='True | False')
    p.add_argument('--dp', type=eval, default=True,help='True | False')
    p.add_argument('--condition_time', type=eval, default=True,help='True | False')
    p.add_argument('--clip_grad', type=eval, default=True,help='True | False')
    p.add_argument('--trace', type=str, default='hutch',help='hutch | exact')
    # EGNN args -->
    p.add_argument('--n_layers', type=int, default=6,   help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--nf', type=int, default=128,  help='number of layers')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=1,help='diff/(|diff| + norm_constant)')
    p.add_argument('--ode_regularization', type=float, default=1e-3)
    p.add_argument('--datadir', type=str, default='qm9/temp',  help='qm9 directory')
    p.add_argument('--filter_n_atoms', type=int, default=None, help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    p.add_argument('--dequantization', type=str, default='argmax_variational',  help='uniform | variational | argmax_variational | deterministic')
    p.add_argument('--n_report_steps', type=int, default=1)
    p.add_argument('--wandb_usr', type=str)
    p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
    p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    p.add_argument('--no-cuda', action='store_true', default=False,  help='enables CUDA training')
    p.add_argument('--save_model', type=eval, default=True, help='save model')
    p.add_argument('--generate_epochs', type=int, default=1,help='save model')
    p.add_argument('--num_workers', type=int, default=4, help='Number of worker for the dataloader')
    p.add_argument('--cache_mode', type=str, default='memory', help='Cache mode: memory, file, or none')
    p.add_argument('--cache_dir', type=str, default='cache', help='Cache directory for file cache')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--augment_noise', type=float, default=0)
    p.add_argument('--n_stability_samples', type=int, default=500,help='Number of samples to compute the stability')
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    p.add_argument('--remove_h', action='store_true')
    p.add_argument('--visualize_every_batch', type=int, default=1e8,help="Can be used to visualize multiple times per epoch")
    p.add_argument('--normalization_factor', type=float, default=1,help="Normalize the sum aggregation of EGNN")
    p.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    p.add_argument('--bidirectional', type=eval, default=False, help='Bidirectional edges')
    p.add_argument('--activation', type=str, default='silu', help='Activation function')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    

    disable_rdkit_logging()

    args = vars(p.parse_args())
    if args['config']:
        config_args = yaml.load(args['config'], Loader=yaml.FullLoader)

    for k,v in config_args.items():
        args[k] = v

    main(args=args)
