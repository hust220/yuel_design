import sys
import random
import os
import glob
import re
from datetime import datetime
from src.lightning1 import LightningWrapper
from pytorch_lightning.loggers import WandbLogger
import wandb

import torch
import numpy as np


def move_data_to_device(data, device):
    """
    Move data from CPU to target device if needed.
    This function is device-agnostic and supports CUDA, MPS, and CPU devices.
    
    Args:
        data: Dictionary containing tensors and other data
        device: Target device (torch.device)
    
    Returns:
        Dictionary with tensors moved to target device
    """
    if device.type == 'cpu' or data is None:
        return data
    
    # Check if data contains tensors on CPU and move them to target device
    moved_data = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                non_blocking = device.type == 'cuda'
                moved_data[key] = value.to(device, non_blocking=non_blocking)
            else:
                moved_data[key] = value
        return moved_data

    # Fallback: return as-is
    return data


def pick_latest(patterns, exclude=None):
    """Find the latest checkpoint file matching the given patterns, excluding specified patterns."""
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if exclude:
        files = [f for f in files if not any(x in f for x in exclude)]
    if not files:
        raise FileNotFoundError('No matching checkpoint found')
    def parse_info(filepath):
        name = os.path.basename(filepath)
        # Extract timestamp from filename like: date01-10_time12-09-56.120775
        tm = re.search(r"date(\d{2})-(\d{2})_time(\d{2})-(\d{2})-(\d{2})(?:\.(\d+))?", name)
        em = re.search(r"epoch=(\d+)", name)
        epoch = int(em.group(1)) if em else -1
        if tm:
            day = int(tm.group(1))
            month = int(tm.group(2))
            hour = int(tm.group(3))
            minute = int(tm.group(4))
            second = int(tm.group(5))
            microsecond = int((tm.group(6) or '0')[:6].ljust(6, '0'))
            # Use file's mtime year to avoid year-boundary ambiguity
            year = datetime.fromtimestamp(os.path.getmtime(filepath)).year
            try:
                dt = datetime(year, month, day, hour, minute, second, microsecond)
            except ValueError:
                dt = None
        else:
            dt = None
        return dt, epoch

    infos = [(f, *parse_info(f)) for f in files]
    with_dt = [it for it in infos if it[1] is not None]

    if with_dt:
        # Prefer latest timestamp, then highest epoch, then filename for determinism
        return max(with_dt, key=lambda it: (it[1], it[2], it[0]))[0]

    with_epoch = [it for it in infos if it[2] >= 0]
    if with_epoch:
        return max(with_epoch, key=lambda it: (it[2], it[0]))[0]

    # Fallback to modification time if neither timestamp nor epoch found
    return max(files, key=os.path.getmtime)

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def isatty(self):
        # delegate to underlying terminal's isatty() for TTY detection
        return self.terminal.isatty() if hasattr(self.terminal, 'isatty') else False


def log(*args):
    print(f'[{datetime.now()}]', *args)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask):
    """
    Subtract center of mass of protein from coordinates of all atoms
    """
    x_masked = x * center_of_mass_mask
    N = center_of_mass_mask.sum(1, keepdims=True)
    mean = torch.sum(x_masked, dim=1, keepdim=True) / N
    # print(f'mean {mean}')
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Partial mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    # TODO: check it
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    # print(x[0, 0, 0])
    x_masked = x * node_mask
    # print(x_masked[0, 0, 0])
    return x_masked


def concatenate_features(x, h):
    xh = torch.cat([x, h['categorical']], dim=2)
    if 'integer' in h:
        xh = torch.cat([xh, h['integer']], dim=2)
    return xh


def split_features(z, n_dims, num_classes):
    assert z.size(2) == n_dims + num_classes
    x = z[:, :, 0:n_dims]
    h = {'categorical': z[:, :, n_dims:]}
    return x, h


# For gradient clipping

class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')
    return grad_norm


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FoundNaNException(Exception):
    def __init__(self, x, h):
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)

        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        idx = set()
        for i in range(z.shape[0]):
            if torch.any(torch.isnan(z[i])):
                idx.add(i)
        return idx


def get_batch_idx_for_animation(batch_size, batch_idx):
    batch_indices = []
    mol_indices = []
    for idx in [0, 110, 360]:
        if idx // batch_size == batch_idx:
            batch_indices.append(idx % batch_size)
            mol_indices.append(idx)
    return batch_indices, mol_indices


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


def init_wandb(args, experiment, project, wandb_api_key_file="wandb_api_key.txt"):
    """
    Initialize wandb logger with API key from file.
    
    Args:
        args: Dictionary containing wandb configuration
        experiment: Experiment name/ID
        project: Wandb project name
        wandb_api_key_file: Path to file containing wandb API key (default: "wandb_api_key.txt")
    
    Returns:
        WandbLogger instance
    """
    # Login to wandb using API key from file
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
    
    wandb_logger = WandbLogger(
        save_dir=args['logs'],
        project=project,
        name=experiment,
        id=experiment,
        resume='must' if args['resume'] is not None else 'allow',
        entity=args['wandb_entity'],
    )
    
    return wandb_logger


def find_latest_checkpoint(checkpoints_dir):
    """
    Find the latest checkpoint file in the given directory.
    
    Priority order:
    1. last.ckpt
    2. epoch=XX.ckpt (max epoch)
    3. step=XX.ckpt (max step)
    4. Any .ckpt file (by modification time)
    
    Args:
        checkpoints_dir: Directory containing checkpoint files
    
    Returns:
        str: Path to the latest checkpoint file
    
    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    import re
    
    # First, try to find last.ckpt
    last_ckpt = os.path.join(checkpoints_dir, 'last.ckpt')
    if os.path.exists(last_ckpt):
        return last_ckpt
    
    # If not found, look for epoch=XX.ckpt files and find the one with max epoch
    epoch_files = []
    for fname in os.listdir(checkpoints_dir):
        if fname.endswith('.ckpt'):
            epoch_match = re.search(r'epoch=(\d+)', fname)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epoch_files.append((epoch, fname))
    
    if epoch_files:
        latest_epoch_file = max(epoch_files, key=lambda t: t[0])[1]
        return os.path.join(checkpoints_dir, latest_epoch_file)
    
    # If no epoch files found, look for step=XX.ckpt files and find max step
    step_files = []
    for fname in os.listdir(checkpoints_dir):
        if fname.endswith('.ckpt'):
            step_match = re.search(r'step=(\d+)', fname)
            if step_match:
                step = int(step_match.group(1))
                step_files.append((step, fname))
    
    if step_files:
        latest_step_file = max(step_files, key=lambda t: t[0])[1]
        return os.path.join(checkpoints_dir, latest_step_file)
    
    # Fallback: return any .ckpt file (by modification time)
    ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    if ckpt_files:
        latest_file = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
        return os.path.join(checkpoints_dir, latest_file)
    
    raise FileNotFoundError(f'No checkpoint files found in {checkpoints_dir}')


def run_training(args=None, model=None, dataset=None, config=None):
    """
    Run the complete training pipeline.
    
    Args:
        args: Dictionary containing training configuration (optional if config provided)
        model: The model class to train (e.g., C4Opt)
        dataset: The dataset class (e.g., C4OptDataset) (optional if args provided)
        config: The config module (e.g., src.c4_opt.config) (optional if args provided)
    """
    import multiprocessing
    from pytorch_lightning import Trainer, callbacks
    
    # Handle configuration loading if config is provided
    if config is not None and args is None:
        # Convert config to args dictionary
        args = {}
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                args[attr_name] = getattr(config, attr_name)
        
        args['dataset_class'] = dataset
        
    # Disable rdkit logging
    disable_rdkit_logging()

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Setup experiment directories, logging, and wandb
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{args["exp_name"]}_bs{args["batch_size"]}_{start_time}'
    experiment = run_name if args["resume"] is None else args["resume"]
    checkpoints_dir = os.path.join(args["checkpoints"], experiment)
    
    print(f'Checkpoints directory: {checkpoints_dir}')
    
    # Create directories
    os.makedirs(os.path.join(args["logs"], "general_logs", experiment), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args["logs"], exist_ok=True)
    
    # Setup logging
    sys.stdout = Logger(logpath=os.path.join(args["logs"], "general_logs", experiment, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(args["logs"], "general_logs", experiment, f'log.log'), syspart=sys.stderr)
    
    # Initialize wandb
    wandb_logger = init_wandb(args, experiment, args['project'])

    set_deterministic(args["seed"])
    device = args["device"]
    args['device'] = torch.device(device)
    
    # Create model with wrapper
    model = LightningWrapper(model_class=model, **args)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_{epoch:02d}',
        monitor='loss/val',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    # Optional frequent checkpointing (every N train steps)
    callbacks_list = [checkpoint_callback]
    save_every_n_steps = args.get('save_every_n_steps')
    if save_every_n_steps is not None and int(save_every_n_steps) > 0:
        frequent_checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=experiment + '_iter_{step:06d}',
            every_n_train_steps=int(save_every_n_steps),
            monitor='loss/train',
            save_top_k=3,
            save_last=True,
        )
        callbacks_list.append(frequent_checkpoint_callback)

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
        callbacks=callbacks_list,
        accelerator=accelerator,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=args['enable_progress_bar'],
        # precision='16-mixed',  # Enable automatic mixed precision training
    )

    if args['resume'] is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_latest_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training model')
    trainer.fit(model=model, ckpt_path=last_checkpoint)
