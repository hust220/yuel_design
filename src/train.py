import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
import glob

from src.utils import (
    Logger, log, set_deterministic, disable_rdkit_logging, 
    move_data_to_device, find_latest_checkpoint
)


def filter_dataset_params(dataset_class, kwargs):
    """Filter hyperparameters to only include dataset-relevant ones"""
    import inspect
    sig = inspect.signature(dataset_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'split', 'device'}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def get_collate_fn(dataset_class):
    """Get collate function from dataset class or use default"""
    dataset_collate = getattr(dataset_class, 'collate_fn', None)
    if callable(dataset_collate):
        return dataset_collate
    else:
        from src.datasets import collate
        return collate


def create_dataloader(dataset, batch_size, collate_fn, shuffle=False, num_workers=0):
    """Create DataLoader with appropriate settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
    )


def save_checkpoint(model, optimizer, epoch, step, metrics, checkpoint_dir, experiment_name, 
                   is_best=False, is_step_checkpoint=False):
    """Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current global step
        metrics: Metrics history
        checkpoint_dir: Directory to save checkpoint
        experiment_name: Experiment name
        is_best: Whether this is the best checkpoint
        is_step_checkpoint: Whether this is a mid-epoch step checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if is_step_checkpoint:
        # Save step checkpoint (for training interruption recovery)
        checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_step_{step:06d}.ckpt')
        torch.save(checkpoint, checkpoint_path)
    else:
        # Save epoch checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_epoch_{epoch:04d}.ckpt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save last checkpoint
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
        torch.save(checkpoint, last_checkpoint_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best.ckpt')
            torch.save(checkpoint, best_checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint.get('step', 0)
    metrics = checkpoint.get('metrics', {})
    return epoch, step, metrics


def cleanup_old_step_checkpoints(checkpoint_dir, experiment_name, keep_top_k=3):
    """Clean up old step checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        experiment_name: Experiment name
        keep_top_k: Number of most recent step checkpoints to keep
    """
    # Find all step checkpoints
    step_checkpoints = glob.glob(os.path.join(checkpoint_dir, f'{experiment_name}_step_*.ckpt'))
    
    if len(step_checkpoints) <= keep_top_k:
        return
    
    # Sort by modification time (newest first)
    step_checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Remove older checkpoints beyond keep_top_k
    for checkpoint_path in step_checkpoints[keep_top_k:]:
        try:
            os.remove(checkpoint_path)
            print(f'Removed old step checkpoint: {os.path.basename(checkpoint_path)}')
        except Exception as e:
            print(f'Warning: Failed to remove {checkpoint_path}: {e}')


def init_wandb(args, experiment, project, wandb_api_key_file="wandb_api_key.txt"):
    """Initialize wandb logger with API key from file"""
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
    
    wandb.init(
        project=project,
        name=experiment,
        id=experiment,
        resume='must' if args.get('resume') is not None else 'allow',
        entity=args.get('wandb_entity'),
        dir=args.get('logs', 'logs'),
        config=args,
    )
    
    return wandb


def run_training(args=None, model=None, dataset=None, config=None):
    """
    Run the complete training pipeline without PyTorch Lightning.
    
    Args:
        args: Dictionary containing training configuration (optional if config provided)
        model: The model class to train (e.g., LigandModel)
        dataset: The dataset class (e.g., LigandDataset) (optional if args provided)
        config: The config module (e.g., src.ligand.config) (optional if args provided)
    """
    import multiprocessing
    
    # Handle configuration loading if config is provided
    if config is not None and args is None:
        args = {}
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                args[attr_name] = getattr(config, attr_name)
        args['dataset_class'] = dataset
    
    # Disable rdkit logging
    disable_rdkit_logging()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Setup experiment directories and logging
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    run_name = f'{args["exp_name"]}_bs{args["batch_size"]}_{start_time}'
    experiment = run_name if args.get("resume") is None else args["resume"]
    checkpoints_dir = os.path.join(args["checkpoints"], experiment)
    
    print(f'Checkpoints directory: {checkpoints_dir}')
    
    # Create directories
    os.makedirs(os.path.join(args["logs"], "general_logs", experiment), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args["logs"], exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(args["logs"], "general_logs", experiment, 'log.log')
    sys.stdout = Logger(logpath=log_path, syspart=sys.stdout)
    sys.stderr = Logger(logpath=log_path, syspart=sys.stderr)
    
    # Initialize wandb
    if args.get('wandb_entity') is not None or os.path.exists("wandb_api_key.txt"):
        wandb_run = init_wandb(args, experiment, args['project'])
    else:
        wandb_run = None
        print("Wandb not initialized (no entity or API key)")
    
    # Set random seeds
    set_deterministic(args["seed"])
    
    # Setup device
    device_str = args["device"]
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_str == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Get dataset class
    dataset_class = args['dataset_class']
    dataset_params = filter_dataset_params(dataset_class, args)
    
    # Create datasets
    train_dataset = dataset_class(split='train', **dataset_params)
    val_dataset = dataset_class(split='val', **dataset_params)
    
    # Get collate function
    collate_fn = get_collate_fn(dataset_class)
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=args['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=args.get('num_workers', 0)
    )
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=args['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=args.get('num_workers', 0)
    )
    
    # Create model
    model = model(**args)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], amsgrad=True, weight_decay=1e-12)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    metrics_history = {}
    
    if args.get('resume') is not None:
        try:
            checkpoint_path = find_latest_checkpoint(checkpoints_dir)
            print(f'Resuming training from checkpoint: {checkpoint_path}')
            checkpoint_epoch, checkpoint_step, metrics_history = load_checkpoint(model, optimizer, checkpoint_path, device)
            # Resume from next epoch after checkpoint (checkpoint_epoch is the last completed epoch)
            start_epoch = checkpoint_epoch + 1
            # Use checkpoint step as starting global step
            global_step = checkpoint_step
            if 'loss/val' in metrics_history and len(metrics_history['loss/val']) > 0:
                best_val_loss = min(metrics_history['loss/val'])
            print(f'Resumed from epoch {start_epoch} (checkpoint was at epoch {checkpoint_epoch}, step {checkpoint_step})')
        except FileNotFoundError as e:
            print(f'Warning: {e}. Starting training from scratch.')
    
    # Training loop
    log_iterations = args.get('log_iterations', 20)
    save_every_n_steps = args.get('save_every_n_steps')
    keep_top_k_steps = 3  # Keep only the most recent 3 step checkpoints
    n_epochs = args['n_epochs']
    enable_progress_bar = args.get('enable_progress_bar', True)
    
    print('Start training model')
    if save_every_n_steps is not None and int(save_every_n_steps) > 0:
        print(f'Will save checkpoint every {save_every_n_steps} training steps (keeping top {keep_top_k_steps})')
    
    for epoch in range(start_epoch, n_epochs):
        # Training phase
        model.train()
        train_metrics = {}
        train_losses = []
        
        if enable_progress_bar:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        else:
            train_pbar = train_loader
        
        for batch_idx, data in enumerate(train_pbar):
            
            # Move data to device
            data = move_data_to_device(data, device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data, training=True)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics (keep on GPU to avoid synchronization)
            train_losses.append(loss.detach().cpu())
            for key, value in outputs.items():
                if key not in train_metrics:
                    train_metrics[key] = []
                if isinstance(value, torch.Tensor):
                    train_metrics[key].append(value.detach().cpu())
                else:
                    train_metrics[key].append(value)
            
            global_step += 1
            
            # Log metrics periodically
            if global_step % log_iterations == 0:
                # Calculate average over recent iterations (convert to numpy on CPU)
                recent_iterations = min(log_iterations, len(train_losses))
                avg_metrics = {}
                for k, v in train_metrics.items():
                    if len(v) > 0:
                        # Convert tensors to scalars
                        v_numpy = [x.item() if isinstance(x, torch.Tensor) else x for x in v[-recent_iterations:]]
                        avg_metrics[f'{k}/train'] = np.mean(v_numpy)
                
                if wandb_run is not None:
                    wandb_run.log(avg_metrics, step=global_step)
                
                if enable_progress_bar:
                    train_pbar.set_postfix({k: f'{v:.4f}' for k, v in avg_metrics.items()})
                
                log(f'Step {global_step}: {avg_metrics}')
            
            # Save checkpoint every N steps (for training interruption recovery)
            if save_every_n_steps is not None and int(save_every_n_steps) > 0:
                if global_step % int(save_every_n_steps) == 0:
                    step_checkpoint_path = save_checkpoint(
                        model, optimizer, epoch, global_step, metrics_history,
                        checkpoints_dir, experiment, is_best=False, is_step_checkpoint=True
                    )
                    log(f'Saved step checkpoint: {step_checkpoint_path}')
                    
                    # Clean up old step checkpoints (keep only the most recent ones)
                    cleanup_old_step_checkpoints(checkpoints_dir, experiment, keep_top_k=keep_top_k_steps)
        
        # Calculate epoch training metrics (convert tensors to scalars)
        epoch_train_metrics = {}
        if train_metrics:
            for k, v in train_metrics.items():
                if len(v) > 0:
                    v_numpy = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
                    epoch_train_metrics[f'{k}/train'] = np.mean(v_numpy)
        
        # Validation phase
        model.eval()
        val_metrics = {}
        val_losses = []
        
        with torch.no_grad():
            if enable_progress_bar:
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')
            else:
                val_pbar = val_loader
            
            for data in val_pbar:
                # Move data to device
                data = move_data_to_device(data, device)
                
                # Forward pass
                outputs = model(data, training=False)
                loss = outputs['loss']
                
                # Update metrics (keep on GPU to avoid synchronization)
                val_losses.append(loss.detach().cpu())
                for key, value in outputs.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    if isinstance(value, torch.Tensor):
                        val_metrics[key].append(value.detach().cpu())
                    else:
                        val_metrics[key].append(value)
        
        # Calculate epoch validation metrics (convert tensors to scalars)
        epoch_val_metrics = {}
        if val_metrics:
            for k, v in val_metrics.items():
                if len(v) > 0:
                    v_numpy = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
                    epoch_val_metrics[f'{k}/val'] = np.mean(v_numpy)
        
        # Update metrics history
        for key, value in epoch_train_metrics.items():
            if key not in metrics_history:
                metrics_history[key] = []
            metrics_history[key].append(value)
        
        for key, value in epoch_val_metrics.items():
            if key not in metrics_history:
                metrics_history[key] = []
            metrics_history[key].append(value)
        
        # Log epoch metrics
        all_epoch_metrics = {**epoch_train_metrics, **epoch_val_metrics}
        if wandb_run is not None:
            wandb_run.log(all_epoch_metrics, step=global_step)
        
        log(f'Epoch {epoch+1}/{n_epochs}: {all_epoch_metrics}')
        
        # Save checkpoint
        val_loss = epoch_val_metrics.get('loss/val', None)
        is_best = False
        if val_loss is not None:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, global_step, metrics_history,
            checkpoints_dir, experiment, is_best=is_best
        )
        log(f'Saved checkpoint: {checkpoint_path}')
    
    print('Training completed!')
    if wandb_run is not None:
        wandb_run.finish()

