import pytorch_lightning as pl
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional

from torch.utils.data import Dataset, DataLoader

def get_dataloader(dataset, batch_size, collate_fn, shuffle=False, num_workers=0, device=None):
    return DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

class LightningWrapper(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    # Note: metrics tracking is handled by Lightning's self.log() - no need to manually store

    def __init__(self, model_class, **kwargs):
        super(LightningWrapper, self).__init__()

        self.save_hyperparameters(ignore=['config', 'torch_device'])
        self.dataset_class = self._get_dataset_class(kwargs['dataset_class'])
        self.batch_size = kwargs['batch_size']
        self.log_iterations = kwargs.get('log_iterations', 20)
        self.lr = kwargs['lr']
        self.num_workers = kwargs.get('num_workers', 0)
        self.validation_step_outputs = []
        self.model = model_class(**kwargs)

    def _get_dataset_class(self, dataset_class):
        if isinstance(dataset_class, type) and issubclass(dataset_class, torch.utils.data.Dataset):
            return dataset_class
        else:
            raise ValueError(f"Invalid dataset class: {dataset_class}")

    def setup(self, stage: Optional[str] = None):
        # Get all hyperparameters and filter out dataset-specific ones
        dataset_params = self._filter_dataset_params()
        
        # Use CPU for data loading to reduce GPU memory usage
        self.train_dataset = self.dataset_class(split='train', **dataset_params)
        self.val_dataset = self.dataset_class(split='val', **dataset_params)
    
    def _filter_dataset_params(self):
        """Filter hyperparameters to only include dataset-relevant ones"""
        import inspect
        sig = inspect.signature(self.dataset_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'split', 'device'}  # Exclude common params
        
        return {k: v for k, v in self.hparams.items() if k in valid_params}

    def train_dataloader(self, collate_fn=None):
        if collate_fn is None:
            dataset_collate = getattr(self.dataset_class, 'collate_fn', None)
            if callable(dataset_collate):
                collate_fn = dataset_collate
            else:
                raise ValueError(f"No collate function found for dataset class: {self.dataset_class}")
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, collate_fn=None):
        if collate_fn is None:
            dataset_collate = getattr(self.dataset_class, 'collate_fn', None)
            if callable(dataset_collate):
                collate_fn = dataset_collate
            else:
                raise ValueError(f"No collate function found for dataset class: {self.dataset_class}")
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers)

    def test_dataloader(self, collate_fn=None):
        if collate_fn is None:
            dataset_collate = getattr(self.dataset_class, 'collate_fn', None)
            if callable(dataset_collate):
                collate_fn = dataset_collate
            else:
                raise ValueError(f"No collate function found for dataset class: {self.dataset_class}")
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers)

    def forward(self, data, training=None):
        return self.model.forward(data, training)
    
    def _move_data_to_device(self, data):
        """Move data from CPU to target device if needed"""
        from src.utils import move_data_to_device
        return move_data_to_device(data, self.device)

    def training_step(self, data, *args):
        # Move data to target device if it's on CPU
        data = self._move_data_to_device(data)
        training_metrics = self.forward(data, training=True)

        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                # Lightning's self.log() automatically tracks metrics - no need to manually store
                prog_bar = True if metric_name == 'loss' else False
                self.log(f'{metric_name}/train', metric, prog_bar=prog_bar)

        return training_metrics

    def validation_step(self, data, *args):
        # Move data to target device if it's on CPU
        data = self._move_data_to_device(data)
        validation_metrics = self.forward(data, training=False)
        self.validation_step_outputs.append(validation_metrics)
        return validation_metrics

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            # Lightning's self.log() automatically tracks metrics - no need to manually store
            self.log(f'{metric}/val', avg_metric, prog_bar=False)

        self.validation_step_outputs = []

    def sample_chain(self, **kwargs):
        return self.model.sample_chain(**kwargs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        # Use torch.stack to avoid creating intermediate list and potential synchronization
        return torch.stack([out[metric] for out in step_outputs]).mean()
