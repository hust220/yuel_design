import pytorch_lightning as pl
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Optional

# Import dataset module for dynamic class loading
from . import datasets

class LightningWrapper(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

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
        """Dynamically get dataset class from datasets module"""
        if isinstance(dataset_class, type) and issubclass(dataset_class, torch.utils.data.Dataset):
            return dataset_class
        elif isinstance(dataset_class, str) and hasattr(datasets, dataset_class):
            return getattr(datasets, dataset_class)
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
                from src.datasets import collate
                collate_fn = collate
        from src.datasets import get_dataloader
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, collate_fn=None):
        if collate_fn is None:
            dataset_collate = getattr(self.dataset_class, 'collate_fn', None)
            if callable(dataset_collate):
                collate_fn = dataset_collate
            else:
                from src.datasets import collate
                collate_fn = collate
        from src.datasets import get_dataloader
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn, num_workers=self.num_workers)

    def test_dataloader(self, collate_fn=None):
        if collate_fn is None:
            dataset_collate = getattr(self.dataset_class, 'collate_fn', None)
            if callable(dataset_collate):
                collate_fn = dataset_collate
            else:
                from src.datasets import collate
                collate_fn = collate
        from src.datasets import get_dataloader
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
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
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
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=False)

        self.validation_step_outputs = []

    def sample_chain(self, **kwargs):
        return self.model.sample_chain(**kwargs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
