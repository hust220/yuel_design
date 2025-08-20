import numpy as np
import pytorch_lightning as pl
import torch

from src import metrics, utils
from src.const import N_RESIDUE_TYPES
from src.egnn import Dynamics
from src.edm import EDM
from src.datasets import (
    ProteinLigandDataset, create_templates_for_generation, get_dataloader, collate, molecule_feat_mask
)
from src.molecule_builder import build_molecules
from typing import Dict, List, Optional
from tqdm import tqdm

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, model,
        data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False,
        center_of_mass='protein', graph_type=None,
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics = Dynamics(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=False,
            graph_type=graph_type,
        )
        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )

    def update_device(self, device):
        """Update the device for the model and its components"""
        self.torch_device = device
        if hasattr(self.edm, 'dynamics'):
            self.edm.dynamics.update_device(device)

    def setup(self, stage: Optional[str] = None):
        dataset_type = ProteinLigandDataset
        if stage == 'fit':
            self.train_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.val_dataset = dataset_type(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self, collate_fn=collate):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    def forward(self, data, training):
        x = data['positions']
        h = data['one_hot']
        node_mask = data['atom_mask']
        edge_mask = data['edge_mask']
        protein_mask = data['protein_mask']
        ligand_mask = data['ligand_mask']

        feat_mask = torch.tensor(molecule_feat_mask(), device=x.device)

        context = protein_mask

        center_of_mass_mask = protein_mask
        # if center_of_mass_mask[i] has 0 proteins, we need to set it to 1
        center_of_mass_mask = torch.where(center_of_mass_mask.sum(dim=1, keepdim=True) == 0, torch.ones_like(center_of_mass_mask), center_of_mass_mask)
        center_of_mass_mask *= node_mask

        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

        # Applying random rotation
        if training and self.data_augmentation:
            x = utils.random_rotation(x)

        return self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
            edge_mask=edge_mask,
            feat_mask=feat_mask,
            context=context
        )

    def training_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=True)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss,
        }
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                self.log(f'{metric_name}/train', metric, prog_bar=True)
        # self.training_step_outputs.append(training_metrics)
        return training_metrics

    def validation_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        rt = {
            'loss': loss,
        }
        self.validation_step_outputs.append(rt)
        return rt

    def test_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        rt = {
            'loss': loss,
        }
        self.test_step_outputs.append(rt)
        return rt

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.val_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=False)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            # Logging the results corresponding to the best validation_and_connectivity
            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=False, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=False, batch_size=self.batch_size)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.test_dataloader())
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/test', metric_value, prog_bar=False)
                self.metrics.setdefault(f'{metric_name}/test', []).append(metric_value)

        self.test_step_outputs = []

    def sample_and_analyze(self, dataloader):
        pred_molecules = []
        true_molecules = []

        # self.n_stability_samples = 1
        bar = tqdm(total=len(dataloader) * self.n_stability_samples, desc='Sampling')
        for b, data in enumerate(dataloader):
            true_molecules_batch = build_molecules(
                data['one_hot'][:, :, N_RESIDUE_TYPES:],
                data['positions'],
                data['ligand_mask'],
            )

            for _ in range(self.n_stability_samples):
                try:
                    chain_batch, node_mask = self.sample_chain(data, keep_frames=self.FRAMES)
                except utils.FoundNaNException as e:
                    for idx in e.x_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [xh], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_x_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [x ], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [ h], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    continue

                # Get final molecules from chains – for computing metrics
                x, h = utils.split_features(
                    z=chain_batch[0],
                    n_dims=self.n_dims,
                    num_classes=self.num_classes,
                )

                one_hot = h['categorical'][:, :, N_RESIDUE_TYPES:]
                node_mask = data['ligand_mask']
                pred_molecules_batch = build_molecules(one_hot, x, node_mask)

                # Adding only results for valid ground truth molecules
                for pred_mol, true_mol in zip(pred_molecules_batch, true_molecules_batch):
#                    print('true_mol', metrics.is_valid(true_mol))
                    if metrics.is_valid(true_mol):
                        pred_molecules.append(pred_mol)
                        true_molecules.append(true_mol)

                bar.update(1)

        our_metrics = metrics.compute_metrics(
            pred_molecules=pred_molecules,
            true_molecules=true_molecules
        )
        return {
            **our_metrics,
        }

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            ligand_sizes = data['ligand_mask'].sum(1).view(-1).int()
        else:
            ligand_sizes = sample_fn(data)

        template_data = create_templates_for_generation(data, ligand_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        protein_mask = template_data['protein_mask']
        ligand_mask = template_data['ligand_mask']

#        print(x.shape, h.shape, node_mask.shape, edge_mask.shape, protein_mask.shape, ligand_mask.shape)

        context = protein_mask

        center_of_mass_mask = protein_mask
        center_of_mass_mask = torch.where(center_of_mass_mask.sum(dim=1, keepdim=True) == 0, torch.ones_like(center_of_mass_mask), center_of_mass_mask)
        center_of_mass_mask *= node_mask

#        print(1)

        # x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        x_masked = x * center_of_mass_mask # x: b,n,3
        N = center_of_mass_mask.sum(1, keepdims=True)
        mean = torch.sum(x_masked, dim=1, keepdim=True) / N # shape of mean: b,1,3
        mean = mean * node_mask
        # print(f'mean {mean}')
        x = x - mean
#        print(2)

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
            context=context,
            keep_frames=keep_frames,
        )
        chain[:,:,:,:self.n_dims] = chain[:,:,:,:self.n_dims] + mean

#        print('Chain', chain.shape)
        return chain, node_mask

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
