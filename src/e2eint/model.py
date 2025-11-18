# Third-party imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from tqdm import tqdm
import numpy as np

# Local imports
from src.e3former import E3former

class DiffusionTransitionMatrix(nn.Module):
    
    def __init__(self, num_classes, timesteps, beta_t, forward_type='uniform', eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.n_T = timesteps
        self.forward_type = forward_type
        self.eps = eps
        
        q_onestep_mats = []
        for beta in beta_t:
            if self.forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes, dtype=torch.float64) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)
        
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)
    
    def _at(self, a, t, x):
        # a: [T, num_classes, num_classes] - transition matrices
        # t: [B] - timestep
        # x: [B, N] - discrete class indices
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]
    
    def q_sample(self, x0, t, noise=None):
        logits = torch.log(self._at(self.q_mats, t, x0) + self.eps)  # [B, N, num_classes]
        if noise is None:
            noise = torch.rand(*logits.shape, device=x0.device)  # [B, N, num_classes]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)
    
    def q_posterior_logits(self, x_0, x_t, t):
        num_classes = self.num_classes
        eps = self.eps
        
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, num_classes) + eps
            )
        else:
            x_0_logits = x_0.clone()
        
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        fact1 = self._at(self.q_one_step_transposed, t, x_t)
        
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        t_is_1 = (t == 1).float().reshape((t.shape[0], *[1] * (x_t.dim())))
        
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        fact2 = torch.matmul(softmaxed, qmats2)
        
        out = torch.log(fact1 + eps) + torch.log(fact2 + eps)
        
        return torch.where(t_is_1.bool(), x_0_logits, out)
    
    def p_sample(self, x, t, predicted_logits, noise=None):
        pred_q_posterior_logits = self.q_posterior_logits(predicted_logits, x, t)
        if noise is None:
            noise = torch.rand(*pred_q_posterior_logits.shape, device=x.device)
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

class PredefinedNoiseSchedule(torch.nn.Module):
    """Predefined noise schedule for polynomial_2. Accepts Python scalar input."""
    def __init__(self, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        # Compute polynomial schedule: alphas2 = (1 - (x/T)^2)^2
        x = np.linspace(0, timesteps + 1, timesteps + 1)
        alphas2 = (1 - (x / (timesteps + 1)) ** 2) ** 2
        
        # Clip steps and apply precision
        alphas2_with_1 = np.concatenate([np.ones(1), alphas2])
        steps = np.clip(alphas2_with_1[1:] / alphas2_with_1[:-1], 0.001, 1.0)
        alphas2 = np.cumprod(steps) * (1 - 2 * precision) + precision
        
        # Compute gamma = -log(alpha^2 / sigma^2)
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-(np.log(alphas2) - np.log(1 - alphas2))).float(),
            requires_grad=False)

    def forward(self, t):
        """t: [batch] tensor - normalized time step [0, 1] for each sample"""
        t_float = t.float() if isinstance(t, torch.Tensor) else torch.tensor([t])
        idx = (t_float).round().long().clamp(0, len(self.gamma) - 1)
        return self.gamma[idx]  # [batch]

class E2EModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Extract parameters with defaults
        timesteps = kwargs.get('diffusion_steps', 100)
        noise_precision = kwargs.get('diffusion_noise_precision', 1e-5)
        self.gamma = PredefinedNoiseSchedule(timesteps=timesteps, precision=noise_precision)

        self.T = timesteps
        self.t_weight_power = kwargs.get('t_weight_power', 0.0)

        self.n_blocks = kwargs['n_blocks']

        hidden_nf = kwargs['hidden_nf']
        in_node_features = kwargs['in_node_features']
        in_edge_features = kwargs['in_edge_features']
        
        # D3PM for ligand atom types
        self.num_ligand_atom_types = kwargs.get('num_ligand_atom_types', 20)
        
        # E3former initialization
        # seq_input_dim = in_node_features + num_ligand_atom_types (atom_onehot) + 2 (anchor_mask and time)
        # z_input_dim = 1 (is_same_residue only, no distance)
        # c_s = num_ligand_atom_types: s_out will directly output atom type logits
        self.e3former = E3former(
            seq_input_dim=in_node_features + self.num_ligand_atom_types + 2,  # +atom_onehot + anchor_mask + time
            z_input_dim=in_edge_features,
            c_m=hidden_nf,
            c_z=hidden_nf,
            c_s=self.num_ligand_atom_types,  # s_out will be atom type logits
            c_hidden_seq_att=kwargs.get('c_hidden_seq_att', hidden_nf // 2),
            c_hidden_opm=kwargs.get('c_hidden_opm', hidden_nf // 2),
            c_hidden_mul=kwargs.get('c_hidden_mul', hidden_nf // 2),
            c_hidden_pair_att=kwargs.get('c_hidden_pair_att', hidden_nf // 2),
            no_heads_seq=kwargs.get('no_heads_seq', 8),
            no_heads_pair=kwargs.get('no_heads_pair', 4),
            no_blocks=self.n_blocks,
            transition_n=kwargs.get('transition_n', 4),
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', self.n_blocks),
            eps=kwargs.get('eps', 1e-8),
        )
        
        self.chunk_size = kwargs.get('chunk_size', None)
        beta_t = self.create_beta_schedule(timesteps)
        self.hmm_ligand_atoms = DiffusionTransitionMatrix(
            num_classes=self.num_ligand_atom_types,
            timesteps=timesteps,
            beta_t=beta_t,
            forward_type='uniform',
            eps=1e-6
        )
    
    def create_beta_schedule(self, timesteps):
        """Create cosine noise schedule for beta values"""
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )
        return beta_t     

    def sample_weighted_t(self):
        # Compute weights and probabilities
        t_weights = np.arange(1, self.T + 1, dtype=np.float32) ** self.t_weight_power
        t_probs = t_weights / t_weights.sum()
        
        # Use numpy for weighted sampling
        t_index = np.random.choice(self.T, p=t_probs)
        return t_index + 1  # +1 because index starts from 0 but t starts from 1

    def model_predict(self, xt, t, data):
        """Helper function to run E3former forward pass
        
        Args:
            xt: dict with keys:
                - 'coords': [batch_size, n, 3] - noisy coordinates
                - 'atoms': [batch_size, n] - noisy atom type indices
            t: [batch_size] tensor - normalized time step for each sample
            data: dict with batched tensors (batch_size, n, ...)
        
        Returns:
            s_out: [batch_size, n, hidden_nf] - sequence output
            dx: [batch_size, n, 3] - predicted coordinate update
        """
        t = t / self.T
        
        h = data['h']  # [batch_size, n, h_dim]
        z = data['z']  # [batch_size, n, n, 1] - [is_same_residue]

        receptor_mask = data['receptor_mask'].unsqueeze(-1)  # [batch_size, n, 1]
        seq_mask = data['seq_mask']  # [batch_size, n]
        pair_mask = data['pair_mask']  # [batch_size, n, n]
        anchor_mask = data['anchor_mask'].unsqueeze(-1)  # [batch_size, n, 1]

        ligand_mask = (1 - receptor_mask) * seq_mask.unsqueeze(-1)  # [batch_size, n, 1]

        coords = xt['coords']  # [batch_size, n, 3]
        atoms = xt['atoms']  # [batch_size, n]

        batch_size, n_nodes = h.shape[0], h.shape[1]
        
        # Create one-hot encoding for atom types
        atom_onehot = F.one_hot(atoms, num_classes=self.num_ligand_atom_types).float()  # [batch_size, n, num_ligand_atom_types]
        atom_onehot = atom_onehot * ligand_mask  # [batch_size, n, num_ligand_atom_types]
                
        t_expanded = t.unsqueeze(-1).unsqueeze(-1).expand(batch_size, n_nodes, 1)
        
        seq = torch.cat([h, atom_onehot, anchor_mask, t_expanded], dim=-1)  # [batch_size, n, h_dim + num_ligand_atom_types + 1 + 1]
        
        x_in = coords.clone()  # Save input coordinates
        
        # E3former forward
        s_out, x_out = self.e3former.forward(
            seq=seq,
            x=x_in,
            z=z,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            chunk_size=self.chunk_size,
        )
        
        # Return coordinate update
        dx = x_out - x_in
        
        return s_out, dx

    def forward(self, data, training=None):
        """Forward pass for training with continuous and discrete diffusion"""        
        
        seq_mask = data['seq_mask'].unsqueeze(-1)  # [B, N, 1]
        anchor_mask = data['anchor_mask'].unsqueeze(-1)  # [B, N, 1]
        receptor_mask = data['receptor_mask'].unsqueeze(-1)  # [B, N, 1]
        ligand_mask = (1 - receptor_mask) * seq_mask  # [B, N, 1] - ligand mask
        sample_mask = (1 - anchor_mask) * seq_mask  # [B, N, 1] - sample mask

        anchor_coords = data['x'] * anchor_mask  # [B, N, 3]
        anchor_mean = anchor_coords.sum(dim=1, keepdim=True) / anchor_mask.sum(dim=1, keepdim=True)  # [B, 1, 3]
        
        target = {
            'coords': data['x'] - anchor_mean,  # [B, N, 3] - centered coordinates
            'atoms': data['ligand_atoms'],  # [B, N]
        }
        
        batch_size, n_nodes = data['h'].shape[0], data['h'].shape[1]
        device = target['coords'].device
        
        # Sample timestep
        t = self.sample_weighted_t()  # [1] - timestep
        t_tensor = torch.full((batch_size,), t, dtype=torch.float32, device=device)  # [B] - timestep
        
        # === Continuous diffusion: coordinates ===
        gamma_t = self.gamma(t)  # [B]
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t)).view(-1, 1, 1)  # [B, 1, 1]
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t)).view(-1, 1, 1)  # [B, 1, 1]
        eps_sample = torch.randn_like(target['coords']) * sample_mask
        noisy_coords = alpha_t * target['coords'] + sigma_t * eps_sample

        # === Discrete diffusion: atom types ===
        noisy_atoms = self.hmm_ligand_atoms.q_sample(target['atoms'], t_tensor.long())
        noisy_atoms = (noisy_atoms * ligand_mask.squeeze(-1)).long()

        xt = {
            'coords': noisy_coords,
            'atoms': noisy_atoms,
        }
        
        s_out, pred_eps_coords = self.model_predict(xt, t_tensor, data)
        x0_pred = (noisy_coords - sigma_t * pred_eps_coords) / alpha_t
        
        loss = {
            'coords': self.coord_loss_fn(pred_eps_coords, eps_sample, sample_mask),
            'atoms': self.atom_loss_fn(s_out, target['atoms'], ligand_mask),
            'interaction': self.interaction_loss_fn(x0_pred, data, receptor_mask, ligand_mask),
        }
        
        return {
            'loss': loss['coords'] + loss['atoms'] + loss['interaction'],
            'coord_loss': loss['coords'],
            'atom_loss': loss['atoms'],
            'interaction_loss': loss['interaction'],
        }
    
    def coord_loss_fn(self, pred, target, mask):
        """Compute coordinate loss (MSE) with mask"""
        loss = ((pred - target) ** 2) * mask
        loss = loss.sum() / mask.sum().clamp(min=1)
        return loss
    
    def atom_loss_fn(self, logits, target, mask):
        """Compute atom type loss (CE) with mask"""
        loss = F.cross_entropy(
            logits.view(-1, self.num_ligand_atom_types),
            target.view(-1),
            reduction='none'
        )
        loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
        return loss
    
    def interaction_loss_fn(self, x0_pred, data, receptor_mask, ligand_mask):
        """Compute interaction loss: check if receptor_interaction atoms are within 5A of ligand
        
        Args:
            x0_pred: [B, N, 3] - predicted coordinates (uncentered)
            data: dict containing 'receptor_interaction' [B, N] - 1 for interacting receptor atoms
            receptor_mask: [B, N, 1] - 1 for receptor atoms
            ligand_mask: [B, N, 1] - 1 for ligand atoms
        
        Returns:
            Loss value (penalty if interacting receptor atoms are too far from ligand)
        """
        if 'receptor_interaction' not in data:
            return torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)
        
        receptor_interaction = data['receptor_interaction']  # [B, N]
        receptor_mask_1d = receptor_mask.squeeze(-1)  # [B, N]
        ligand_mask_1d = ligand_mask.squeeze(-1)  # [B, N]
        
        # Find interacting receptor atoms and ligand atoms
        interacting_receptor = (receptor_interaction == 1) & (receptor_mask_1d == 1)  # [B, N]
        ligand_atoms = ligand_mask_1d == 1  # [B, N]
        
        # Compute pairwise distances: [B, N, N]
        coords_i = x0_pred.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = x0_pred.unsqueeze(1)  # [B, 1, N, 3]
        distances = torch.sqrt(((coords_i - coords_j) ** 2).sum(dim=-1) + 1e-8)  # [B, N, N]
        
        # For each interacting receptor atom, find minimum distance to any ligand atom
        # interacting_receptor: [B, N] -> [B, N, 1]
        # ligand_atoms: [B, N] -> [B, 1, N]
        # Mask: only distances from interacting receptor to ligand atoms
        rec_lig_distances = distances.clone()  # [B, N, N]
        
        # For rows (interacting receptor atoms), only consider columns (ligand atoms)
        # Set distances to non-ligand atoms to inf for interacting receptor atoms
        ligand_mask_expanded = ligand_atoms.unsqueeze(1).float()  # [B, 1, N]
        rec_lig_distances = torch.where(
            ligand_mask_expanded > 0.5,
            rec_lig_distances,
            torch.tensor(float('inf'), device=distances.device, dtype=distances.dtype)
        )  # [B, N, N]
        
        # Minimum distance for each atom to any ligand atom: [B, N]
        min_distances_to_ligand = rec_lig_distances.min(dim=2)[0]  # [B, N]
        
        # Only consider interacting receptor atoms
        # For interacting receptor atoms, get their min distance; for others, set to 0
        min_distances_interacting = torch.where(
            interacting_receptor,
            min_distances_to_ligand,
            torch.tensor(0.0, device=min_distances_to_ligand.device, dtype=min_distances_to_ligand.dtype)
        )  # [B, N]
        
        # Penalty if minimum distance > 5.0 (using ReLU: max(0, min_dist - 5.0))
        penalty = F.relu(min_distances_interacting - 5.0)  # [B, N]
        
        # Average penalty over all interacting receptor atoms
        n_interacting = interacting_receptor.sum()  # total number of interacting receptor atoms across batch
        if n_interacting > 0:
            return penalty.sum() / n_interacting
        else:
            return torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

    @torch.no_grad()
    def sample_chain(self, data, keep_frames=None):
        """Unified sample_chain method that handles unbatched data (single sample)"""
        batch_size, n_nodes = data['h'].shape[0], data['h'].shape[1]

        seq_mask = data['seq_mask'].unsqueeze(-1)  # [B, N, 1] - valid atoms mask (all 1s for single sample)
        receptor_mask = data['receptor_mask'].unsqueeze(-1)  # [B, N, 1] - receptor mask
        anchor_mask = data['anchor_mask'].unsqueeze(-1)  # [B, N, 1]
        ligand_mask = (1 - receptor_mask) * seq_mask  # [B, N, 1] - ligand mask
        sample_mask = (1 - anchor_mask) * seq_mask  # [B, N, 1] - sample mask

        x0 = data['x']  # [B, N, 3]
        anchor_mean = (x0 * anchor_mask).sum(dim=1, keepdim=True) / anchor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        x0_centered = x0 - anchor_mean
        x = x0_centered * anchor_mask + torch.randn_like(x0_centered) * sample_mask
        
        atoms = torch.randint(0, self.num_ligand_atom_types, (batch_size, n_nodes), device=x0.device) * ligand_mask.squeeze(-1)

        chain = []

        xt = {
            'coords': x,
            'atoms': atoms,
        }
        
        # Store centered x0 for anchor reference in each step
        data_with_x0 = {**data, 'x0_centered': x0_centered}

        # Sample p(z_s | z_t) - treat as single graph
        for t in tqdm(reversed(range(1, self.T + 1)), desc="Diffusion sampling", total=self.T):
            t_tensor = torch.full((batch_size,), t, dtype=torch.float32, device=x.device)  # [1]
            xt = self.sample_step(t_tensor, xt, data_with_x0)
            # Clone tensors to avoid reference issues in chain
            chain.append({
                'coords': xt['coords'].clone(),
                'atoms': xt['atoms'].clone(),
            })

        return chain[-1], chain

    def sample_step(self, t_tensor, xt, data):
        seq_mask = data['seq_mask'].unsqueeze(-1)  # [B, N, 1]
        anchor_mask = data['anchor_mask'].unsqueeze(-1)  # [B, N, 1]
        sample_mask = (1 - anchor_mask) * seq_mask  # [B, N, 1] - sample mask
        
        # Get centered x0 for anchor
        x0_centered = data['x0_centered']  # [B, N, 3]
        
        # For model input: anchor part should be alpha_t * x0, sample part uses xt
        alpha_t = torch.sqrt(torch.sigmoid(-self.gamma(t_tensor))).view(-1, 1, 1)  # [B, 1, 1]
        xt_input = {
            'coords': alpha_t * x0_centered * anchor_mask + xt['coords'] * sample_mask,
            'atoms': xt['atoms'],
        }

        atoms_logits, eps_hat_coords = self.model_predict(xt_input, t_tensor, data) # [B, N, num_ligand_atom_types], [B, N, 3]
        eps_hat_coords = eps_hat_coords * sample_mask

        s_tensor = t_tensor - 1
        gamma_s = self.gamma(s_tensor).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        gamma_t = self.gamma(t_tensor).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))  # [B, 1, 1]
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)  # [B, 1, 1]
        alpha_t_given_s = torch.exp(0.5 * (F.logsigmoid(-gamma_t) - F.logsigmoid(-gamma_s)))  # [B, 1, 1]
        sigma_s = torch.sqrt(torch.sigmoid(gamma_s))  # [B, 1, 1]
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))  # [B, 1, 1]
        
        # Compute mu for sample part
        mu_coords = xt['coords'] / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat_coords  # [B, N, 3]
        sigma_sampling = sigma_t_given_s * sigma_s / sigma_t  # [B, 1, 1]
        
        # For next step s: anchor part should be alpha_s * x0, sample part uses reverse diffusion
        alpha_s = torch.sqrt(torch.sigmoid(-gamma_s))  # [B, 1, 1]
        xs_coords = mu_coords * sample_mask + alpha_s * x0_centered * anchor_mask
        xs_coords = xs_coords + sigma_sampling * torch.randn_like(mu_coords) * sample_mask  # [B, N, 3]

        xs_atoms = self.hmm_ligand_atoms.p_sample(
            xt['atoms'],  # [B, N]
            t_tensor.long(),  # [B]
            atoms_logits,  # [B, N, num_ligand_atom_types]
        )  # [B, N]

        return {
            'coords': xs_coords,
            'atoms': xs_atoms,
        }

    







