import torch
import torch.nn as nn
from src.evoformer import EvoformerStack, Linear
from tqdm import tqdm

class DiffusionTransitionMatrix(nn.Module):
    """Transition matrices for discrete diffusion process."""
    
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
        """Helper function to index into transition matrices."""
        # a: [T, num_classes, num_classes] - transition matrices
        # t: [B] - timestep
        # x: [B, N] - discrete class indices
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :] # (B, N, num_classes)


class DiscModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Extract parameters from kwargs with unified parameter mapping
        # Map diffusion_steps to timesteps for compatibility
        timesteps = kwargs.get('timesteps', kwargs.get('diffusion_steps', 1000))
        
        # D3PM specific parameters
        self.n_T = timesteps
        self.forward_type = kwargs.get('forward_type', 'uniform')
        self.eps = 1e-6
        
        # Create cosine noise schedule for beta values
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )
        
        # Create transition matrices for distance, ligand_atoms, and ligand_bonds
        self.num_classes = kwargs.get('no_dist_bins', 12)
        self.hmm_dist = DiffusionTransitionMatrix(
            num_classes=self.num_classes,
            timesteps=timesteps,
            beta_t=self.beta_t,
            forward_type=self.forward_type,
            eps=self.eps
        )
        
        self.num_ligand_atom_types = kwargs.get('no_ligand_atom_types', 19)
        self.hmm_ligand_atoms = DiffusionTransitionMatrix(
            num_classes=self.num_ligand_atom_types,
            timesteps=timesteps,
            beta_t=self.beta_t,
            forward_type=self.forward_type,
            eps=self.eps
        )
        
        self.num_ligand_bond_types = kwargs.get('no_ligand_bond_types', 5)
        self.hmm_ligand_bonds = DiffusionTransitionMatrix(
            num_classes=self.num_ligand_bond_types,
            timesteps=timesteps,
            beta_t=self.beta_t,
            forward_type=self.forward_type,
            eps=self.eps
        )
        
        # Create the Evoformer for distogram prediction
        self.evoformer = EvoformerStack(
            c_m=kwargs.get('c_m', 64),
            c_z=kwargs.get('c_z', 64),
            c_hidden_seq_att=kwargs.get('c_hidden_seq_att', 32),
            c_hidden_opm=kwargs.get('c_hidden_opm', 32),
            c_hidden_mul=kwargs.get('c_hidden_mul', 32),
            c_hidden_pair_att=kwargs.get('c_hidden_pair_att', 32),
            c_s=self.num_ligand_atom_types,
            no_heads_seq=kwargs.get('no_heads_seq', 8),
            no_heads_pair=kwargs.get('no_heads_pair', 4),
            no_blocks=kwargs.get('no_blocks', 4),
            transition_n=kwargs.get('transition_n', 4),
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', 4),
            inf=kwargs.get('inf', 1e9),
            eps=kwargs.get('eps', 1e-10),
        )
        
        # Embed seq: base seq_input_dim + atoms_onehot + seq_mask + seq_ligand_mask + time
        seq_input_dim_total = kwargs.get('seq_input_dim', 32) + self.num_ligand_atom_types + 3
        self.embed_seq = nn.Linear(seq_input_dim_total, kwargs.get('c_m', 64))
        
        # Persist bb_dist bins and embed z from [N, N, z + bb_dist_bins + num_classes + bonds + masks(2) + time] to [N, N, c_z]
        self.bb_dist_bins = kwargs.get('bb_dist_bins', 13)
        z_input_dim_total = (
            kwargs.get('z_input_dim', 1)
            + self.bb_dist_bins
            + self.num_classes
            + self.num_ligand_bond_types
            + 3  # pair_mask + pair_ligand_mask + time
        )
        self.embed_z = nn.Linear(z_input_dim_total, kwargs.get('c_z', 64))
        
        # Predict distogram logits from pair embedding z
        self.z_out = Linear(kwargs.get('c_z', 64), self.num_classes + self.num_ligand_bond_types, init="final")

    def forward(self, data, training=None):
        """Forward pass for training using D3PM discrete diffusion on DistAttDataset"""
        cond = {
            'seq': data['seq'], # [B, N, C_m] - positional encoding
            'z': data['z'], # [B, N, N, 1] - residue relationship matrix
            'bb_dist': data['bb_dist'], # [B, N, N] - backbone distance classes
            'seq_mask': data['seq_mask'], # [B, N] - sequence mask
            'pair_mask': data['pair_mask'], # [B, N, N] - pair mask
            'seq_ligand_mask': data['seq_ligand_mask'], # [B, N] - ligand atom mask
            'pair_ligand_mask': data['pair_ligand_mask'], # [B, N, N] - ligand bond mask
        }
        
        target = {
            'dist':  data['dist'], # [B, N, N] - target distance classes
            'atoms': data['ligand_atoms'], # [B, N] - target ligand atom classes
            'bonds': data['ligand_bonds'], # [B, N, N] - target ligand bond classes
        }
        
        b, n, _ = cond['seq'].shape
        
        # Convert dist_target to discrete classes (flatten to 1D for processing)
        x0 = {
            'dist': target['dist'].flatten(start_dim=1), # (B, N*N) - discrete distance class indices
            'atoms': target['atoms'].flatten(start_dim=1), # (B, N) - discrete ligand atom class indices
            'bonds': target['bonds'].flatten(start_dim=1), # (B, N*N) - discrete ligand bond class indices
        }
        
        # Sample random timestep for training
        t = torch.randint(1, self.n_T, (b,), device=x0['dist'].device) # [B] - timestep
        
        # Apply forward process to get noisy data
        xt = {
            'dist': self.q_sample(x0['dist'], t, torch.rand((*x0['dist'].shape, self.num_classes), device=x0['dist'].device), self.hmm_dist),
            'atoms': self.q_sample(x0['atoms'], t, torch.rand((*x0['atoms'].shape, self.num_ligand_atom_types), device=x0['atoms'].device), self.hmm_ligand_atoms),
            'bonds': self.q_sample(x0['bonds'], t, torch.rand((*x0['bonds'].shape, self.num_ligand_bond_types), device=x0['bonds'].device), self.hmm_ligand_bonds)
        }
        
        # Predict clean data from noisy data
        atoms_logits, z_logits = self.model_predict(xt, t, cond)
        bonds_logits = z_logits[:, :, :, self.num_classes:]
        dist_logits = z_logits[:, :, :, :self.num_classes]

        # Cross-entropy loss (direct prediction of clean data) - masked
        loss = {
            'dist': self.ce_masked(dist_logits.view(-1, self.num_classes), x0['dist'].view(-1), cond['pair_mask'].view(-1)),
            'atoms': self.ce_masked(atoms_logits.view(-1, self.num_ligand_atom_types), x0['atoms'].view(-1), cond['seq_ligand_mask'].view(-1)),
            'bonds': self.ce_masked(bonds_logits.view(-1, self.num_ligand_bond_types), x0['bonds'].view(-1), cond['pair_ligand_mask'].view(-1))
        }

        accuracy = {
            'dist': self.accuracy_masked(dist_logits.view(-1, self.num_classes), x0['dist'].view(-1), cond['pair_mask'].view(-1)),
            'atoms': self.accuracy_masked(atoms_logits.view(-1, self.num_ligand_atom_types), x0['atoms'].view(-1), cond['seq_ligand_mask'].view(-1)),
            'bonds': self.accuracy_masked(bonds_logits.view(-1, self.num_ligand_bond_types), x0['bonds'].view(-1), cond['pair_ligand_mask'].view(-1))
        }

        return {
            'loss': loss['dist'] + loss['atoms'] + loss['bonds'], 
            'dist_loss': loss['dist'], 
            'atoms_loss': loss['atoms'], 
            'bonds_loss': loss['bonds'], 
            'dist_accuracy': accuracy['dist'], 
            'atoms_accuracy': accuracy['atoms'], 
            'bonds_accuracy': accuracy['bonds']
        }

    def accuracy_masked(self, logits, targets, mask):
        """
        Compute accuracy.
        
        Args:
            logits: Predicted logits (-1, num_classes)
            targets: Target class indices (-1,)
            mask: Mask indicating valid elements (-1,)
        """
        correct = (logits.argmax(dim=-1) == targets) & mask.bool()
        return correct.sum().float() / mask.sum().float()

    @torch.no_grad()
    def sample_chain(self, data, keep_frames=None):
        """Sample from the reverse diffusion process using D3PM on DistAttDataset"""
            
        cond = {
            'seq': data['seq'], # [B, N, C_m] - positional encoding
            'z': data['z'], # [B, N, N, 1] - residue relationship matrix
            'bb_dist': data['bb_dist'], # [B, N, N] - backbone distance classes
            'seq_mask': data['seq_mask'], # [B, N] - sequence mask
            'pair_mask': data['pair_mask'], # [B, N, N] - pair mask
            'seq_ligand_mask': data['seq_ligand_mask'], # [B, N] - ligand atom mask
            'pair_ligand_mask': data['pair_ligand_mask'], # [B, N, N] - ligand bond mask
        }
        
        b, n, _ = cond['seq'].shape
        
        # Initialize with pure noise (random discrete classes)

        x = {
            'dist': torch.randint(0, self.num_classes, (b, n * n), device=cond['seq'].device),
            'atoms': torch.randint(0, self.num_ligand_atom_types, (b, n), device=cond['seq'].device),
            'bonds': torch.randint(0, self.num_ligand_bond_types, (b, n * n), device=cond['seq'].device),
        }
        
        # Store intermediate results
        chain = []
        
        # Sample p(x_{t-1} | x_t)
        for i, t in enumerate(tqdm(reversed(range(1, self.n_T)), desc="Diffusion sampling", total=self.n_T-1)):
            t_tensor = torch.full((b,), fill_value=t, device=cond['seq'].device) # [B] - timestep

            # Generate noise for Gumbel sampling
            noise = {
                'dist': torch.rand((b, n * n, self.num_classes), device=cond['seq'].device),
                'atoms': torch.rand((b, n, self.num_ligand_atom_types), device=cond['seq'].device),
                'bonds': torch.rand((b, n * n, self.num_ligand_bond_types), device=cond['seq'].device),
            }
            
            # Predict clean data from current noisy state
            atoms_logits, z_logits = self.model_predict(x, t_tensor, cond)
            bonds_logits = z_logits[:, :, :, self.num_classes:]
            dist_logits = z_logits[:, :, :, :self.num_classes]
            x = {
                'dist': self.p_sample(x['dist'], t_tensor, dist_logits, noise['dist'], self.hmm_dist),
                'atoms': self.p_sample(x['atoms'], t_tensor, atoms_logits, noise['atoms'], self.hmm_ligand_atoms),
                'bonds': self.p_sample(x['bonds'], t_tensor, bonds_logits, noise['bonds'], self.hmm_ligand_bonds),
            }

            chain.append(x)
        
        return x, chain


    def q_posterior_logits(self, x_0, x_t, t, hmm):
        """
        Compute the posterior logits q(x_{t-1} | x_t, x_0).
        
        This implements the analytical posterior distribution for discrete diffusion.
        The formula is: q(x_{t-1} | x_t, x_0) ∝ Q_t^T(x_t, x_{t-1}) * Q_bar_{t-1}(x_{t-1}, x_0)
        
        Args:
            x_0: Clean data (either one-hot or logits)
            x_t: Noisy data at timestep t
            t: Current timestep
            hmm: DiffusionTransitionMatrix instance (defaults to self.hmm_dist)
        
        Returns:
            Logits for the posterior distribution q(x_{t-1} | x_t, x_0)
        """
        num_classes = hmm.num_classes
        eps = hmm.eps
        
        # Convert discrete indices to logits if needed
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, num_classes) + eps
            )
        else:
            x_0_logits = x_0.clone()
        
        # Convert to probabilities for matrix multiplication
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        
        # Q_t^T(x_t, x_{t-1}) - probability of reaching x_t from x_{t-1}
        fact1 = hmm._at(hmm.q_one_step_transposed, t, x_t)
        
        # Q_bar_{t-1}(x_{t-1}, x_0) - probability of reaching x_0 from x_{t-1}
        # Handle t=1 case (no previous timestep)
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        t_is_1 = (t == 1).float().reshape((t.shape[0], *[1] * (x_t.dim())))
        
        # For t > 1, compute Q_bar_{t-1}
        qmats2 = hmm.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # softmaxed: [B, ..., C], qmats2: [B, C, C]
        fact2 = torch.matmul(softmaxed, qmats2)
        
        # Combine both factors in log space
        out = torch.log(fact1 + eps) + torch.log(fact2 + eps)
        
        # Special case for t=1: posterior is just the clean data
        return torch.where(t_is_1.bool(), x_0_logits, out)

    def ce_masked(self, logits, targets, mask):
        """
        Compute masked cross-entropy loss.
        
        Args:
            logits: Predicted logits (-1, num_classes)
            targets: Target class indices (-1,)
            mask: Mask indicating valid elements (-1,)
        
        Returns:
            Average cross-entropy loss over valid elements only
        """
        # Compute cross-entropy loss
        ce_per_edge = torch.nn.functional.cross_entropy(logits, targets, reduction='none') # (B*E,)
        
        # Apply mask and compute mean only over valid edges
        masked_ce = ce_per_edge * mask
        valid_count = mask.sum()
        
        if valid_count > 0:
            return masked_ce.sum() / valid_count
        else:
            return torch.tensor(0.0, device=logits.device)

    def q_sample(self, x0, t, noise, hmm):
        """
        Sample from the forward process q(xt | x0).
        
        This applies noise to clean data x0 to get noisy data xt at timestep t.
        Uses Gumbel-Max trick for discrete sampling.
        
        Args:
            x0: Clean discrete data (B, N) - discrete class indices
            t: Timestep [B] - timestep
            noise: Uniform random noise for Gumbel sampling (B, N, num_classes)
            hmm: DiffusionTransitionMatrix instance
        
        Returns:
            Noisy discrete data xt (B, N) - discrete class indices
        """
        eps = hmm.eps # 1e-6
        
        # Get transition probabilities from x0 to xt
        logits = torch.log(hmm._at(hmm.q_mats, t, x0) + eps) # (B, N, num_classes)
        
        # Clip noise to avoid numerical issues
        noise = torch.clip(noise, eps, 1.0) # (B, N, num_classes)
        
        # Apply Gumbel-Max trick for discrete sampling
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1) # (B, N) - discrete class indices

    def model_predict(self, xt, t, cond):
        """
        Predict clean data x_0 from noisy data x_t using Evoformer.
        
        Args:
            x_t_classes: Noisy discrete data (B, N*N) - discrete class indices
            t: Current timestep [B] - timestep
            seq: Sequence features [B, N, seq_input_dim]
            z: Pair features [B, N, N, z_input_dim]
            bb_dist: Backbone distance features [B, N, N]
            seq_mask: Sequence mask [B, N]
            pair_mask: Pair mask [B, N, N]
            seq_ligand_mask: Sequence ligand mask [B, N]
            pair_ligand_mask: Pair ligand mask [B, N, N]
            xt_atoms: Noisy discrete data (B, N) - discrete ligand atom class indices
            xt_dist: Noisy discrete data (B, N*N) - discrete distance class indices
            xt_bonds: Noisy discrete data (B, N*N) - discrete ligand bond class indices
        
        Returns:
            Predicted logits for clean data x_0
        """
        b, n, _ = cond['seq'].shape

        # Add time step to sequence features
        seq = cond['seq'] # [B, N, seq_input_dim]
        atoms_onehot = torch.nn.functional.one_hot(xt['atoms'], num_classes=self.num_ligand_atom_types).float()
        seq_mask = cond['seq_mask'][..., None] # [B, N, 1]
        seq_ligand_mask = cond['seq_ligand_mask'][..., None] # [B, N, 1]
        # Expand timestep to match seq feature shape for concat: [B] -> [B, N, 1]
        t_reshaped = t.view(b, 1, 1).expand(b, n, 1)
        seq = torch.cat([seq, atoms_onehot, seq_mask, seq_ligand_mask, t_reshaped], dim=-1) # [B, N, seq_input_dim + num_ligand_atom_types + 1 + 1 + 1]

        z = cond['z'] # [B, N, N, z_input_dim]
        bb_dist_onehot = torch.nn.functional.one_hot(cond['bb_dist'].long(), num_classes=self.bb_dist_bins).float() # [B, N, N, bb_dist_bins]
        dist_onehot = torch.nn.functional.one_hot(xt['dist'].view(b, n, n).long(), num_classes=self.num_classes).float() # [B, N, N, num_classes]
        bonds_onehot = torch.nn.functional.one_hot(xt['bonds'].view(b, n, n).long(), num_classes=self.num_ligand_bond_types).float() # [B, N, N, num_ligand_bond_types]
        pair_mask = cond['pair_mask'][..., None] # [B, N, N, 1]
        pair_ligand_mask = cond['pair_ligand_mask'][..., None] # [B, N, N, 1]
        # Expand timestep to match pair feature shape for concat: [B] -> [B, N, N, 1]
        t_reshaped = t.view(b, 1, 1, 1).expand(b, n, n, 1)
        z = torch.cat([z, bb_dist_onehot, dist_onehot, bonds_onehot, pair_mask, pair_ligand_mask, t_reshaped], dim=-1) # [B, N, N, z_input_dim + bb_dist_bins + num_classes + num_ligand_bond_types + 1 + 1 + 1]
                
        # Embed seq and z
        seq_embedded = self.embed_seq(seq)
        z_embedded = self.embed_z(z)
        
        # Run Evoformer
        _, z_out, seq_out = self.evoformer(
            seq=seq_embedded,
            z=z_embedded,
            seq_mask=cond['seq_mask'],
            pair_mask=cond['pair_mask'],
            chunk_size=4,
        )
        z_out = self.z_out(z_out) 
                
        return seq_out, z_out
    
    def p_sample(self, x, t, predicted_logits, noise, hmm):
        """
        Sample from the reverse process p_theta(x_{t-1} | x_t).
        
        This is the denoising step during generation. It predicts x_{t-1} from x_t
        using the learned model and Gumbel sampling for discrete data.
        
        Args:
            x: Current noisy data x_t
            t: Current timestep
            predicted_logits: Predicted logits for clean data x_0
            hmm: DiffusionTransitionMatrix instance
            noise: Uniform random noise for Gumbel sampling
        
        Returns:
            Sampled data x_{t-1}
        """
        
        # Compute posterior p_theta(x_{t-1} | x_t) using predicted x_0
        pred_q_posterior_logits = self.q_posterior_logits(predicted_logits, x, t, hmm) # (B, N, num_classes)
        
        # Clip noise to avoid numerical issues
        noise = torch.clip(noise, hmm.eps, 1.0) # (B, N, num_classes)
        
        # For t=1, don't add noise (direct sampling from predicted x_0)
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim()))) # (B, N, 1)
        
        # Apply Gumbel-Max trick for discrete sampling
        gumbel_noise = -torch.log(-torch.log(noise)) # (B, N, num_classes)
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        ) # (B, N) - discrete class indices
        return sample # (B, N) - discrete class indices





