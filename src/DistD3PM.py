import torch
import torch.nn as nn
from .evoformer import EvoformerStack, Linear

def add_time_step(t, x, node_mask):
    # t: (B)
    # x: (B, N, D) for seq features or (B, N, N, D) for pair features
    if x.dim() == 3:
        # Sequence features: (B, N, D)
        b, n, d = x.shape
        
        # expand t to (B, N, 1)
        if isinstance(t, int):
            t = torch.full((b, n, 1), fill_value=t, device=x.device)
        else:
            # Handle different t shapes
            if t.dim() == 0:  # scalar tensor
                t = torch.full((b, n, 1), fill_value=t.item(), device=x.device)
            elif t.shape == torch.Size([b]):  # (B,) shape
                t = t.unsqueeze(1).unsqueeze(2).expand(b, n, 1)  # (B, 1, 1) -> (B, N, 1)
            elif t.shape == torch.Size([b, n]):  # (B, N) shape
                t = t.unsqueeze(2)  # (B, N, 1)
            elif t.shape == torch.Size([b, n, 1]):  # already correct shape
                pass
            elif t.shape == torch.Size([b, 1]):  # (B, 1) shape - common case
                t = t.unsqueeze(1).expand(b, n, 1)  # (B, 1, 1) -> (B, N, 1)
            else:
                # Try to reshape if possible
                print(f"Unknown t shape: {t.shape}")
                t = t.view(b, n, 1)
        
        # Concatenate time step with features
        x_with_time = torch.cat([t, x], dim=2)  # (B, N, D+1)
        
        # Apply mask: expand node_mask to match the new dimension
        if node_mask is not None:
            # node_mask: (B, N) -> (B, N, 1) -> (B, N, D+1)
            mask_expanded = node_mask.unsqueeze(-1).expand_as(x_with_time)
            return x_with_time * mask_expanded
        else:
            return x_with_time
            
    elif x.dim() == 4:
        # Pair features: (B, N, N, D)
        b, n, n2, d = x.shape
        
        # expand t to (B, N, N, 1)
        if isinstance(t, int):
            t = torch.full((b, n, n2, 1), fill_value=t, device=x.device)
        else:
            # Handle different t shapes
            if t.dim() == 0:  # scalar tensor
                t = torch.full((b, n, n2, 1), fill_value=t.item(), device=x.device)
            elif t.shape == torch.Size([b]):  # (B,) shape
                t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(b, n, n2, 1)  # (B, 1, 1, 1) -> (B, N, N, 1)
            elif t.shape == torch.Size([b, n]):  # (B, N) shape
                t = t.unsqueeze(2).unsqueeze(3).expand(b, n, n2, 1)  # (B, N, 1, 1) -> (B, N, N, 1)
            elif t.shape == torch.Size([b, n, n2]):  # (B, N, N) shape
                t = t.unsqueeze(3)  # (B, N, N, 1)
            elif t.shape == torch.Size([b, n, n2, 1]):  # already correct shape
                pass
            elif t.shape == torch.Size([b, 1]):  # (B, 1) shape - common case
                t = t.unsqueeze(1).unsqueeze(2).expand(b, n, n2, 1)  # (B, 1, 1, 1) -> (B, N, N, 1)
            else:
                # Try to reshape if possible
                print(f"Unknown t shape: {t.shape}")
                t = t.view(b, n, n2, 1)
        
        # Concatenate time step with features
        x_with_time = torch.cat([t, x], dim=3)  # (B, N, N, D+1)
        
        # Apply mask: expand pair_mask to match the new dimension
        if node_mask is not None:
            # pair_mask: (B, N, N) -> (B, N, N, 1) -> (B, N, N, D+1)
            mask_expanded = node_mask.unsqueeze(-1).expand_as(x_with_time)
            return x_with_time * mask_expanded
        else:
            return x_with_time
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")


class D3PM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Extract parameters from kwargs with unified parameter mapping
        # Map diffusion_steps to timesteps for compatibility
        timesteps = kwargs.get('timesteps', kwargs.get('diffusion_steps', 1000))
        self.num_classes = kwargs.get('no_dist_bins', 12)
        
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
        
        # Precompute forward process transition matrices
        q_onestep_mats = []
        for beta in self.beta_t:
            if self.forward_type == "uniform":
                mat = torch.ones(self.num_classes, self.num_classes) * beta / self.num_classes
                mat.diagonal().fill_(1 - (self.num_classes - 1) * beta / self.num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        
        # Stack all one-step transition matrices
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)
        
        # Precompute cumulative transition matrices Q_bar_t
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        
        # Register as buffers
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)
        
        # Create the Evoformer for distogram prediction
        self.evoformer = EvoformerStack(
            c_m=kwargs.get('c_m', 64),
            c_z=kwargs.get('c_z', 64),
            c_hidden_seq_att=kwargs.get('c_hidden_seq_att', 32),
            c_hidden_opm=kwargs.get('c_hidden_opm', 32),
            c_hidden_mul=kwargs.get('c_hidden_mul', 32),
            c_hidden_pair_att=kwargs.get('c_hidden_pair_att', 32),
            c_s=kwargs.get('c_s', 384),
            no_heads_seq=kwargs.get('no_heads_seq', 8),
            no_heads_pair=kwargs.get('no_heads_pair', 4),
            no_blocks=kwargs.get('no_blocks', 4),
            transition_n=kwargs.get('transition_n', 4),
            blocks_per_ckpt=kwargs.get('blocks_per_ckpt', 4),
            inf=kwargs.get('inf', 1e9),
            eps=kwargs.get('eps', 1e-10),
        )
        
        # Embed seq from positional encoding + time to c_m
        seq_input_dim = kwargs.get('seq_input_dim', 32) + 1  # +1 for time step
        self.embed_seq = nn.Linear(seq_input_dim, kwargs.get('c_m', 64))
        
        # Embed z from [N, N, 4 + num_classes + 1] to [N, N, c_z]
        z_input_dim = kwargs.get('z_input_dim', 4) + self.num_classes + 1  # +num_classes for one-hot dist +1 for time step
        self.embed_z = nn.Linear(z_input_dim, kwargs.get('c_z', 64))
        
        # Predict distogram logits from pair embedding z
        self.dist_head = Linear(kwargs.get('c_z', 64), self.num_classes, init="final")

    def forward(self, data, training=None):
        """Forward pass for training using D3PM discrete diffusion on DistAttDataset"""
        seq = data['seq']  # [B, N, C_m] - positional encoding
        z = data['z']      # [B, N, N, 4] - residue relationship matrix
        seq_mask = data['seq_mask']
        pair_mask = data['pair_mask']
        dist_target = data['dist']  # [B, N, N] - target distance classes
        
        b, n, _ = seq.shape
        
        # Convert dist_target to discrete classes (flatten to 1D for processing)
        x_0 = dist_target.flatten(start_dim=1)  # (B, N*N) - discrete distance class indices
        
        # Sample random timestep for training
        t = torch.randint(1, self.n_T, (b,), device=x_0.device)
        
        # Apply forward process to get noisy data
        x_t = self.q_sample(
            x_0, t, torch.rand((*x_0.shape, self.num_classes), device=x_0.device)
        )
        
        # Predict clean data from noisy data
        predicted_x0_logits = self.model_predict(x_t, t, seq, z, seq_mask, pair_mask)
        
        # Apply pair mask to loss computation
        pair_mask_flat = pair_mask.flatten(start_dim=1)  # (B, N*N)
        
        # Cross-entropy loss (direct prediction of clean data) - masked
        predicted_x0_logits_flat = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x_0_flat = x_0.flatten(start_dim=0, end_dim=-1)
        pair_mask_flat = pair_mask_flat.flatten(start_dim=0, end_dim=-1)
        ce_loss = self.ce_masked(predicted_x0_logits_flat, x_0_flat, pair_mask_flat)
        
        # Calculate accuracy (only on masked pairs)
        predicted_x0_flat = predicted_x0_logits_flat.argmax(dim=-1)
        correct = (predicted_x0_flat == x_0_flat) & pair_mask_flat.bool()
        accuracy = correct.sum().float() / pair_mask_flat.sum().float()
        
        return {'loss': ce_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def sample_chain(self, data, keep_frames=None):
        """Sample from the reverse diffusion process using D3PM on DistAttDataset"""
        if keep_frames is None:
            keep_frames = self.n_T
        else:
            assert keep_frames <= self.n_T
            
        seq = data['seq']  # [B, N, C_m] - positional encoding
        z = data['z']      # [B, N, N, 4] - residue relationship matrix
        seq_mask = data['seq_mask']
        pair_mask = data['pair_mask']
        
        n_samples = seq.size(0)
        n = seq.size(1)
        
        # Initialize with pure noise (random discrete classes)
        x = torch.randint(0, self.num_classes, (n_samples, n * n), device=seq.device)
        
        # Store intermediate results
        # chain = torch.zeros((keep_frames, n_samples, n * n), device=seq.device)
        chain = []
        
        # Sample p(x_{t-1} | x_t)
        for i, t in enumerate(reversed(range(1, self.n_T))):
            t_tensor = torch.full((n_samples,), fill_value=t, device=x.device)
            
            # Generate noise for Gumbel sampling
            noise = torch.rand((n_samples, n * n, self.num_classes), device=x.device)
            
            # Sample one step back
            x = self.p_sample(x, t_tensor, seq, z, seq_mask, pair_mask, noise)

            chain.append(x)
            
            # Store in chain
            # write_index = (i * keep_frames) // self.n_T
            # if write_index < keep_frames:
            #     chain[write_index] = x
        
        # Convert back to dist format (reshape to [B, N, N])
        # x_reshaped = x.view(n_samples, n, n)
        # chain_reshaped = chain.view(keep_frames, n_samples, n, n)

        chain = torch.stack(chain, dim=0)
        chain = chain.view(-1, n_samples, n, n)
        
        return chain[-1], chain


    def _at(self, a, t, x):
        """
        Helper function to index into transition matrices.
        
        Args:
            a: Transition matrix tensor of shape [T, num_classes, num_classes]
            t: Timestep tensor of shape [batch_size]
            x: Current state tensor of shape [batch_size, ...]
        
        Returns:
            Selected transition probabilities of shape [batch_size, ..., num_classes]
        """
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        """
        Compute the posterior logits q(x_{t-1} | x_t, x_0).
        
        This implements the analytical posterior distribution for discrete diffusion.
        The formula is: q(x_{t-1} | x_t, x_0) ∝ Q_t^T(x_t, x_{t-1}) * Q_bar_{t-1}(x_{t-1}, x_0)
        
        Args:
            x_0: Clean data (either one-hot or logits)
            x_t: Noisy data at timestep t
            t: Current timestep
        
        Returns:
            Logits for the posterior distribution q(x_{t-1} | x_t, x_0)
        """
        # Convert discrete indices to logits if needed
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
        
        # Convert to probabilities for matrix multiplication
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        
        # Q_t^T(x_t, x_{t-1}) - probability of reaching x_t from x_{t-1}
        fact1 = self._at(self.q_one_step_transposed, t, x_t)
        
        # Q_bar_{t-1}(x_{t-1}, x_0) - probability of reaching x_0 from x_{t-1}
        # Handle t=1 case (no previous timestep)
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        t_is_1 = (t == 1).float().reshape((t.shape[0], *[1] * (x_t.dim())))
        
        # For t > 1, compute Q_bar_{t-1}
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)
        
        # Combine both factors in log space
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # Special case for t=1: posterior is just the clean data
        return torch.where(t_is_1.bool(), x_0_logits, out)

    def vb(self, dist1, dist2):
        """
        Compute variational bound loss (KL divergence between two distributions).
        
        This implements D_KL(dist1 || dist2) = sum(p * log(p/q))
        where p = softmax(dist1) and q = softmax(dist2).
        
        Args:
            dist1: Logits of first distribution (target)
            dist2: Logits of second distribution (prediction)
        
        Returns:
            Average KL divergence loss
        """
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        
        # Compute KL divergence: D_KL(dist1 || dist2)
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def vb_masked(self, dist1, dist2, edge_mask):
        """
        Compute masked variational bound loss (KL divergence between two distributions).
        
        Args:
            dist1: Logits of first distribution (target)
            dist2: Logits of second distribution (prediction)
            edge_mask: Edge mask indicating valid edges
        
        Returns:
            Average KL divergence loss over valid edges only
        """
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        
        # Compute KL divergence: D_KL(dist1 || dist2)
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        
        # Apply mask and compute mean only over valid edges
        kl_per_edge = out.sum(dim=-1)  # (B*E,)
        masked_kl = kl_per_edge * edge_mask  # Zero out invalid edges
        valid_count = edge_mask.sum()
        
        if valid_count > 0:
            return masked_kl.sum() / valid_count
        else:
            return torch.tensor(0.0, device=dist1.device)

    def ce_masked(self, logits, targets, edge_mask):
        """
        Compute masked cross-entropy loss.
        
        Args:
            logits: Predicted logits (B*E, num_classes)
            targets: Target class indices (B*E,)
            edge_mask: Edge mask indicating valid edges (B*E,)
        
        Returns:
            Average cross-entropy loss over valid edges only
        """
        # Compute cross-entropy loss
        ce_per_edge = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        
        # Apply mask and compute mean only over valid edges
        masked_ce = ce_per_edge * edge_mask
        valid_count = edge_mask.sum()
        
        if valid_count > 0:
            return masked_ce.sum() / valid_count
        else:
            return torch.tensor(0.0, device=logits.device)

    def q_sample(self, x_0, t, noise):
        """
        Sample from the forward process q(x_t | x_0).
        
        This applies noise to clean data x_0 to get noisy data x_t at timestep t.
        Uses Gumbel-Max trick for discrete sampling.
        
        Args:
            x_0: Clean discrete data
            t: Timestep
            noise: Uniform random noise for Gumbel sampling
        
        Returns:
            Noisy discrete data x_t
        """
        # Get transition probabilities from x_0 to x_t
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        
        # Clip noise to avoid numerical issues
        noise = torch.clip(noise, self.eps, 1.0)
        
        # Apply Gumbel-Max trick for discrete sampling
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_t_classes, t, seq, z, seq_mask, pair_mask):
        """
        Predict clean data x_0 from noisy data x_t using Evoformer.
        
        Args:
            x_t_classes: Noisy discrete data (B, N*N) - discrete class indices
            t: Current timestep
            seq: Sequence features [B, N, C_m]
            z: Pair features [B, N, N, 4]
            seq_mask: Sequence mask [B, N]
            pair_mask: Pair mask [B, N, N]
        
        Returns:
            Predicted logits for clean data x_0
        """
        b, n, _ = seq.shape
        
        # Reshape x_t_classes back to [B, N, N]
        x_t_reshaped = x_t_classes.view(b, n, n)
        
        # Add time step to sequence features
        seq_with_time = add_time_step(t, seq, seq_mask)
        
        # Convert noisy dist to one-hot and concatenate with z
        x_t_onehot = torch.nn.functional.one_hot(x_t_reshaped, num_classes=self.num_classes).float()
        # x_t_onehot: [B, N, N, num_classes]
        
        # Concatenate z with one-hot dist
        z_with_dist = torch.cat([z, x_t_onehot], dim=-1)  # [B, N, N, 4 + num_classes]
        
        # Add time step to pair features
        z_with_time = add_time_step(t, z_with_dist, pair_mask)
        
        # Embed seq and z
        seq_embedded = self.embed_seq(seq_with_time)
        z_embedded = self.embed_z(z_with_time)
        
        # Run Evoformer
        seq_out, z_out, s = self.evoformer(
            seq=seq_embedded,
            z=z_embedded,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            chunk_size=4,
        )
        
        # Predict distogram logits from pair embedding
        dist_logits = self.dist_head(z_out)  # [B, N, N, num_classes]
        
        # Reshape back to [B, N*N, num_classes] for consistency with D3PM
        dist_logits_flat = dist_logits.view(b, n * n, self.num_classes)
        
        return dist_logits_flat
    
    def _d3pm_sample(self, x, seq, z, seq_mask, pair_mask):
        """Generate clean data by iteratively denoising from pure noise using D3PM"""
        # Iterate backwards through timesteps
        for t in reversed(range(1, self.n_T)):
            t_tensor = torch.tensor([t] * x.shape[0], device=x.device)
            # Generate noise for Gumbel sampling
            noise = torch.rand((*x.shape, self.num_classes), device=x.device)
            # Denoise one step
            x = self.p_sample(x, t_tensor, seq, z, seq_mask, pair_mask, noise)
        return x

    def p_sample(self, x, t, seq, z, seq_mask, pair_mask, noise):
        """
        Sample from the reverse process p_theta(x_{t-1} | x_t).
        
        This is the denoising step during generation. It predicts x_{t-1} from x_t
        using the learned model and Gumbel sampling for discrete data.
        
        Args:
            x: Current noisy data x_t
            t: Current timestep
            seq: Sequence features [B, N, C_m]
            z: Pair features [B, N, N, 4]
            seq_mask: Sequence mask [B, N]
            pair_mask: Pair mask [B, N, N]
            noise: Uniform random noise for Gumbel sampling
        
        Returns:
            Sampled data x_{t-1}
        """
        # Predict clean data from current noisy state
        predicted_x0_logits = self.model_predict(x, t, seq, z, seq_mask, pair_mask)
        
        # Compute posterior p_theta(x_{t-1} | x_t) using predicted x_0
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)
        
        # Clip noise to avoid numerical issues
        noise = torch.clip(noise, self.eps, 1.0)
        
        # For t=1, don't add noise (direct sampling from predicted x_0)
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        
        # Apply Gumbel-Max trick for discrete sampling
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample





