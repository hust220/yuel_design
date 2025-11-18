import torch
import torch.nn as nn


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

