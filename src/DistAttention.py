import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x, dim=-1, eps=1e-8):
    """L2归一化工具函数"""
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, activation='silu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class DistAttention(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, eps=1e-8):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # self.num_heads = num_heads
        # assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        # self.head_dim = d_model // num_heads

        self.linear_Q1 = FeedForward(d_model, dim_feedforward)
        self.linear_K1 = FeedForward(d_model, dim_feedforward)
        self.linear_V1 = FeedForward(d_model, dim_feedforward)

        self.linear_Q2 = FeedForward(d_model, dim_feedforward)
        self.linear_K2 = FeedForward(d_model, dim_feedforward)
        self.linear_V2 = FeedForward(d_model, dim_feedforward)

        self.ffn1 = FeedForward(d_model, dim_feedforward)
        self.ffn2 = FeedForward(d_model, dim_feedforward)
        
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.norm3 = nn.LayerNorm(d_model, eps=eps)
        self.norm4 = nn.LayerNorm(d_model, eps=eps) 

    def forward(self, h, z):
        # z = self.norm1(z)
        # Q1 = self.linear_Q1(z) # (b, n, n, d) 
        # K1 = self.linear_K1(z) # (b, n, n, d)
        # V1 = self.linear_V1(z) # (b, n, n, d)
        # Q1 = l2_normalize(Q1, dim=-1, eps=self.eps)
        # K1 = l2_normalize(K1, dim=-1, eps=self.eps)
        # z = z + torch.einsum('bijd,bikd,bikt->bijt', Q1, K1, V1) # (b, n, n, d)

        # z = self.norm2(z)
        # z = z + self.ffn1(z)

        # 问题：1. 缺乏归一化
        # 2. 缺乏多头注意力

        h = self.norm1(h)
        Q1 = self.linear_Q1(h) # (b, n, n, d)
        K1 = self.linear_K1(h) # (b, n, n, d)
        V1 = self.linear_V1(h) # (b, n, n, d)
        Q1 = l2_normalize(Q1, dim=-1, eps=self.eps)
        K1 = l2_normalize(K1, dim=-1, eps=self.eps)
        h = h + torch.einsum('bid,bjd,bjt->bit', Q1, K1, V1) # (b, n, n, d)

        h = self.norm2(h)
        h = h + self.ffn1(h)

        z = z + h.unsqueeze(2)

        z = self.norm3(z)
        Q2 = self.linear_Q2(z) # (b, n, n, d)
        K2 = self.linear_K2(z) # (b, n, n, d)
        V2 = self.linear_V2(z) # (b, n, n, d)

        Q2 = l2_normalize(Q2, dim=-1, eps=self.eps)
        K2 = l2_normalize(K2, dim=-1, eps=self.eps)
        z = z + torch.einsum('bijd,bkjd,bkjt->bijt', Q2, K2, V2) # (b, n, n, d)

        z = self.norm4(z)
        z = z + self.ffn2(z)

        h = h + z.sum(dim=2)

        return h, z
