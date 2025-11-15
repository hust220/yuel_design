import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x, dim=-1, eps=1e-8):
    """L2归一化工具函数"""
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)


class ToAttention(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, eps=1e-8, with_bias=False, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads

        # 二阶注意力核心组件（多头）
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        # 三阶注意力核心组件
        self.linear_A = nn.Linear(d_model, d_model)
        self.linear_B = nn.Linear(d_model, d_model)
        self.linear_C = nn.Linear(d_model, d_model)
        self.linear_V1 = nn.Linear(d_model, d_model)
        self.linear_V2 = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        if with_bias:
            self.linear_V_bias = nn.Linear(d_model, d_model)
        
        # FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # LayerNorm（前置归一化用）
        self.norm1 = nn.LayerNorm(d_model, eps=eps)  # 二阶注意力输入归一化
        self.norm2 = nn.LayerNorm(d_model, eps=eps)  # 三阶注意力输入归一化
        self.norm3 = nn.LayerNorm(d_model, eps=eps)  # FFN输入归一化

    def forward(self, x, V_bias=None):
        # 1. 二阶多头注意力（Pre-Norm：先归一化，再计算注意力）
        b, n, _ = x.shape
        h, hd, d = self.num_heads, self.head_dim, self.d_model
        
        x_norm = self.norm1(x)
        # transpose的目的是把head维度放在第二位，这样pytorch就可以调用高度优化后的矩阵乘法，效率更高
        Q = self.linear_Q(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        K = self.linear_K(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        V = self.linear_V(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)

        # L2 normalize Q and K
        Q = l2_normalize(Q, dim=-1, eps=self.eps)
        K = l2_normalize(K, dim=-1, eps=self.eps)

        # W = torch.matmul(Q, K.transpose(-2, -1))
        # attn_output = torch.matmul(W, V)
        # attn_output = attn_output.view(b, h, n, hd).transpose(1, 2).reshape(b, n, d)
        attn_output = torch.einsum('bid,bjd,bjt->bit', Q, K, V).view(b, h, n, hd).transpose(1, 2).reshape(b, n, d)
        attn_output = self.out_proj(attn_output)
        
        x = x + attn_output
        
        # 2. 三阶注意力（Pre-Norm：先归一化，再计算注意力）
        # 对输入做LayerNorm后再送入注意力计算
        x_norm = self.norm2(x)

        A = self.linear_A(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        B = self.linear_B(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        C = self.linear_C(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        V1 = self.linear_V1(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        V2 = self.linear_V2(x_norm).view(b, n, h, hd).transpose(1, 2).reshape(b*h, n, hd)
        
        A = l2_normalize(A, dim=-1, eps=self.eps)
        B = l2_normalize(B, dim=-1, eps=self.eps)
        C = l2_normalize(C, dim=-1, eps=self.eps)
        
        if V_bias is not None:
            # 计算三阶注意力权重: W[i,j,k] = A[i] · B[j] · C[k]
            # 使用einsum避免显式扩展维度，节省显存
            W = torch.einsum('bid,bjd,bkd->bijk', A, B, C)
            # 计算注意力输出并投影
            V_outer = torch.einsum('bjt,bkt->bjkt', V1, V2)
            # 加上V_bias
            if V_bias is not None:
                V_outer = V_outer + self.linear_V_bias(V_bias).view(b, n*n, h, hd).transpose(1, 2).reshape(b*h, n, n, hd)
            # attn_output = torch.matmul(W, V_outer)
            # attn_output = attn_output.view(b, h, n, hd).transpose(1, 2).reshape(b, n, d)
            attn_output = self.out_proj(torch.einsum('bijk,bjkt->bit', W, V_outer).view(b, h, n, hd).transpose(1, 2).reshape(b, n, d))
        else:
            attn_output = self.out_proj(torch.einsum('bid,bjd,bkd,bjt,bkt->bit', A, B, C, V1, V2).view(b, h, n, hd).transpose(1, 2).reshape(b, n, d))        

        # 注意力部分：残差连接
        x = x + attn_output
        
        # 3. FFN（Pre-Norm：先归一化，再计算FFN）
        x_norm = self.norm3(x)
        
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x
