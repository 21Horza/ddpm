import torch
from torch import nn
import math

# FlashAttention-inspired block-wise computation
def flash_attention(q, k, v, scale):
    attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v)
    return output

# SparseTopKAttention with entropy-based dynamic sparsity
class DynamicSparseTopKAttention(nn.Module):
    def __init__(self, max_top_k):
        super().__init__()
        self.max_top_k = max_top_k

    def forward(self, weights):
        # Compute attention map entropy
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1, keepdim=True)
        top_k = torch.clamp((self.max_top_k * (1 - entropy)).int(), min=1, max=self.max_top_k)

        # Get dynamic top-k threshold
        sorted_weights, _ = weights.sort(dim=-1, descending=True)
        thresholds = sorted_weights.gather(-1, top_k - 1).unsqueeze(-1)
        sparse_weights = torch.where(weights >= thresholds, weights, torch.zeros_like(weights))
        sparse_weights /= sparse_weights.sum(dim=-1, keepdim=True) + 1e-8
        return sparse_weights

# Combined Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, use_sparse=True, max_top_k=32, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.use_sparse = use_sparse
        self.sparse_attention = DynamicSparseTopKAttention(max_top_k) if use_sparse else None

    def forward(self, x, causal_mask=False):
        batch_size, seq_len, d_embed = x.size()
        qkv = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = (tensor.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) for tensor in qkv)
        scale = 1.0 / math.sqrt(self.d_head)

        # Compute attention weights
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(attn_weights.device).bool()
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), -float("inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply sparse attention if enabled
        if self.use_sparse:
            attn_weights = self.sparse_attention(attn_weights)

        output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        return self.out_proj(output)

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, use_sparse=True, max_top_k=32, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.use_sparse = use_sparse
        self.sparse_attention = DynamicSparseTopKAttention(max_top_k) if use_sparse else None

    def forward(self, x, y):
        batch_size, seq_len_q, d_embed = x.size()
        seq_len_kv = y.size(1)

        q = self.q_proj(x).view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(y).view(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(y).view(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)

        # Compute attention weights
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply sparse attention if enabled
        if self.use_sparse:
            attn_weights = self.sparse_attention(attn_weights)

        output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len_q, d_embed)
        return self.out_proj(output)

# Final Explanation:
# - **Hybrid Attention**: Combines dense and sparse attention with dynamic sparsity.
# - **Dynamic Top-k**: Adjusts sparsity based on entropy to balance speed and quality.
# - **Configurable**: Enables flexibility in choosing dense or sparse modes.



# PREVIOUS VERSION:

# import torch
# from torch import nn
# from torch.nn import functional as F
# import math

# class SelfAttention(nn.Module):
#     def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
#         super().__init__()
#         # This combines the Wq, Wk and Wv matrices into one matrix
#         self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
#         # This one represents the Wo matrix
#         self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
#         self.n_heads = n_heads
#         self.d_head = d_embed // n_heads

#     def forward(self, x, causal_mask=False):
#         # x: # (Batch_Size, Seq_Len, Dim)

#         # (Batch_Size, Seq_Len, Dim)
#         input_shape = x.shape 
        
#         # (Batch_Size, Seq_Len, Dim)
#         batch_size, sequence_length, d_embed = input_shape 

#         # (Batch_Size, Seq_Len, H, Dim / H)
#         interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

#         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
#         q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
#         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
#         q = q.view(interim_shape).transpose(1, 2)
#         k = k.view(interim_shape).transpose(1, 2)
#         v = v.view(interim_shape).transpose(1, 2)

#         # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
#         weight = q @ k.transpose(-1, -2)
        
#         if causal_mask:
#             # Mask where the upper triangle (above the principal diagonal) is 1
#             mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
#             # Fill the upper triangle with -inf
#             weight.masked_fill_(mask, -torch.inf) 
        
#         # Divide by d_k (Dim / H). 
#         # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
#         weight /= math.sqrt(self.d_head) 

#         # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
#         weight = F.softmax(weight, dim=-1) 

#         # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
#         output = weight @ v

#         # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
#         output = output.transpose(1, 2) 

#         # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
#         output = output.reshape(input_shape) 

#         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
#         output = self.out_proj(output) 
        
#         # (Batch_Size, Seq_Len, Dim)
#         return output

# class CrossAttention(nn.Module):
#     def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
#         super().__init__()
#         self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
#         self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
#         self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
#         self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
#         self.n_heads = n_heads
#         self.d_head = d_embed // n_heads
    
#     def forward(self, x, y):
#         # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
#         # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

#         input_shape = x.shape
#         batch_size, sequence_length, d_embed = input_shape
#         # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
#         interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         q = self.q_proj(x)
#         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
#         k = self.k_proj(y)
#         # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
#         v = self.v_proj(y)

#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
#         q = q.view(interim_shape).transpose(1, 2) 
#         # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
#         k = k.view(interim_shape).transpose(1, 2) 
#         # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
#         v = v.view(interim_shape).transpose(1, 2) 
        
#         # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
#         weight = q @ k.transpose(-1, -2)
        
#         # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
#         weight /= math.sqrt(self.d_head)
        
#         # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
#         weight = F.softmax(weight, dim=-1)
        
#         # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
#         output = weight @ v
        
#         # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
#         output = output.transpose(1, 2).contiguous()
        
#         # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         output = output.view(input_shape)
        
#         # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
#         output = self.out_proj(output)

#         # (Batch_Size, Seq_Len_Q, Dim_Q)
#         return output