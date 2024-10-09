###dilated convolutions for timetsamp representations
#https://github.com/bighuang624/DSANet/blob/master/dsanet
#https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master
import torch
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=10, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)  
            assert not torch.isnan(attn).any()
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn
class ConvBlockWithAttention(nn.Module): ##residual block
    def __init__(self, in_channels, out_channels, kernel_size, dilation, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.attention = MultiHeadAttention(n_head=n_head, d_model=out_channels, d_k=d_k, d_v=d_v, dropout=dropout)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(self.conv1(x))
        x = x.transpose(1, 2)  # [Batch, Channels, Time] to [Batch, Time, Channels]
        x = self.attention(x, x, x, mask=mask)
        x = x.transpose(1, 2)  # [Batch, Time, Channels] to [Batch, Channels, Time]
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        return x + residual
class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, n_head=4, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d_k = d_k or (channels[0] // n_head)  # Default to channels[0] // n_head if d_k is not provided
        d_v = d_v or (channels[0] // n_head)  # Default to channels[0] // n_head if d_v is not provided
        
        for i, ch in enumerate(channels):
            layer = ConvBlockWithAttention(
                in_channels=in_channels if i == 0 else channels[i - 1],
                out_channels=ch,
                kernel_size=kernel_size,
                dilation=2 ** i,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout
            )
            self.layers.append(layer)
    def forward(self, x, use_mask=False):
        mask = None
        if use_mask:
            seq_len = x.size(2)  # x shape (batch, channels, seq_len)
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
