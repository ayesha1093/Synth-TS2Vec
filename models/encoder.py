import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
def generate_gaussian_mask(B, T, center_prob=0.8, sigma=3, noise_std=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    center = T // 2
    for i in range(B):
        for t in range(T):
            prob = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
            if np.random.rand() < center_prob * prob:
                res[i, t] = False
                # Add Gaussian noise to the masked value
                res[i, t] = res[i, t] + torch.randn(1) * noise_std
    return res
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res
def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='continuous'):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            in_channels=hidden_dims,
            channels=[hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=2
        )
        #print(self.feature_extractor)
        self.repr_dropout = nn.Dropout(p=0.1)
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        if mask is not None:
            # mask the final timestamp for anomaly detection task
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            mask &= nan_mask
            x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
        
