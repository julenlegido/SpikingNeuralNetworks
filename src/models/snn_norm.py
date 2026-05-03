import torch
import torch.nn as nn

#using Threshold-dependent Batch Normalization (tdBN)
class SNNNorm(nn.Module):
    def __init__(self, v_th=1.0):
        super().__init__()
        self.v_th = v_th

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)

        x = (x - mean) / (std + 1e-5)
        x = x * self.v_th

        return x