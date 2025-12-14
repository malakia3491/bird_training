import torch.nn as nn

class IdentityFrontend(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    def forward(self, x):
        return x