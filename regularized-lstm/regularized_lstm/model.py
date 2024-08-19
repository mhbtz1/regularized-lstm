import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, d_hidden, d_memcell):
        self.hidden = torch.rand(d_hidden, 1)
        self.memory_cell = torch.rand(d_memcell, 1)
    
    def forward(self, input, upper_hidden, left_hidden):
        pass