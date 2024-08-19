import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, d_hidden, d_memcell, d_trans):
        self.hidden = nn.Parameter(torch.rand(d_hidden, 1))
        self.memory_cell = nn.Parameter(torch.rand(d_memcell, 1))
        self.trans_matrix = nn.Parameter(torch.rand(2 * d_trans, 4 * d_trans))
    
    def forward(self, input, prev_memory_cell, upper_hidden, left_hidden):
        i = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (upper_hidden, left_hidden) )))
        f = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (upper_hidden, left_hidden) )))
        o = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (upper_hidden, left_hidden) )))
        g = torch.tanh( torch.matmul(self.trans_matrix, torch.concat( (upper_hidden, left_hidden) )))
        self.memory_cell = ((f * prev_memory_cell) + (i * g))
        return ( (o * torch.tanh(self.memory_cell)), self.memory_cell)
    



    

