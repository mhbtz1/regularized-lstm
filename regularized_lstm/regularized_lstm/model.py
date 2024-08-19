from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from modconfig import RegularizedLSTMConfig
from regularized_lstm.infra.dataloader import CoQADataLoader


dataloader = CoQADataLoader()

class RegularizedLSTM(nn.Module):
    def __init__(self, d_hidden, d_memcell, d_trans, dropout_prob=0.1):
        super(RegularizedLSTM, self).__init__()
        self.hidden = nn.Parameter(torch.rand(d_hidden, 1))
        self.memory_cell = nn.Parameter(torch.rand(d_memcell, 1))
        self.trans_matrix = nn.Parameter(torch.rand(2 * d_trans, 4 * d_trans))
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, prev_memory_cell, upper_hidden, left_hidden):
        t_upper_hidden = self.dropout(upper_hidden)
        i = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (t_upper_hidden, left_hidden) )))
        f = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (t_upper_hidden, left_hidden) )))
        o = torch.sigmoid(torch.matmul(self.trans_matrix, torch.concat( (t_upper_hidden, left_hidden) )))
        g = torch.tanh( torch.matmul(self.trans_matrix, torch.concat( (t_upper_hidden, left_hidden) )))
        self.memory_cell = ((f * prev_memory_cell) + (i * g))
        return ( (o * torch.tanh(self.memory_cell)), self.memory_cell)
    

d_hidden, d_memcell, d_trans = RegularizedLSTMConfig.d_hidden, RegularizedLSTMConfig.d_memcell, RegularizedLSTMConfig.d_trans
MODEL_M, MODEL_N = RegularizedLSTMConfig.MODEL_M, RegularizedLSTMConfig.MODEL_N

model = [[RegularizedLSTM(d_hidden, d_memcell, d_trans) for i in range(MODEL_N)] for j in range(MODEL_M)]

def train_model(nmt_data: DataLoader):
    for i, (data, result) in enumerate(nmt_data):
        pass


train_model(dataloader)



    



    

