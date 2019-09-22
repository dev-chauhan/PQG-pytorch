import torch
import torch.nn as nn
from misc.FixedGRU import FixedGRU

class HybridCNNLong(nn.Module):

    def __init__(self, alphasize, emb_dim, dropout=0.0, avg=False, cnn_dim=512):
        super(HybridCNNLong, self).__init__()
        self.conv1 = nn.Conv1d(alphasize, 256, 1)
        self.threshold = nn.Threshold(0.000001, 0)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.GRU = FixedGRU(27, avg, cnn_dim)
        self.linear = nn.Linear(cnn_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x = [batch_size, seq, vocab]
        '''
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.threshold(x)
        x = self.conv2(x)
        x = self.threshold(x)
        
        x = x.permute(2, 0, 1) # [seq, batch, vocab]
        
        x = self.GRU(x)
    
        x = self.linear(self.dropout_layer(x)) # x = [batch_size, cnn_dim]
        

        return x
