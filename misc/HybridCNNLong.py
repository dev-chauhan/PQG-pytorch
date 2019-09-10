import torch
import torch.nn as nn
from misc.FixedGRU import FixedGRU

class HybridCNNLong(nn.Module):

    def __init__(self, alphasize, emb_dim, dropout=0.0, avg=False, cnn_dim=512):
        super(HybridCNNLong, self).__init__()
        self.conv1 = nn.Conv1d(alphasize, 256, 1)
        self.threshold = nn.Threshold(0.000001, 0)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.GRU = FixedGRU(26, avg, cnn_dim)
        self.linear = nn.Linear(cnn_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input):
        # print(input.size())
        # input1 = torch.stack([input[i].t() for i in range(input.size()[0])])
        input1 = input.permute(0, 2, 1)
        out1 = self.conv1(input1)
        out1 = self.threshold(out1)
        out2 = self.conv2(out1)
        out2 = self.threshold(out2)
        
        # h1 = torch.stack([out2[:,:,i] for i in range(out2.size()[-1])])
        h1 = out2.permute(2, 0, 1)
        r2 = self.GRU(h1)
        out = self.linear(self.dropout_layer(r2))
        
        return out
