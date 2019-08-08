import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, output_size, rnn_size, n_layers, dropout=0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, rnn_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.dense = nn.Linear(rnn_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.soft = nn.LogSoftmax(dim=-1)
        self.n_layers = n_layers
        self.rnn_size = rnn_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, inputs):
        '''
        inputs : list
        inputs[0] : size == (batch_size, feat_size)
        inputs[i] : size == (batch_size, rnn_size)
        '''
        input = torch.stack([inputs[0]])
        h_0 = torch.zeros(self.n_layers, input.size()[1], self.rnn_size, device=self.device)
        c_0 = torch.zeros(self.n_layers, input.size()[1], self.rnn_size, device=self.device)

        for i in range(1, 2 * (self.n_layers) + 1):
            if i % 2 == 0:
                h_0[(i-1)//2,:,:] = inputs[i]
            else:
                c_0[(i-1)//2,:,:] = inputs[i]
        self.rnn.flatten_parameters()
        out, (h_n, c_n) = self.rnn(input, (h_0, c_0))
        out = self.dropout_layer(out[0])
        proj = self.dense(out)
        logsoft = self.soft(proj)

        outputs = []
        for i in range(self.n_layers):
            outputs.append(c_n[i])
            outputs.append(h_n[i])
        
        outputs.append(logsoft)

        return outputs
