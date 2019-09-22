import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, output_size, rnn_size, n_layers, dropout=0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, rnn_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)
        self.dense = nn.Linear(rnn_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.soft = nn.LogSoftmax(dim=-1)
        self.n_layers = n_layers
        self.rnn_size = rnn_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, input_batch, lengths, h = None):
        
        '''
        input_batch : (batch_size, seq_len + 1, feat_size)
        lengths: (batch_size, )
        '''
        batch_size = input_batch.size()[0]
        seq_len = input_batch.size()[1]
        out_lstm = torch.zeros(batch_size, seq_len , self.rnn_size, device=self.device)
        # for batch in range(batch_size):
        #     seq = input_batch[batch].unsqueeze(0)
            
        #     if h:
                
        #         out1, h = self.rnn(seq[:,:lengths[batch],:], h)
        #     else:
        #         out1, h = self.rnn(seq[:,:lengths[batch],:])
        #     # print(out1.size())
        #     out_lstm[batch, : lengths[batch], :] = out1[:,:,:]
        #     hidden.append(h)
        if h:
            out_lstm, hidden = self.rnn(input_batch, h)
        else :
            out_lstm, hidden = self.rnn(input_batch)

        logsoft = self.soft(self.dense(self.dropout_layer(out_lstm)))

        return logsoft, hidden # (batch_size, seq_len + 1, vocab_size), list(batch_size)
