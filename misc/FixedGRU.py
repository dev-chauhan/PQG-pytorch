import torch
import torch.nn as nn

class FixedGRU(nn.Module):
    
    def __init__(self, nstep, avg=False, emb_dim=256):
        '''
        nstep : number of time steps
        avg : take average of all hidden states or not
        emb_dim : num of features of each word
        '''
        super(FixedGRU, self).__init__()
        self.layer = nn.GRU(emb_dim, emb_dim)
        self.avg = avg
        self.nstep = nstep

    def forward(self, inputs):
        '''
        inputs : (nstep, batch_size, emb_dim) torch tensor

        out : (batch_size, emb_dim) torch tensor
        '''
        output , h_n = self.layer(inputs)
        if self.avg == True:
            out = output[0]
            for it in range(1, output.size()[0]):
                out = out + output[it]
            out = out / self.nstep
        else:
            out = h_n
        
        return out
