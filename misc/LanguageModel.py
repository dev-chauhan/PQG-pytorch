import torch
import torch.nn as nn
import misc.utils as utils
import misc.net_utils as net_utils
from misc.LSTM import LSTM

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class layer(nn.Module):

    def __init__(self, input_encoding_size, rnn_size, seq_length, vocab_size, num_layers=1, dropout=0):
        super(layer, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.core = LSTM(input_encoding_size, vocab_size + 1, rnn_size, num_layers, dropout=dropout)
	    # 0 is padding token
	    # vocab_size + 1 is start token
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.embedding = nn.Embedding(vocab_size + 2, input_encoding_size, padding_idx=0)
        self._createInitialState(1)

    def _createInitialState(self, batch_size):
        self.init_state = [None for i in range(self.num_layers * 2)]
        for i in range(2 * self.num_layers):
            self.init_state[i] = torch.zeros(batch_size, self.rnn_size, device=self.device)
        self.num_state = 2 * self.num_layers
    
    def getModulesList(self):
        return [self.core, self.embedding]

    def parameters(self, recurse=True):
        params = super().parameters(recurse=recurse)
        grad_params = [param.grad for param in params]

        return params, grad_params

    def forward(self, encoded, seq, lengths,teacher_forcing=True):
        '''
        encoded : (batch_size, feat_size)
        seq: (batch_size, seq_len)
        lengths: (batch_size, )
        '''
        
        if teacher_forcing:
            embedded = self.embedding(seq)
            input_rnn = torch.cat([encoded.unsqueeze(1), embedded], dim=1)
            lengths = lengths + 1
            output, _ = self.core(input_rnn, lengths)
            return output
        else:
            print('Sampling...')
            output = torch.zeros(encoded.size()[0], self.seq_length, device=self.device)
            for batch in range(encoded.size()[0]):
                
                encoding = encoded[batch]
                encoding = encoding.unsqueeze(0)
                encoding = encoding.unsqueeze(1)
                
                idx = 0
                it = 0
                h = None
                while True:
                    if idx >= self.seq_length or it == self.vocab_size + 1:
                        break
                    if h != None:
                        prob, hidden = self.core(encoding, torch.tensor([1]), h=hidden[-1])
                    else:
                        prob, hidden = self.core(encoding, torch.tensor([1]))
                    prob = prob.view(1, -1)
                    prob_dist = torch.exp(prob)
                    it = torch.multinomial(prob_dist, 1)
                    encoding = self.embedding(it + 1)
                    
                    output[batch, idx] = it + 1
                    idx += 1

            return output.t()
        

    def sample(self, imgs, sample_max=1, beam_size=1, temperature=1.0):

        if sample_max == 1 and beam_size > 1 :
            return self.beam_search(imgs, beam_size)
        
        seq = self.forward(imgs, None, torch.tensor([1]), teacher_forcing=False)
        
        return seq
    
    def beam_search(self, imgs, beam_size=10):

        batch_size, feat_dim = imgs.size()[0], imgs.size()[1]

        def compare_key(a):
            return a.p

        assert(beam_size <= self.vocab_size + 1)

        seq = torch.zeros(self.seq_length, batch_size, dtype=torch.long, device=self.device)
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size, device=self.device)

        for k in range(batch_size):
            self._createInitialState(beam_size)
            state = self.init_state

            beam_seq = torch.zeros(self.seq_length, beam_size, dtype=torch.long, device=self.device)
            beam_seq_logprobs = torch.zeros(self.seq_length, beam_size, device=device)
            beam_logprobs_sum = torch.zeros(beam_size, device=self.device)
            logprobs = torch.zeros(beam_size, self.vocab_size + 1, device = self.device)
            done_beams = []

            for t in range(self.seq_length + 2):

                if t == 0:
                    imgk = imgs[k].expand(beam_size, feat_dim)
                    xt = imgk
                elif t == 1:
                    it = torch.LongTensor(beam_size).fill_(self.vocab_size + 1)
                    xt = self.embedding(it)
                else:
                    logprobsf = logprobs.float()
                    ys, ix = torch.sort(logprobsf, dim=-1, descending=True)
                    candidates = []
                    cols = min(beam_size, ys.size()[-1])
                    rows = beam_size

                    if t == 2:
                        rows = 1
                    
                    for c in range(cols):
                        for q in range(rows):
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            d = {'c':ix[q, c], 'q':q, 'p':candidate_logprob, 'r':local_logprob}
                            candidates.append(Struct(**d))
                    
                    candidates.sort(key = compare_key, reverse = True)
                    new_state = net_utils.clone_list(state)
                    
                    if t > 2:
                        beam_seq_prev = beam_seq[0:t-2, :].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[0:t-2, :].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 2:
                            beam_seq[0:t-2, vix] = beam_seq_prev[:, v.q]
                            beam_seq_logprobs[0:t-2, vix] = beam_seq_logprobs_prev[:, v.q]
                        
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][vix] = state[state_ix][v.q]
                        
                        beam_seq[t-2,vix] = v.c
                        beam_seq_logprobs[t-2, vix] = v.r
                        beam_logprobs_sum[vix] = v.p

                        if v.c == self.vocab_size or t == self.seq_length + 1:
                            beam = {'seq': beam_seq[:, vix].clone(), 'logps':beam_seq_logprobs[:,vix].clone(), 'p':beam_logprobs_sum[vix]}
                            done_beams.append(Struct(**beam))

                    it = beam_seq[t-2]
                    xt = self.embedding(it)
                
                if new_state:
                    state = new_state
                
                inputs = [xt, *state]
                out = self.core(inputs)
                logprobs = out[-1]
                state = []
                for i in range(self.num_state):
                    state.append(out[i])
            
            done_beams.sort(key = compare_key, reverse=True)
            seq[:, k] = done_beams[0].seq
            seqLogprobs[:,k] = done_beams[0].logps
        
        return seq, seqLogprobs


class crit(nn.Module):

    def __init__(self):
        super(crit, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def forward(self, input, seq):
        
        self.gradInput = torch.zeros(*input.size(), device=self.device)
        L, N, Mp1 = input.size()
        D = seq.size()[0]

        assert(D == L - 2)

        loss = 0
        n = 0
        for b in range(N):
            first_time = True

            for t in range(1, L-1):
                
                if t-1 > D:
                    target_index = 0
                else:
                    target_index = seq[t-1, b]

                if target_index == 0 and first_time:
                    target_index = Mp1 - 1
                    first_time = False

                if target_index != 0 :
                    loss -= input[t, b, target_index]
                    self.gradInput[t, b, target_index] = -1
                    n += 1
                
        self.output = loss / n
        self.gradInput = self.gradInput / n
        return self.output
    
