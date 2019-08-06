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

    def forward(self, input):
        print('LanguageModel forward', len(input))
        imgs = input[0]
        seq = input[1] # shape must be (seq_len, batch_size)
        assert(seq.size()[0] == self.seq_length)
        batch_size = seq.size()[1]
        self.output = torch.zeros(self.seq_length + 2, batch_size, self.vocab_size + 1, device=self.device)
        self._createInitialState(batch_size)
        self.state = [self.init_state]
        self.inputs = []
        self.embedding_inputs = []
        self.tmax = 0
        for t in range(2 + self.seq_length):
            can_skip = False
            if t == 0:
                xt = imgs
            elif t == 1:
                it = torch.zeros(batch_size, dtype=torch.long, device=self.device) + self.vocab_size + 1
                self.embedding_inputs.append(it)
                xt = self.embedding(it)
            else:
                it = seq[t - 2]
                if torch.sum(it) == 0:
                    can_skip = True
                
                if not can_skip:
                    self.embedding_inputs.append(it)
                    xt = self.embedding(it)
                
            if not can_skip:
                self.inputs.append([xt, *self.state[t]])
                out = self.core(self.inputs[-1])
                self.output[t] = out[-1]
                self.state.append([])
                for i in range(self.num_state):
                    self.state[t+1].append(out[i])
                self.tmax = t
            
        return self.output

    def sample(self, imgs, sample_max=1, beam_size=1, temperature=1.0):

        if sample_max == 1 and beam_size > 1 :
            return self.beam_search(imgs, beam_size)

        batch_size = imgs.size()[0]
        self._createInitialState(batch_size)

        state = self.init_state
        seq = torch.zeros(self.seq_length, batch_size, dtype=torch.long, device=self.device)
        seqLogprobs = torch.zeros(self.seq_length, batch_size, device=self.device)
        logprobs = torch.zeros(batch_size, self.rnn_size, device=self.device)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = imgs
            elif t == 1:
                it = torch.zeros(batch_size, dtype=torch.long, device=self.device) + self.vocab_size + 1
                xt = self.embedding(it)

            else:
                if sample_max == 1:
                    sampleLogprobs, it = torch.max(logprobs, -1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0 :
                        prob_prev = torch.exp(logprobs)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(-1, it)
                    it = it.view(-1).long()
		# vocab indexing starts from 1 so have to increase each index by 1
                it = it + 1
                xt = self.embedding(it)
            # xt : (batch_size, emb_size)
            # it : (batch_size)
            if t >= 2:
                seq[t-2] = it
                seqLogprobs[t-2] = sampleLogprobs.view(-1).float()
            
            inputs = [xt, *self.state]
            out = self.core(inputs)
            logprobs = out[-1]
            state = []
            for i in range(self.num_state) :
                state.append(out[i])
        
        return seq, seqLogprobs
    
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
    
