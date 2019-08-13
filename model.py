import torch
import torch.nn as nn
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model(nn.Module):

    def __init__(self, args, dataloader):
        
        super(Model, self).__init__()
        self.vocab_size = dataloader.getVocabSize()
        self.input_encoding_size = args.input_encoding_size
        self.rnn_size = args.rnn_size
        self.num_layers = args.rnn_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.seq_length = dataloader.getSeqLength()
        self.batch_size = args.batch_size
        self.emb_size = args.input_encoding_size
        self.hidden_size = args.input_encoding_size
        self.att_size = args.att_size
        self.device = device
        
        self.encoder = DocumentCNN(self.vocab_size + 1, args.txtSize, 0, 1, args.cnn_dim)
        
        self.decoder = LanguageModel(self.input_encoding_size, self.rnn_size, self.seq_length, self.vocab_size, num_layers=self.num_layers, dropout=self.drop_prob_lm)
        
    
    def JointEmbeddingLoss(self, feature_emb1, feature_emb2):

        batch_size = feature_emb1.size()[0]
        score = torch.zeros(batch_size, batch_size, device = self.device)

        loss = 0
        acc_smooth = 0.0
        acc_batch = 0.0
        
        for i in range(batch_size):
            for j in range(batch_size):
                score[i, j] = torch.dot(feature_emb2[i], feature_emb1[j])
            
            label_score = score[i, i]
            for j in range(batch_size):
                if i != j :
                    cur_score = score[i, j]
                    thresh = cur_score - label_score + 1
                    if thresh > 0:
                        loss += thresh
            
        denom = batch_size * batch_size
        
        return loss / denom

    def forward(self, input_sentences):
        
        input_one_hot = torch.zeros(*input_sentences.size(), self.vocab_size + 1, device=self.device)
        for batch in range(input_sentences.size()[0]):
            for idx in range(input_sentences.size()[1]):
                input_one_hot[batch][idx][input_sentences[batch][idx]] = 1
        input_sentences_t = input_sentences.t()
        encoded = self.encoder(input_one_hot)
        probs = self.decoder([encoded, input_sentences_t])

        return (probs, encoded)

    def sample(self, encoded_input):
        
        return self.decoder.sample(encoded_input)
