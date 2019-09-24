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
        
        self.encoder = DocumentCNN(self.vocab_size, args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
        
        self.decoder = LanguageModel(self.input_encoding_size, self.rnn_size, self.seq_length, self.vocab_size, num_layers=self.num_layers, dropout=self.drop_prob_lm)
        
    
    def JointEmbeddingLoss(self, feature_emb1, feature_emb2):
        
        batch_size = feature_emb1.size()[0]
        loss = 0
        for i in range(batch_size):
            label_score = torch.dot(feature_emb1[i], feature_emb2[i])
            for j in range(batch_size):
                cur_score = torch.dot(feature_emb2[i], feature_emb1[j])
                score = cur_score - label_score + 1
                if 0 < score.item():
                    loss += max(0, cur_score - label_score + 1)

        denom = batch_size * batch_size
        
        return loss / denom

    def forward(self, input_sentences, lengths):
        
        input_one_hot = torch.zeros(*input_sentences.size(), self.vocab_size, device=self.device)
        # for batch in range(input_sentences.size()[0]):
        #     for idx in range(input_sentences.size()[1]):
        #         input_one_hot[batch][idx][input_sentences[batch][idx]] = 1

        input_one_hot.scatter_(-1, input_sentences.unsqueeze(-1), 1) # [batch_size, seq_len, vocab_len]
        
        encoded = self.encoder(input_one_hot)
        
        probs = self.decoder(encoded, input_sentences, lengths) # (batch_size, seq_len, vocab_len)
        
        return (probs, encoded) # (batch_size, seq_len , vocab_size), (batch_size, feat_size)

    def sample(self, encoded_input):
        
        return self.decoder.sample(encoded_input)
