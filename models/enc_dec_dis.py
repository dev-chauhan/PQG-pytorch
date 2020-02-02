import torch
import torch.nn as nn

import misc.utils as utils


class ParaphraseGenerator(nn.Module):
    """
    pytorch module which generates paraphrase of given phrase
    """
    def __init__(self, op):

        super(ParaphraseGenerator, self).__init__()

        # encoder :
        self.emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0))
        self.enc_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.enc_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # generator :
        self.gen_emb = nn.Embedding(op["vocab_sz"], op["emb_dim"])
        self.gen_rnn = nn.LSTM(op["enc_dim"], op["gen_rnn_dim"])
        self.gen_lin = nn.Sequential(
            nn.Dropout(op["gen_dropout"]),
            nn.Linear(op["gen_rnn_dim"], op["vocab_sz"]),
            nn.LogSoftmax(dim=-1))
        
        # pair-wise discriminator :
        self.dis_emb_layer = nn.Sequential(
            nn.Linear(op["vocab_sz"], op["emb_hid_dim"]),
            nn.Threshold(0.000001, 0),
            nn.Linear(op["emb_hid_dim"], op["emb_dim"]),
            nn.Threshold(0.000001, 0),
        )
        self.dis_rnn = nn.GRU(op["emb_dim"], op["enc_rnn_dim"])
        self.dis_lin = nn.Sequential(
            nn.Dropout(op["enc_dropout"]),
            nn.Linear(op["enc_rnn_dim"], op["enc_dim"]))
        
        # some useful constants :
        self.max_seq_len = op["max_seq_len"]
        self.vocab_sz = op["vocab_sz"]

    def forward(self, phrase, sim_phrase=None, train=False):
        """
        forward pass

        inputs :-

        phrase : given phrase , shape = (max sequence length, batch size)
        sim_phrase : (if train == True), shape = (max seq length, batch sz)
        train : if true teacher forcing is used to train the module

        outputs :-

        out : generated paraphrase, shape = (max sequence length, batch size, )
        enc_out : encoded generated paraphrase, shape=(batch size, enc_dim)
        enc_sim_phrase : encoded sim_phrase, shape=(batch size, enc_dim)

        """

        if sim_phrase is None:
            sim_phrase = phrase

        if train:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(utils.one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            emb_sim_phrase_gen = self.gen_emb(sim_phrase)
            out_rnn, _ = self.gen_rnn(
                torch.cat([enc_phrase, emb_sim_phrase_gen[:-1, :]], dim=0))
            out = self.gen_lin(out_rnn)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(utils.one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        else:

            # encode input phrase
            enc_phrase = self.enc_lin(
                self.enc_rnn(
                    self.emb_layer(utils.one_hot(phrase, self.vocab_sz)))[1])
            
            # generate similar phrase using teacher forcing
            words = []
            h = None
            for __ in range(self.max_seq_len):
                word, h = self.gen_rnn(enc_phrase, hx=h)
                word = self.gen_lin(word)
                words.append(word)
                word = torch.multinomial(torch.exp(word[0]), 1)
                word = word.t()
                enc_phrase = self.gen_emb(word)
            out = torch.cat(words, dim=0)

            # propagated from shared discriminator to calculate
            # pair-wise discriminator loss
            enc_sim_phrase = self.dis_lin(
                self.dis_rnn(
                    self.dis_emb_layer(utils.one_hot(sim_phrase,
                                                     self.vocab_sz)))[1])
            enc_out = self.dis_lin(
                self.dis_rnn(self.dis_emb_layer(torch.exp(out)))[1])

        enc_out.squeeze_(0)
        enc_sim_phrase.squeeze_(0)
        return out, enc_out, enc_sim_phrase
