import torch
import misc.utils as utils


def decode_sequence(ix_to_word, seq):
    N, D = seq.size()[0], seq.size()[1]
    out = []
    EOS_flag = False
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if int(ix.item()) not in ix_to_word:
                print("UNK token ", str(ix.item()))
                word = ix_to_word[len(ix_to_word) - 1]
            else:
                word = ix_to_word[int(ix.item())]
            if word == '<EOS>':
                txt = txt + ' '
                txt = txt + word
                break
            if word == '<SOS>':
                txt = txt + '<SOS>'
                continue
            if j > 0:
                txt = txt + ' '
            txt = txt + word
        out.append(txt)
    return out

def prob2pred(prob):

    return torch.multinomial(torch.exp(prob.view(-1, prob.size(-1))), 1).view(prob.size(0), prob.size(1))

def JointEmbeddingLoss(feature_emb1, feature_emb2):
       
    batch_size = feature_emb1.size()[0]

    return torch.sum(
        torch.clamp(
            torch.mm(feature_emb1, feature_emb2.t()) - torch.sum(feature_emb1 * feature_emb2, dim=-1) + 1,
            min=0.0
        )
    ) / (batch_size * batch_size)
