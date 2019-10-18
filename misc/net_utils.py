import misc.utils as utils
import torch

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
            if word == '<PAD>':
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
    # loss = 0
    # for i in range(batch_size):
    #     label_score = torch.dot(feature_emb1[i], feature_emb2[i])
    #     for j in range(batch_size):
    #         cur_score = torch.dot(feature_emb2[i], feature_emb1[j])
    #         score = cur_score - label_score + 1
    #         if 0 < score.item():
    #             loss += max(0, cur_score - label_score + 1)

    # denom = batch_size * batch_size
    
    return torch.sum(torch.clamp(torch.mm(feature_emb1, feature_emb2.t()) - torch.sum(feature_emb1 * feature_emb2, dim=-1) + 1, min=0.0)) / (batch_size * batch_size)

def clone_list(lst):
    new = []
    for t in lst:
        new.append(t)
    return new

def language_eval(predictions, id):
    out_struct = {
        "val_predictions": predictions
    }
    utils.write_json('coco-caption/val'+id+'.json', out_struct)
    import subprocess
    subprocess.run(['./misc/call_python_caption_eval.sh', 'val' + id + '.json'])
    result_struct = utils.read_json('coco-caption/val'+id+'.json_out.json')
    return result_struct