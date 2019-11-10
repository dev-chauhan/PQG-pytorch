import torch
import torch.nn as nn
import argparse
import misc.utils as utils
import torch.utils.data as Data
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from misc.LanguageModel import layer as LanguageModel

parser = utils.make_parser()
parser.add_argument('--start_from_file', type=int)
parser.add_argument('--end_to_file', type=int)
args = parser.parse_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from misc.dataloader import Dataloader

# get dataloader
data = Dataloader(args.input_json, args.input_ques_h5)

test_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len, args.train_dataset_len + args.val_dataset_len)), batch_size = args.batch_size, shuffle=True)


import math

decay_factor = math.exp(math.log(0.1) / (1500 * 1250))


import misc.net_utils as net_utils
from pycocoevalcap.eval import COCOEvalCap

def getObjsForScores(real_sents, pred_sents):
    class coco:

        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption' : sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]


    return coco(real_sents), coco(pred_sents)

import time
import os

if args.start_from != 'None':
    print('loading model from ' + args.start_from)
    

folder_name = args.start_from.split('/')[-1]

s = args.start_from_file
e = args.end_to_file

import subprocess

subprocess.run(['mkdir', os.path.join('result', folder_name)])

for i in range(s, e+1):
    print("Evaluating model number", i)
    encoder = DocumentCNN(data.getVocabSize(), args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)

    generator = LanguageModel(args.input_encoding_size, args.rnn_size, data.getSeqLength(), data.getVocabSize(), num_layers=args.rnn_layers, dropout=args.drop_prob_lm)

    load_file = os.path.join(args.start_from, str(i) + '_-1.tar')
    file_sample = os.path.join('result' ,folder_name, str(i))
    file_score = file_sample + '-score'

    checkpoint = torch.load(load_file, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])

    encoder.eval()
    generator.eval()

    pred_sent = []
    gt_sent = []
    idx = 1
    encoder = encoder.to(device)
    generator = generator.to(device)
    encoder = nn.DataParallel(encoder)
    generator = nn.DataParallel(generator)


    with torch.no_grad():
        for input_sentence, lengths, sim_seq, _, _ in test_loader:
            input_sentence = input_sentence.to(device)
            lengths = lengths.to(device)
            sim_seq = sim_seq.to(device)

            encoded_input = encoder(utils.one_hot(input_sentence, data.getVocabSize()))
            
            seq_logprob = generator(encoded_input, teacher_forcing=False)
            seq_prob = torch.exp(seq_logprob)
            seq = net_utils.prob2pred(seq_logprob)
            # local loss criterion
            loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

            # compute local loss
            local_loss = loss(seq_logprob.permute(0, 2, 1), sim_seq)
            
            # get encoding from 
            encoded_output = encoder(seq_prob)
            encoded_sim = encoder(utils.one_hot(sim_seq, data.getVocabSize()))
            # compute global loss
            global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
            
            seq = seq.long()
            sents = net_utils.decode_sequence(data.ix_to_word, seq)
            real_sents = net_utils.decode_sequence(data.ix_to_word, input_sentence)
            out_sents = net_utils.decode_sequence(data.ix_to_word, sim_seq)
            pred_sent = pred_sent + sents
            gt_sent = gt_sent + out_sents

            f_sample = open(file_sample + '.txt', 'a')
            
            for r, s, t in zip(real_sents,out_sents, sents):

                f_sample.write(str(idx) + '\nreal : ' + r + '\nout : ' + s + '\npred : ' + t + '\n\n')
                idx += 1

            f_sample.close()
            torch.cuda.empty_cache()

    print('prediction completed...')

    coco, cocoRes = getObjsForScores(gt_sent, pred_sent)

    evalObj = COCOEvalCap(coco, cocoRes)

    evalObj.evaluate()
    f_score = open(file_score + '.txt', 'w')

    for key in evalObj.eval:
        f_score.write(key + ' : ' + str(evalObj.eval[key]) + '\n')

    f_score.close()

print('Done !!!')