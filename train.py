import torch
import torch.nn as nn
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
from misc.LanguageModel import crit as LanguageModelCriterion
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from model import Model
from pycocoevalcap.eval import COCOEvalCap
from torch.utils.tensorboard import SummaryWriter
import subprocess

# get command line arguments into args
parser = utils.make_parser()
args = parser.parse_args()
writer = SummaryWriter()
torch.manual_seed(args.seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from misc.dataloader import Dataloader

# get dataloader
dataloader = Dataloader(args.input_json, args.input_ques_h5)

# make model
model = Model(args, dataloader)

def getObjsForScores(real_sents, pred_sents):
    class coco:

        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption' : sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]


    return coco(real_sents), coco(pred_sents)


def train_epoch(model, model_optim, device):
    
    n_batch = dataloader.getDataNum(1) // args.batch_size
    n_batch = 3

    epoch_local_loss = 0
    epoch_global_loss = 0
    den = 0
    
    for batch in range(n_batch):

        # zero all gradiants
        model_optim.zero_grad()
        
        # get new batch
        input_sentence, _, __ = dataloader.next_batch(args.batch_size, gpuid=args.gpuid)
        # input_sentence = input_sentence[:5,:]
        input_sentence = input_sentence.to(device)
        
        # forward propagation
        probs, encoded_input = model(input_sentence)
        '''
        probs : size - (seq_len + 2, batch_size, vocab_size + 1)
        encoded_input : size - (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss()

         # compute local loss
        local_loss = loss(probs[1:input_sentence.size()[1] + 1].permute(1, 2, 0), input_sentence)
        
        # get encoding from 
        encoded_output = model.encoder(probs[1:input_sentence.size()[1] + 1].permute(1, 0, 2))
        
        # compute global loss
        global_loss = model.JointEmbeddingLoss(encoded_output, encoded_input)

        # take losses togather
        total_loss = local_loss + global_loss

        # backward propagation
        total_loss.backward()

        # update the parameters
        model_optim.step()

        # get sentence from encodings
        seq , _ = model.decoder.sample(encoded_input)
        sents = net_utils.decode_sequence(dataloader.getVocab(), seq)
        real_sents = net_utils.decode_sequence(dataloader.getVocab(), input_sentence.t())
        # printing 10 samples
        for s in sents[:10]:
            print(s)

        # calculating losses
        epoch_local_loss += local_loss
        epoch_global_loss += global_loss
        den += encoded_input.size()[0]

        coco, cocoRes = getObjsForScores(real_sents, sents)

        evalObj = COCOEvalCap(coco, cocoRes)

        evalObj.evaluate()

        for key in evalObj.eval:
            print(key, ':', evalObj.eval[key], end=' ')
            print()
            writer.add_scalar(key, evalObj.eval[key], model.count)

        writer.add_scalar('Loss', total_loss / encoded_input.size()[0], model.count)
        model.count += 1

    return epoch_local_loss / den, epoch_global_loss / den, encoded_input

print('Start the training...')
model.train()

import math

decay_factor = math.exp(math.log(0.1) / (1500 * 1250))

model_optim = optim.RMSprop(model.parameters(), lr=0.0008) # for trial using default and no decay of lr
sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)

n_epoch = args.n_epoch

for epoch in range(n_epoch):
    
    local_loss, global_loss, encoded_input = train_epoch(model, model_optim, device)
    sheduler.step()
    print(local_loss.item(), global_loss.item())

print('Done !!!')
