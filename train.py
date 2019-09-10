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
# from torch.utils.tensorboard import SummaryWriter
import subprocess
import gc
# get command line arguments into args
parser = utils.make_parser()
args = parser.parse_args()
# writer = SummaryWriter()
torch.manual_seed(args.seed)

import time
import os

log_folder = 'logs'
save_folder = 'save'
folder = time.strftime("%d-%m-%Y_%H:%M:%S")

if args.start_from != 'None':
    folder = args.start_from.split('/')[-2]

subprocess.run(['mkdir', os.path.join(log_folder, folder)])
subprocess.run(['mkdir', os.path.join(save_folder, folder)])

file_scores = os.path.join(log_folder, folder, 'scores.txt')
file_loss = os.path.join(log_folder, folder, 'loss.txt')
file_sample = os.path.join(log_folder, folder, 'samples.txt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from misc.dataloader import Dataloader

# get dataloader
dataloader = Dataloader(args.input_json, args.input_ques_h5)
dataloader = dataloader.to(device)

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

def eval_batch(model, device, epoch, iter):
    
    model.eval()
    for batch in range(1):
        with torch.no_grad():    
            input_sentence, _, _, lengths = dataloader.next_batch_eval(150)
            input_sentence = input_sentence.to(device)
            lengths = lengths.to(device)

            probs, encoded_input = model(input_sentence, lengths)
            
            seq = model.decoder.sample(encoded_input)
            
            
            # local loss criterion
            loss = nn.CrossEntropyLoss()

            # compute local loss
            local_loss = loss(probs[:,:input_sentence.size()[1],:].permute(0, 2, 1), input_sentence)
                
            # get encoding from 
            encoded_output = model.encoder(probs[:,:input_sentence.size()[1],:])
                
            # compute global loss
            global_loss = model.JointEmbeddingLoss(encoded_output, encoded_input)
            global_loss *= 5
            
            seq = seq.long()
            sents = net_utils.decode_sequence(dataloader.getVocab(), seq)
            real_sents = net_utils.decode_sequence(dataloader.getVocab(), input_sentence.t())
            print("eval = ",gc.collect())
            coco, cocoRes = getObjsForScores(real_sents, sents)

            evalObj = COCOEvalCap(coco, cocoRes)

            evalObj.evaluate()

            f_score = open(file_scores, 'a')
            f_score.write(str(epoch) + '-' + str(iter) + '\n')

            for key in evalObj.eval:
                f_score.write(key + ' : ' + str(evalObj.eval[key]) + '\n')

            f_score.write('\n')
            f_score.close()

            f_loss = open(file_loss, 'a')
            f_loss.write(str(epoch) + '-' + str(iter) + '\n')
            f_loss.write('local loss : ' + str(local_loss.item()) + 'global loss : ' + str(global_loss.item()) + 'total loss : ' + str(local_loss.item() + global_loss.item()) + '\n')
            f_loss.close()

            f_sample = open(file_sample, 'a')
            
            idx = 1
            for r, s in zip(real_sents, sents):

                f_sample.write(str(epoch) + '-' + str(iter) + '\n')
                f_sample.write(str(idx) + '\nreal : ' + r + '\npred : ' + s + '\n\n')
                idx += 1

            f_sample.close()
            print("eval = ",gc.collect())
            torch.cuda.empty_cache()

def save_model(model, model_optim, epoch, iter, local_loss, global_loss):

    PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(iter) + '.tar')
    
    checkpoint = {
        'epoch' : epoch,
        'iter' : iter,
        'model_state_dict' : model.state_dict(), 
        'optimizer_state_dict' : model_optim.state_dict(),
        'local_loss' : local_loss, 
        'global_loss' : global_loss
    }

    torch.save(checkpoint, PATH)
    

def train_epoch(model, model_optim, device, epoch, log_per_iter=100, save_per_iter=100):
    
    n_batch = dataloader.getDataNum(1) // args.batch_size

    epoch_local_loss = 0
    epoch_global_loss = 0
    den = 0
    
    for batch in range(n_batch):
        
        if model.training == False:
            model.train()

        # zero all gradiants
        model_optim.zero_grad()
        
        # get new batch
        input_sentence, _, __, lengths = dataloader.next_batch(args.batch_size, gpuid=args.gpuid)
        # input_sentence = input_sentence[:5,:]
        # lengths = lengths[:5]
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        
        # forward propagation
        probs, encoded_input = model(input_sentence, lengths)
        gc.collect()
        '''
        probs : size - (seq_len + 2, batch_size, vocab_size + 1)
        encoded_input : size - (batch_size, emb_size)
        '''
        '''
        probs: (batch_size, seq_len + 1, vocab_size + 1)
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss()

         # compute local loss
        local_loss = loss(probs[:,:input_sentence.size()[1],:].permute(0, 2, 1), input_sentence)
        
        # get encoding from 
        encoded_output = model.encoder(probs[:,0:input_sentence.size()[1],:])
        
        # compute global loss
        global_loss = model.JointEmbeddingLoss(encoded_output, encoded_input)
        
        global_loss *= 5
        # take losses togather
        total_loss = local_loss + global_loss

        # backward propagation
        total_loss.backward()

        # update the parameters
        model_optim.step()

        
        # calculating losses
        epoch_local_loss += local_loss.item()
        epoch_global_loss += global_loss.item()
        den += encoded_input.size()[0]
        print(batch, end=' | ')
        if (batch + 1) % log_per_iter == 0:
            eval_batch(model, device, epoch, batch + 1)

        if (batch + 1) % save_per_iter == 0:
            save_model(model, model_optim, epoch, batch + 1, local_loss.item(), global_loss.item())

        gc.collect()
        torch.cuda.empty_cache()
    return epoch_local_loss, epoch_global_loss, encoded_input

print('Start the training...')

import math

decay_factor = math.exp(math.log(0.1) / (1500 * 1250))

model_optim = optim.RMSprop(model.parameters(), lr=0.0008) # for trial using default and no decay of lr

if args.start_from != 'None':
    print('loading model from ' + args.start_from)
    checkpoint = torch.load(args.start_from, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0


sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)

n_epoch = args.n_epoch

for epoch in range(start_epoch, start_epoch + n_epoch):
    
    local_loss, global_loss, encoded_input = train_epoch(model, model_optim, device, epoch)
    sheduler.step()

    n_batch = dataloader.getDataNum(1) // args.batch_size
    
    local_loss = local_loss / n_batch
    global_loss = global_loss / n_batch
    gc.collect()
    eval_batch(model, device, epoch + 1, 0)
    save_model(model, model_optim, epoch + 1, 0, local_loss, global_loss)

print('Done !!!')
