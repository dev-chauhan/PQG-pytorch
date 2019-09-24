import torch
import torch.nn as nn
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from model import Model
from pycocoevalcap.eval import COCOEvalCap
from tensorboardX import SummaryWriter
import subprocess
import torch.utils.data as Data

# get command line arguments into args
parser = utils.make_parser()
args = parser.parse_args()

torch.manual_seed(args.seed)

import time
import os

log_folder = 'logs'
save_folder = 'save'
folder = time.strftime("%d-%m-%Y_%H:%M:%S")

if args.start_from != 'None':
    folder = args.start_from.split('/')[-2]

writer_train = SummaryWriter(os.path.join(log_folder, 'train'))
writer_val = SummaryWriter(os.path.join(log_folder, 'val'))
writer_score = SummaryWriter(os.path.join(log_folder, 'score'))

subprocess.run(['mkdir', os.path.join(save_folder, folder)])
subprocess.run(['mkdir', os.path.join('samples', folder)])

# file_scores = os.path.join(log_folder, folder, 'scores.txt')
# file_loss = os.path.join(log_folder, folder, 'loss.txt')
file_sample = os.path.join('samples', folder, 'samples')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from misc.dataloader import Dataloader

# get dataloader
data = Dataloader(args.input_json, args.input_ques_h5)

train_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len)), batch_size = args.batch_size, shuffle=True)
test_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len, args.train_dataset_len + args.val_dataset_len)), batch_size = args.batch_size, shuffle=True)
test_loader_iter = iter(test_loader)
# make model
model = Model(args, data)
iter_per_epoch = (args.train_dataset_len + args.batch_size - 1)/ args.batch_size

def getObjsForScores(real_sents, pred_sents):
    class coco:

        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption' : sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]


    return coco(real_sents), coco(pred_sents)

def eval_batch(model, device, epoch, it):
    
    model.eval()
    
    with torch.no_grad():    
        input_sentence, lengths, _, _, _ = next(test_loader_iter)
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)

        probs, encoded_input = model(input_sentence, lengths)
        
        seq = model.decoder.sample(encoded_input)
        
        
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

        # compute local loss
        local_loss = loss(probs.permute(0, 2, 1), input_sentence)
            
        # get encoding from 
        encoded_output = model.encoder(probs)

        # compute global loss
        global_loss = model.JointEmbeddingLoss(encoded_output, encoded_input)
        global_loss *= 5
        
        seq = seq.long()
        sents = net_utils.decode_sequence(data.ix_to_word, seq)
        real_sents = net_utils.decode_sequence(data.ix_to_word, input_sentence)
        
        coco, cocoRes = getObjsForScores(real_sents, sents)

        evalObj = COCOEvalCap(coco, cocoRes)

        evalObj.evaluate()

        for key in evalObj.eval:
            writer_score.add_scalar(key, evalObj.eval[key], epoch * iter_per_epoch + it)

        writer_val.add_scalar('local_loss', local_loss.item(), epoch * iter_per_epoch + it)
        writer_val.add_scalar('global_loss', global_loss.item(), epoch * iter_per_epoch + it)
        writer_val.add_scalar('total_loss', local_loss.item() + global_loss.item(), epoch * iter_per_epoch + it)

        f_sample = open(file_sample + str(epoch) + '_' + str(it) + '.txt', 'w')
        
        idx = 1
        for r, s in zip(real_sents, sents):

            f_sample.write(str(idx) + '\nreal : ' + r + '\npred : ' + s + '\n\n')
            idx += 1

        f_sample.close()
        torch.cuda.empty_cache()

def save_model(model, model_optim, epoch, it, local_loss, global_loss):

    PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(it) + '.tar')
    
    checkpoint = {
        'epoch' : epoch,
        'iter' : it,
        'model_state_dict' : model.state_dict(), 
        'optimizer_state_dict' : model_optim.state_dict(),
        'local_loss' : local_loss, 
        'global_loss' : global_loss
    }

    torch.save(checkpoint, PATH)
    

def train_epoch(model, model_optim, device, epoch, log_per_iter=args.log_every, save_per_iter=args.log_every):
    
    n_batch = (len(data) - 30000) // args.batch_size

    epoch_local_loss = 0
    epoch_global_loss = 0
    den = 0
    idx = 0
    for batch in train_loader:
        
        if model.training == False:
            model.train()

        # zero all gradiants
        model_optim.zero_grad()
        
        # get new batch
        input_sentence, lengths, _, _, _ = batch
        
        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        
        # forward propagation
        probs, encoded_input = model(input_sentence, lengths)
        
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

         # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(probs.permute(0, 2, 1), input_sentence)
        
        # get encoding from
        '''([N, d1, C])''' 
        encoded_output = model.encoder(probs)
        
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
        print(idx, end='|')
        if (idx + 1) % log_per_iter == 0:
            writer_train.add_scalar('local_loss', local_loss.item(), epoch * iter_per_epoch + idx)
            writer_train.add_scalar('global_loss', global_loss.item(), epoch * iter_per_epoch + idx)
            writer_train.add_scalar('total_loss', local_loss.item() + global_loss.item(), epoch * iter_per_epoch + idx)

            eval_batch(model, device, epoch, idx)

        if (idx + 1) % save_per_iter == 0:
            save_model(model, model_optim, epoch, idx, local_loss.item(), global_loss.item())

        torch.cuda.empty_cache()
        idx+=1
        
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
    lr = 0.0008 * (decay_factor ** (start_epoch))
    for g in model_optim.param_groups:
        g['lr'] = lr
    print("learning rate = ", lr)
else:
    start_epoch = 0


sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)

n_epoch = args.n_epoch

for epoch in range(start_epoch, start_epoch + n_epoch):
    
    local_loss, global_loss, encoded_input = train_epoch(model, model_optim, device, epoch)
    sheduler.step()

    n_batch = data.getDataNum(1) // args.batch_size
    
    local_loss = local_loss / n_batch
    global_loss = global_loss / n_batch
    
    eval_batch(model, device, epoch + 1, -1)
    save_model(model, model_optim, epoch + 1, -1, local_loss, global_loss)

print('Done !!!')
