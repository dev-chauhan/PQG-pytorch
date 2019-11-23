import torch
import torch.nn as nn
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
# from model import Model
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
folder = args.name + folder

if args.start_from != 'None':
    folder = args.start_from.split('/')[-2]

if args.start_from != 'None':
    writer_train = SummaryWriter(os.path.join(log_folder, folder + 'train' + args.name))
    writer_val = SummaryWriter(os.path.join(log_folder, folder + 'val'+ args.name))
    writer_score = SummaryWriter(os.path.join(log_folder, folder + 'score'+ args.name))
else:
    writer_train = SummaryWriter(os.path.join(log_folder, folder + 'train'))
    writer_val = SummaryWriter(os.path.join(log_folder, folder + 'val'))
    writer_score = SummaryWriter(os.path.join(log_folder, folder + 'score'))

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
# import itertools
# test_loader_iter = itertools.cycle(test_loader)
# make model


encoder = DocumentCNN(data.getVocabSize(), args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)

generator = LanguageModel(args.input_encoding_size, args.rnn_size, data.getSeqLength(), data.getVocabSize(), num_layers=args.rnn_layers, dropout=args.drop_prob_lm)

iter_per_epoch = (args.train_dataset_len + args.batch_size - 1)/ args.batch_size

def getObjsForScores(real_sents, pred_sents):
    class coco:

        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption' : sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]


    return coco(real_sents), coco(pred_sents)

def eval_batch(encoder, generator, device, log_idx):
    
    encoder.eval()
    generator.eval()

    with torch.no_grad():
        acc_local_loss = 0
        acc_global_loss = 0
        acc_seq = []
        acc_real = []
        acc_out = []
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
            
            acc_local_loss += local_loss.item()
            acc_global_loss += global_loss.item()

            seq = seq.long()
            acc_seq += net_utils.decode_sequence(data.ix_to_word, seq)
            acc_real += net_utils.decode_sequence(data.ix_to_word, input_sentence)
            acc_out += net_utils.decode_sequence(data.ix_to_word, sim_seq)
        
        coco, cocoRes = getObjsForScores(acc_out, acc_seq)

        evalObj = COCOEvalCap(coco, cocoRes)

        evalObj.evaluate()

        for key in evalObj.eval:
            writer_score.add_scalar(key, evalObj.eval[key], log_idx)

        writer_val.add_scalar('local_loss', acc_local_loss/len(test_loader), log_idx)
        writer_val.add_scalar('global_loss', acc_global_loss/len(test_loader), log_idx)
        writer_val.add_scalar('total_loss', (acc_local_loss + acc_global_loss)/len(test_loader), log_idx)

        f_sample = open(file_sample + str(log_idx) + '.txt', 'w')
        
        idx = 1
        for r, s, t in zip(acc_real,acc_out, acc_seq):

            f_sample.write(str(idx) + '\nreal : ' + r + '\nout : ' + s + '\npred : ' + t + '\n\n')
            idx += 1

        f_sample.close()
        torch.cuda.empty_cache()

def save_model(encoder, generator, model_optim, epoch, it, local_loss, global_loss):

    PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(it) + '.tar')
    
    checkpoint = {
        'epoch' : epoch,
        'iter' : it,
        'encoder_state_dict' : encoder.state_dict(), 
        'generator_state_dict' : generator.state_dict(),         
        'optimizer_state_dict' : model_optim.state_dict(),
        'local_loss' : local_loss, 
        'global_loss' : global_loss
    }

    torch.save(checkpoint, PATH)
    

def train_epoch(encoder, generator, model_optim, device, epoch, log_per_iter=args.log_every, save_per_iter=args.save_every, log_idx=0):
    
    n_batch = (len(data) - 30000) // args.batch_size

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    acc_l_loss = 0
    acc_g_loss = 0
    for batch in train_loader:
        
        encoder.train()
        generator.train()

        # zero all gradiants
        model_optim.zero_grad()
        
        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch
        
        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len  = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

         # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)
        
        # get encoding from
        '''([N, d1, C])''' 
        encoded_output = encoder(probs)
        encoded_sim = encoder(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
        
        # take losses togather
        total_loss = local_loss + global_loss

        # backward propagation
        total_loss.backward()

        # update the parameters
        model_optim.step()

        
        # calculating losses
        epoch_local_loss += local_loss.item()
        acc_l_loss += local_loss.item()
        epoch_global_loss += global_loss.item()
        acc_g_loss += global_loss.item()
        # print(idx, end='|')
        if (idx + 1) % log_per_iter == 0:
            writer_train.add_scalar('local_loss', acc_l_loss /log_per_iter, log_idx)
            writer_train.add_scalar('global_loss', acc_g_loss /log_per_iter, log_idx)
            writer_train.add_scalar('total_loss', (acc_l_loss  + acc_g_loss ) / log_per_iter, log_idx)
            acc_l_loss = 0
            acc_g_loss = 0
            eval_batch(encoder, generator, device, log_idx)
            log_idx += 1

        if (idx + 1) % save_per_iter == 0:
            save_model(encoder, generator, model_optim, epoch, idx, local_loss.item(), global_loss.item())

        torch.cuda.empty_cache()
        idx+=1
        
    return epoch_local_loss, epoch_global_loss, log_idx

print('Start the training...')

import math

decay_factor = math.exp(math.log(0.1) / (1500 * 1250))

model_optim = optim.RMSprop(list(encoder.parameters()) + list(generator.parameters()), lr=args.learning_rate) # for trial using default and no decay of lr

if args.start_from != 'None':
    print('loading model from ' + args.start_from)
    checkpoint = torch.load(args.start_from, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    # model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    lr = args.learning_rate * (decay_factor ** (start_epoch))
    for g in model_optim.param_groups:
        g['lr'] = lr
    print("learning rate = ", lr)
else:
    start_epoch = 0


sheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=1, gamma=decay_factor)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

encoder = encoder.to(device)
generator = generator.to(device)
n_epoch = args.n_epoch
log_idx = 0
for epoch in range(start_epoch, start_epoch + n_epoch):
    
    local_loss, global_loss, log_idx = train_epoch(encoder, generator, model_optim, device, epoch, log_idx = log_idx)
    sheduler.step()

    n_batch = data.getDataNum(1) // args.batch_size
    
    local_loss = local_loss / n_batch
    global_loss = global_loss / n_batch
    
    writer_train.add_scalar('local_loss', local_loss, log_idx)
    writer_train.add_scalar('global_loss', global_loss, log_idx)
    writer_train.add_scalar('total_loss', local_loss + global_loss, log_idx)
    
    eval_batch(encoder, generator, device, log_idx)
    log_idx += 1
    save_model(encoder, generator, model_optim, epoch + 1, -1, local_loss, global_loss)

print('Done !!!')
