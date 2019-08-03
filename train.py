import torch
import torch.nn as nn
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
from misc.LanguageModel import crit as LanguageModelCriterion
import misc.utils as utils
import misc.net_utils as net_utils
from misc.FixedGRU import FixedGRU
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--input_ques_h5',default='data/quora_data_prepro.h5',help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json',default='data/quora_data_prepro.json',help='path to the json file containing additional info and vocab')

# starting point
parser.add_argument('--start_from', default='pretrained/model_epoch7.t7', help='path to a model checkpoint to initialize model weights from. Empty = don\'t')
parser.add_argument('--feature_type', default='VGG', help='VGG or Residual')

# # Model settings
parser.add_argument('--batch_size', type=int, default=150, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
parser.add_argument('--rnn_size', default=512, type=int, help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--input_encoding_size', type=int, default=512,help='the encoding size of each token in the vocabulary, and the image.')
parser.add_argument('--att_size', type=int, default=512, help='size of sttention vector which refer to k in paper')
parser.add_argument('--emb_size',type=int, default=512, help='the size after embeeding from onehot')
parser.add_argument('--rnn_layers',type=int, default=1, help='number of the rnn layer')

# Optimization
parser.add_argument('--optim',default='rmsprop',help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate',default=0.0008,help='learning rate', type=float)#0.0001,#0.0002,#0.005
parser.add_argument('--learning_rate_decay_start', default=5, type=int, help='at what epoch to start decaying learning rate? (-1 = dont)')#learning_rate_decay_start', 100,
parser.add_argument('--learning_rate_decay_every', type=int, default=5, help='every how many epoch thereafter to drop LR by half?')#-learning_rate_decay_every', 1500,
parser.add_argument('--momentum',type=float, default=0.9,help='momentum')
parser.add_argument('--optim_alpha',type=float, default=0.8,help='alpha for adagrad/rmsprop/momentum/adam')#optim_alpha',0.99
parser.add_argument('--optim_beta',type=float, default=0.999,help='beta used for adam')#optim_beta',0.995
parser.add_argument('--optim_epsilon',type=float, default=1e-8,help='epsilon that goes into denominator in rmsprop')
parser.add_argument('--max_iters', type=int, default=-1, help='max number of iterations to run for (-1 = run forever)')
parser.add_argument('--iterPerEpoch', default=1250, type=int)
parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of drop_prob_lm in the Language Model RNN')

# Evaluation/Checkpointing

parser.add_argument('--save', default='Results', help='save directory')
parser.add_argument('--checkpoint_dir', default='Results/checkpoints', help='folder to save checkpoints into (empty = this folder)')
parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--val_images_use', type=int, default=24800, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
parser.add_argument('--save_checkpoint_every', type=int, default=2500, help='how often to save a model checkpoint?')
parser.add_argument('--losses_log_every', type=int , default=200, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

# misc
parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
parser.add_argument('--id', default='1', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
parser.add_argument('--seed', type=int, default=1234, help='random number generator seed to use')
parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
parser.add_argument('--nGPU', type=int, default=3, help='Number of GPUs to use by default')

#text encoder
parser.add_argument('--txtSize', type=int, default=512,help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--cnn_dim',type=int, default=512,help='the encoding size of each token in the vocabulary, and the image.')

args = parser.parse_args()

torch.manual_seed(args.seed)
print(args)

subprocess.run(['mkdir', '-p', args.save])

import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

import os

logger_cmdline = setup_logger('Log_cmdline', os.path.join(args.save, 'Log_cmdline.txt'))
logger_cmdline.info(args)

subprocess.run(['mkdir', '-p', args.checkpoint_dir])

# from torch.utils.tensorboard import SummaryWriter

err_log_filename = os.path.join(args.save, 'ErrorProgress')
err_log = setup_logger('ErrorProgress' , err_log_filename)

errT_log_filename = os.path.join(args.save, 'ErrorProgress')
errT_log = setup_logger('ErrorTProgress', errT_log_filename)

lang_stats_filename = os.path.join(args.save, 'language_statstics')
lang_stats_log = os.path.join('language_statstics', lang_stats_filename)

from misc.dataloader import Dataloader

dataloader = Dataloader(args.input_json, args.input_ques_h5)

class Model(nn.Module):

    def __init__(self):
        
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
        
        self.encoder = DocumentCNN(self.vocab_size + 1, args.txtSize, 0, 1, args.cnn_dim)
        
        self.decoder = LanguageModel(self.input_encoding_size, self.rnn_size, self.seq_length, self.vocab_size, num_layers=self.num_layers, dropout=self.dropout)
        
        self.crit = LanguageModelCriterion()

    
    def forward(self, inputs, )
