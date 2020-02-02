import torch
import argparse


def one_hot(t, c):
    return torch.zeros(*t.size(), c, device=t.device).scatter_(-1, t.unsqueeze(-1), 1)


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_ques_h5',default='data/quora_data_prepro.h5',help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_json',default='data/quora_data_prepro.json',help='path to the json file containing additional info and vocab')

    # starting point
    parser.add_argument('--start_from', default='None', help='path to a model checkpoint to initialize model weights from. Empty = don\'t')

    # # Model settings
    parser.add_argument('--model', default='EDLPS', help='which model to use? EDL|EDP|EDLP|EDLPS|EDLPG|EDLPGS|EDG|EDPG')
    parser.add_argument('--batch_size', type=int, default=150, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--input_encoding_size', type=int, default=512,help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_size', type=int, default=512, help='size of sttention vector which refer to k in paper')
    parser.add_argument('--emb_size',type=int, default=512, help='the size after embeeding from onehot')
    parser.add_argument('--rnn_layers',type=int, default=1, help='number of the rnn layer')
    parser.add_argument('--train_dataset_len', type=int, default=100000, help='length of train dataset')
    parser.add_argument('--val_dataset_len', type=int, default=30000, help='length of validation dataset')

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
    parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs during training')

    # Evaluation/Checkpointing

    parser.add_argument('--save', default='Results', help='save directory')
    parser.add_argument('--checkpoint_dir', default='Results/checkpoints', help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--val_images_use', type=int, default=24800, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_every', type=int, default=500, help='how often to save a model checkpoint?')
    parser.add_argument('--log_every', type=int , default=100, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

    # misc
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--name', default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--seed', type=int, default=1234, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--nGPU', type=int, default=3, help='Number of GPUs to use by default')

    #text encoder
    parser.add_argument('--emb_dim',type=int, default=512,help='dim of word embedding')
    parser.add_argument('--emb_hid_dim', type=int, default=256,help='hidden dim of word embedding')
    parser.add_argument('--enc_dropout', type=float, default=0.5,help='dropout for encoder module')
    parser.add_argument('--enc_rnn_dim', default=512, type=int, help='size of the rnn in number of hidden nodes in each layer of gru in encoder')
    parser.add_argument('--enc_dim', type=int, default=512,help='size of the encoded sentence')
    parser.add_argument('--gen_rnn_dim', default=512, type=int, help='size of the rnn in number of hidden nodes in each layer of lstm in generator')
    parser.add_argument('--gen_dropout',type=float, default=0.5,help='dropout for generator module')

    return parser
